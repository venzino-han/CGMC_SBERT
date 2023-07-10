"""IGMC modules"""
import numpy as np
import torch as th
import torch.nn as nn
import dgl

from models.cgat import CGATConv
from models.edge_fusion_cgn import EdgeFusionGCN

class CGMC(nn.Module):

    def __init__(self, in_nfeats, out_nfeats,
                 in_efeats,
                 node_features='onehot', num_heads=4, review =True, rating=False, timestamp=False,
                 relation=True,
                 regression=True, edge_dropout=0.2,):
                 
        super(CGMC, self).__init__()
        self.regression = regression
        self.edge_dropout = edge_dropout
        self.review = review
        self.rating = rating
        self.timestamp = timestamp
        self.node_features = node_features
        self.in_nfeats = in_nfeats
        print('node_features :', node_features)
        self.elu = nn.ELU()
        self.leakyrelu = th.nn.LeakyReLU()

        print(review, rating, timestamp)

        num_relations = 8
        self.conv0 = CGATConv(in_node_feats=4, in_edge_feats=in_efeats,
                                out_node_feats=out_nfeats, out_edge_feats=num_relations,
                                review=review, rating=rating, timestamp=timestamp,
                                num_heads=num_heads)

        conv1 = EdgeFusionGCN(in_node_feats=out_nfeats, 
                              out_node_feats=32, 
                              in_edge_feats=num_relations,
                              in_edge_types=3,
                              relation=relation,
        ) 
        # conv2 = EdgeFusionGCN(in_node_feats=32, 
        #                       out_node_feats=32, 
        #                       in_edge_feats=8,
        #                       in_edge_types=3,
        #                       first=False,
        #                       )
        self.rgcns = nn.ModuleList([conv1])

        self.lin1 = nn.Linear((out_nfeats*2)*2, 128) # concat user, item vector
        self.dropout1 = nn.Dropout(0.5)
        self.lin2 = nn.Linear(128, 1)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def get_parameters(self):
        parameters_dict = {}
        n, p  = self.lin1.named_parameters()
        parameters_dict[n] = p
        n, p  = self.lin2.named_parameters()
        parameters_dict[n] = p
        return parameters_dict


    def forward(self, graph):
        """ graph : subgraph """
        graph = edge_drop(graph, self.edge_dropout, self.training)

        graph.edata['norm'] = graph.edata['edge_mask']
        node_x = graph.ndata['x'].float()

        states = []
        n = len(graph.nodes())

        # get user, item idx --> vector
        users = graph.ndata['nlabel'][:, 0] == 1
        items = graph.ndata['nlabel'][:, 1] == 1

        x = node_x # original
        try:
            e = graph.edata['feature'].float()
        except:
            e = None

        rating, ts = None, None
        if self.rating == True:
            rating = graph.edata['label']
        if self.timestamp == True:
            ts = graph.edata['ts']

        x, e = self.conv0(graph=graph, nfeats=x, efeats=e,
                          norm=graph.edata['edge_mask'].unsqueeze(1),
                          rating=rating, timestamp=ts
                          )
        x = self.elu(x)
        cos_sim = self.cos(x[users], x[items])
        
        e = th.sigmoid(e)
        states.append(x)
        edge_vector = e.squeeze(1)

        for conv in self.rgcns:
            x = conv(graph=graph, 
                     nfeats=x, 
                     etypes=graph.edata['etype'], 
                     mask=graph.edata['edge_mask'].unsqueeze(1),
                     efeats=edge_vector,
                     )
            x = self.elu(x)
            states.append(x)

        states = th.cat(states, 1)
        # cos_sim = th.cat([cos_sim[users], cos_sim[items]], 0)
        x = th.cat([states[users], states[items]], 1)
        x = th.relu(self.lin1(x))
        x = self.dropout1(x)
        x = self.lin2(x)
        x = th.sigmoid(x)
        if self.training:
            return x[:, 0], cos_sim
        return x[:, 0]


def edge_drop(graph, edge_dropout=0.2, training=True):
    assert edge_dropout >= 0.0 and edge_dropout <= 1.0, 'Invalid dropout rate.'

    if not training:
        return graph

    # set edge mask to zero in directional mode
    src, _ = graph.edges()
    to_drop = src.new_full((graph.number_of_edges(), ), edge_dropout, dtype=th.float)
    to_drop = th.bernoulli(to_drop).to(th.bool)
    graph.edata['edge_mask'][to_drop] = 0

    return graph