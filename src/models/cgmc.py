"""IGMC modules"""
import numpy as np
import torch as th
import torch.nn as nn
import dgl


from dgl.nn.pytorch.conv import RelGraphConv,  EGATConv
from models.cgat import CGATConv

class CGMC(nn.Module):

    def __init__(self, in_nfeats, out_nfeats,
                 in_efeats,
                 node_features='onehot', num_heads=4, review =True, rating=False, timestamp=False,
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

        etypes = [0,1,2,3,4,5,6,7]
        num_relations = len(etypes)
        self.conv0 = CGATConv(in_node_feats=4, in_edge_feats=in_efeats,
                                out_node_feats=out_nfeats, out_edge_feats=num_relations,
                                review=review, rating=rating, timestamp=timestamp,
                                num_heads=num_heads)

        conv1 = RelGraphConv(in_feat=out_nfeats, out_feat=32, num_rels=num_relations, num_bases=num_relations, self_loop=True)
        conv2 = RelGraphConv(in_feat=32, out_feat=32, num_rels=num_relations, num_bases=num_relations, self_loop=True)
        conv3 = RelGraphConv(in_feat=32, out_feat=32, num_rels=num_relations, num_bases=num_relations, self_loop=True)
        self.rgcns = nn.ModuleList([conv1, conv2, conv3])

        self.lin1 = nn.Linear((out_nfeats*4)*2, 128) # concat user, item vector
        self.dropout1 = nn.Dropout(0.5)
        self.lin2 = nn.Linear(128, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

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
        e = graph.edata['feature'].float()

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
        e = th.sigmoid(e)
        states.append(x)
        u, v = graph.edges()
        device = graph.device
        mask = graph.edata['edge_mask']
        edge_weight = e.squeeze(1)

        graph = dgl.remove_edges(graph, graph.edges()[0])

        for i in range(8):
            graph.add_edges(
                u=u[mask==1],
                v=v[mask==1],
                data={
                    'edge_mask' : (edge_weight[:,i]*mask)[mask==1],
                    'etype' : th.tensor([i]*len(u)).type(th.int32).to(device)[mask==1],
                }
            )

        for conv in self.rgcns:
            x = conv(g=graph, feat=x, etypes=graph.edata['etype'], norm=graph.edata['edge_mask'].unsqueeze(1))
            x = self.elu(x)
            states.append(x)

        states = th.cat(states, 1)
        x = th.cat([states[users], states[items]], 1)
        x = th.relu(self.lin1(x))
        x = self.dropout1(x)
        x = self.lin2(x)
        x = th.sigmoid(x)
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