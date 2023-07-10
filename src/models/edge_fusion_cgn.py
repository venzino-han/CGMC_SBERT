"""custom graph attention networks with fully valuable edges (EGAT)."""
import torch as th
from torch import nn
from torch.nn import init

from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch.conv import RelGraphConv
from dgl.base import DGLError

from copy import copy

class EdgeFusionGCN(nn.Module):

    def __init__(self,
                 in_node_feats,
                 out_node_feats,
                 in_edge_feats,
                 in_edge_types,
                 first=True,
                 relation=True,
                 ):

        super().__init__()
        self.first=first
        self._in_node_feats = in_node_feats
        self._out_node_feats = out_node_feats
        self._in_edge_feats = in_edge_feats
        self.relation = relation

        self.fc = nn.Linear(out_node_feats, out_node_feats, bias=False)
        if relation:
            self.etype_conv = RelGraphConv(in_feat=in_node_feats, out_feat=out_node_feats, num_rels=in_edge_types, num_bases=in_edge_types, self_loop=True)
            self.fc = nn.Linear(out_node_feats*2, out_node_feats, bias=False)
        self.embedding_conv = RelGraphConv(in_feat=in_node_feats, out_feat=out_node_feats, num_rels=in_edge_feats, num_bases=in_edge_feats, self_loop=True)

    def forward(self, graph, nfeats, etypes, mask, efeats):                
        device = graph.device
        graph = copy(graph)
        
        #social
        x_etype = self.etype_conv(g=graph, feat=nfeats, etypes=etypes, norm=mask)


        if self.first:
            mask = graph.edata['edge_mask']
            u, v = graph.edges()
            # embedding to etype
            for i in range(self._in_edge_feats):
                graph.add_edges(
                    u=u[mask==1],
                    v=v[mask==1],
                    data={
                        'edge_mask' : (efeats[:,i]).to(device)[mask==1],
                        'emb_etype' : th.tensor([i]*len(u)).type(th.int32).to(device)[mask==1],
                    }
                )

        x_embedding = self.embedding_conv(g=graph, feat=nfeats, etypes=graph.edata['emb_etype'], norm=graph.edata['edge_mask'].unsqueeze(1))

        if self.relation:
            x = th.concat([x_etype, x_embedding], dim=1)
        else:
            x = x_embedding
        x = self.fc(x)

        return x

