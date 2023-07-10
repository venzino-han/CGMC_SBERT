"""custom graph attention networks with fully valuable edges (EGAT)."""
import torch as th
from torch import nn
from torch.nn import init

from dgl.nn.functional import edge_softmax
from dgl.base import DGLError

class CGATConv(nn.Module):

    def __init__(self,
                 in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 num_heads,
                 rating=False,
                 timestamp=False,
                 review=True,
                 mpnn_type='node',
                 bias=True):

        super().__init__()
        self.rating = rating
        self.timestamp = timestamp
        self.review = review

        _in_edge_feats = 0
        if rating == True:
            _in_edge_feats = _in_edge_feats + 1
        if timestamp == True:
            _in_edge_feats = _in_edge_feats + 1
        if review == True:
            _in_edge_feats = _in_edge_feats + in_edge_feats

        self._num_heads = num_heads
        self._in_node_feats = in_node_feats
        self._out_node_feats = out_node_feats
        self._in_edge_feats = _in_edge_feats
        self._out_edge_feats = out_edge_feats
        self.mpnn_type = mpnn_type
        self._set_message_func()

        print('fc_fij', _in_edge_feats)

        self.fc_fij = nn.Linear(_in_edge_feats, out_edge_feats*num_heads, bias=False)

        self.fc_ni = nn.Linear(in_node_feats, out_edge_feats*num_heads, bias=False)
        self.fc_nj = nn.Linear(in_node_feats, out_edge_feats*num_heads, bias=False)

        # feature 종류별로 추가
        self.fc_edges = nn.Linear(out_edge_feats*num_heads*3, out_edge_feats*num_heads, bias=False)
        self.fc_attn = nn.Linear(out_edge_feats, num_heads, bias=False)

        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_edge_feats,)))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        gain = init.calculate_gain('relu')
        init.xavier_normal_(self.fc_fij.weight, gain=gain)
        init.xavier_normal_(self.fc_ni.weight, gain=gain)
        init.xavier_normal_(self.fc_nj.weight, gain=gain)
        self.fc_attn.reset_parameters()
        self.fc_edges.reset_parameters()
        init.constant_(self.bias, 0)

    def _set_message_func(self):
        if self.mpnn_type == 'node' :
            self.message_func = self._message_func_node
            self.fc_node = nn.Linear(self._in_node_feats, self._out_node_feats*self._num_heads, bias=True)
            init.xavier_normal_(self.fc_node.weight)
        elif self.mpnn_type == 'edge' :
            self.message_func = self._message_func_edge
            self.fc_edge = nn.Linear(self._in_node_feats, self._out_node_feats*self._num_heads, bias=True)
            init.xavier_normal_(self.fc_edge.weight)
        elif self.mpnn_type == 'mix' :
            self.message_func = self._message_func_node_edge
            self.fc_node = nn.Linear(self._in_node_feats, self._out_node_feats*self._num_heads, bias=True)
            self.fc_edge = nn.Linear(self._in_node_feats, self._out_node_feats*self._num_heads, bias=True)
            init.xavier_normal_(self.fc_node.weight)
            init.xavier_normal_(self.fc_edge.weight)

    def _mask_msg(self, edges, msg):
        # masking with norm
        if 'norm' in edges.data:
            mask = th.unsqueeze(edges.data['norm'],2).repeat(1,self._num_heads, 1)
            msg = msg * mask
        return msg

    def _message_func_node(self, edges):
        h_out = edges.src['h_out']
        # assert th.sum(th.isnan(h_out.view(-1)).int()).item() == 0

        a = edges.data['a']
        msg = h_out * a
        msg = self._mask_msg(edges, msg)
        return {'m': msg}

    def _message_func_edge(self, edges):
        f_out = edges.data['f_out']
        a = edges.data['a']
        # assert th.sum(th.isnan(f_out.view(-1)).int()).item() == 0
        # assert th.sum(th.isnan(a.view(-1)).int()).item() == 0
        msg = f_out * a
        msg = self._mask_msg(edges, msg)
        # assert th.sum(th.isnan(msg.view(-1)).int()).item() == 0
        return {'m': msg}

    def _message_func_node_edge(self, edges):
        h_out = edges.src['h_out']
        f_out = edges.data['f_out']
        a = edges.data['a']
        msg = (f_out*0.5 + h_out*0.5) * a
        msg = self._mask_msg(edges, msg)
        return {'m': msg}

    def reduce_func(self, nodes):
        # assert th.sum(th.isnan(nodes.mailbox['m'].view(-1)).int()).item() == 0, print(nodes.mailbox['m'],nodes.mailbox['m'].shape)
        sum = th.sum(nodes.mailbox['m'], dim=1)
        # assert th.sum(th.isnan(sum.view(-1)).int()).item() == 0
        return {'h_out': sum}


    def edge_attention(self, edges):
        #extract features
        h_src = edges.src['f_ni']
        h_dst = edges.dst['f_nj']
        f = edges.data['f']
        #stack h_i | f_ij | h_j
        stack = th.cat([h_src, f, h_dst], dim=-1)
        # apply FC and activation
        f_out = self.fc_edges(stack)
        f_out = nn.functional.leaky_relu(f_out)
        f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)

        return {'f' : f_out}

    def forward(self, graph, nfeats, norm,
                efeats=None, rating=None, timestamp=None,
                get_attention=False):

            with graph.local_scope():
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                'output for those nodes will be invalid. '
                                'This is harmful for some applications, '
                                'causing silent performance regression. '
                                'Adding self-loop on the input graph by '
                                'calling `g = dgl.add_self_loop(g)` will resolve '
                                'the issue.')

                f_ni = self.fc_ni(nfeats)
                f_nj = self.fc_nj(nfeats)

                # add ni, nj factors
                graph.srcdata.update({'f_ni': f_ni})
                graph.dstdata.update({'f_nj': f_nj})

                efeats_list = []
                if self.review == True:
                    # assert th.sum(th.isnan(efeats.view(-1)).int()).item() == 0
                    efeats_list.append(efeats)
                if self.rating == True:
                    # assert th.sum(th.isnan(rating.view(-1)).int()).item() == 0
                    efeats_list.append(th.unsqueeze(rating, dim=1))
                if self.timestamp == True:
                    # assert th.sum(th.isnan(timestamp.view(-1)).int()).item() == 0
                    efeats_list.append(th.unsqueeze(timestamp, dim=1))

                efeats = th.cat(efeats_list, dim=1)
                f_eij = self.fc_fij(efeats)
                graph.edata.update({'f': f_eij})
                graph.apply_edges(self.edge_attention)
                f_out = graph.edata.pop('f')
                f_out = f_out.view(-1, self._num_heads * self._out_edge_feats)

                # assert th.sum(th.isnan(f_out.view(-1)).int()).item() == 0, print(f_out, f_out.shape)

                if self.bias is not None:
                    f_out = f_out + self.bias

                # assert th.sum(th.isnan(f_out.view(-1)).int()).item() == 0, print(f_out, f_out.shape)
                f_out = f_out.view(-1, self._num_heads, self._out_edge_feats)

                a = self.fc_attn(f_out)
                a = a.sum(-1).unsqueeze(-1)
                graph.edata['a'] = edge_softmax(graph, a)

                if self.mpnn_type == 'edge':
                    graph.edata['f_out'] = self.fc_edge(efeats).view(-1, self._num_heads, self._out_edge_feats)
                elif self.mpnn_type == 'node':
                    graph.ndata['h_out'] = self.fc_node(nfeats).view(-1, self._num_heads, self._out_node_feats)
                else:
                    graph.edata['f_out'] = self.fc_edge(efeats).view(-1, self._num_heads, self._out_edge_feats)
                    graph.ndata['h_out'] = self.fc_node(nfeats).view(-1, self._num_heads, self._out_node_feats)

                # assert th.sum(th.isnan(graph.edata['a'].view(-1)).int()).item() == 0, print(graph.edata['a'])
                # assert th.sum(th.isnan(graph.ndata['h_out'].view(-1)).int()).item() == 0

                graph.edata['norm'] = norm

                # calc weighted sum
                graph.update_all(self.message_func, self.reduce_func)

                h_out = graph.ndata['h_out'].view(-1, self._num_heads, self._out_node_feats)
                # assert th.sum(th.isnan(h_out.view(-1)).int()).item() == 0

                if get_attention:
                    return h_out, f_out, graph.edata.pop('a')
                else:
                    return th.sum(h_out, dim=1), th.sum(f_out, dim=1)
                    # f_out shape (E,1) : same as edata['norm']
