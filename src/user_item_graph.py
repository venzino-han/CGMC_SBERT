"""build graph with edge features"""
from collections import defaultdict

import scipy.sparse as sp

import numpy as np
import pandas as pd
import torch as th

import dgl

import pandas as pd
import numpy as np
import scipy.sparse as sp

# from config import *

#######################
# Build graph
#######################

class UserItemGraph(object):
    """
    Build user-item graph for training
    Bulid Homogeneous graph with df
    only user user-item pairs in range when extracting subgraph
    """
    def __init__(self, label_col:str, user_col:str, item_col:str,
                 df:pd.DataFrame, edge_idx_range:tuple, edge_feature_from:str=None,
                 ):

        self._num_user = len(df[user_col].unique())
        self._num_item = len(df[item_col].unique())
        self._num_label = len(df[label_col].unique())

        df[item_col] += self._num_user
        u_idx, i_idx, = df[user_col].to_numpy(), df[item_col].to_numpy(),
        rating = df[label_col].to_numpy()
        etypes = np.array([0]*len(df))
        labels = (df[label_col].to_numpy() - 1)/4

        if 'unixReviewTime' in df.columns:
            ts = df['unixReviewTime'].to_numpy()
        else:
            ts = df['ts'].to_numpy()

        self.uids = df[user_col].to_list()
        self.iids = df[item_col].to_list()

        # use whole data to build main graph
        # add bidirect edges
        num_nodes = self._num_user + self._num_item
        src_nodes = np.concatenate((u_idx, i_idx))
        dst_nodes = np.concatenate((i_idx, u_idx))
        labels = np.concatenate((labels, labels))
        rating = np.concatenate((rating, rating))
        etypes = np.concatenate((etypes, etypes))
        ts = np.concatenate((ts, ts))

        print('df len ', len(df))
        print('nodes ', num_nodes)
        print('pairs ', src_nodes.shape, dst_nodes.shape )
        print('labels shape', labels.shape)

        sp_mat = sp.coo_matrix((labels,(src_nodes, dst_nodes)), shape=(num_nodes, num_nodes))
        self.graph =dgl.from_scipy(sp_mat=sp_mat, idtype=th.int32)

        start, end = edge_idx_range
        if edge_feature_from is not None:
            e_feature = np.load(edge_feature_from)
            print('edge feature origin shape : ', e_feature.shape)
            e_feature = e_feature[:end]
            print('edge feature shape : ', e_feature.shape)
            self.graph.edata['feature'] = th.tensor(np.concatenate([e_feature, e_feature]), dtype=th.float32)

        self.graph.edata['original_src_idx'] = th.tensor(src_nodes, dtype=th.int32)
        self.graph.edata['original_dst_idx'] = th.tensor(dst_nodes, dtype=th.int32)
        self.graph.edata['label'] = th.tensor(labels, dtype=th.float32)
        self.graph.edata['rating'] = th.tensor(rating, dtype=th.int32)
        self.graph.edata['etype'] = th.tensor(etypes, dtype=th.int32)
        self.graph.edata['ts'] = th.tensor(ts, dtype=th.int32)

        #extract subgraph pair idx
        self.user_indices = th.tensor(u_idx[start:end], dtype=th.int32)
        self.item_indices = th.tensor(i_idx[start:end], dtype=th.int32)
        self.labels = th.tensor(labels[start:end], dtype=th.float32)

        self.user_item_pairs = self.get_user_item_pairs()
        nid_neghibor_dict = defaultdict(list)
        for u, i in zip(u_idx, i_idx):
            nid_neghibor_dict[i].append(u)
            nid_neghibor_dict[u].append(i)

        self.nid_neghibor_dict = dict()
        for k, v in nid_neghibor_dict.items():
            self.nid_neghibor_dict[k] = th.tensor(v)

        # df = df.query('rating==5')
        uids = df[user_col].to_list()
        iids = df[item_col].to_list()
        nid_neghibor_dict = defaultdict(list)
        for u, i in zip(uids, iids):
            nid_neghibor_dict[i].append(u)
            nid_neghibor_dict[u].append(i)
        
        self.nid_neghibor_set_dict = defaultdict(set)
        for k, v in nid_neghibor_dict.items():
            self.nid_neghibor_set_dict[k] = set(v)
        

    def get_user_item_pairs(self):
        pairs = []
        for u, i in zip(self.user_indices, self.item_indices):
            pairs.append((u,i))
        return pairs