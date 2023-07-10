import pandas as pd
from itertools import combinations
import math

from user_item_graph import UserItemGraph
from dataset import UserItemDataset



dataset = 'movie'
dataset_filename = 'movie'

data_path = f'dataset/{dataset}/{dataset_filename}'
train_df = pd.read_csv(f'{data_path}_train.csv')
valid_df = pd.read_csv(f'{data_path}_valid.csv')
test_df = pd.read_csv(f'{data_path}_test.csv')

valid_df = pd.concat([train_df, valid_df])
test_df = pd.concat([valid_df, test_df])

#accumulate
train_graph = UserItemGraph(label_col='rating',
                            user_col='user_id',
                            item_col='item_id',
                            # edge_feature_from=feature_path,
                            df=train_df,
                            edge_idx_range=(0, len(train_df)))

train_dataset = UserItemDataset(user_item_graph=train_graph,
                                hop=1, sample_ratio=1.0, max_nodes_per_hop=100)


valid_graph = UserItemGraph(label_col='rating',
                        user_col='user_id',
                        item_col='item_id',
                        # edge_feature_from=feature_path,
                        df=valid_df,
                        edge_idx_range=(len(train_df), len(valid_df)))

valid_dataset = UserItemDataset(user_item_graph=valid_graph,
                                hop=1, sample_ratio=1.0, max_nodes_per_hop=100)

print(len(valid_graph.nid_neghibor_set_dict))


uids = valid_graph.uids
iids = valid_graph.iids

def set_sim(us, vs, uvs):
    return len(uvs) / len(us.union(vs))

def 

n = len(uids)
l = n*(n-1)//2
c = 0
us, vs, sims = [0]*l, [0]*l, [0.]*l
print(l)
for i, (u,v) in enumerate(combinations(uids,2)):
    if i%500000 == 0:
        print(i,c, ' : ', i/l*100 )
    un =  valid_graph.nid_neghibor_set_dict.get(u,set())
    vn =  valid_graph.nid_neghibor_set_dict.get(v,set())
    uvs = un&vn
    if len(uvs)>0:
        sim = set_sim(un, vn, uvs)
        us[i], vs[i] = u, v
        sims[i] = sim
        c+=1

us, vs, sims = us[:c], vs[:c], sims[:c]

df = pd.DataFrame({'u':us, 'v':vs, 'sim':sims})
df.to_csv(data_path+'_sim.csv')


from multiprocessing import Pool

def get_sim(u,v):
    un =  valid_graph.nid_neghibor_set_dict.get(u,set())
    vn =  valid_graph.nid_neghibor_set_dict.get(v,set())
    uvs = un&vn
    if len(uvs)>0:
        sim = set_sim(un, vn, uvs)
        us[i], vs[i] = u, v
        sims[i] = sim
    return 


def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))


