import pandas as pd
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config import *

def split_origin_data():
    for i, df in enumerate(pd.read_json(ORIGINAL_DATASET_PATH, chunksize=1e6, lines=True)):
        print(f'processing chunk {i}th')
        df = df[['overall', 'reviewerID', 'asin', 'reviewText', 'unixReviewTime']]
        df.to_csv(f"{DATA_PATH}/batch/movie_amz_{i}.csv")


def convert_newid(origin_id:int, id_dict:dict, max_id:int):

    if origin_id in id_dict :
        new_id = id_dict.get(origin_id)
    else:
        id_dict[origin_id] = max_id
        new_id = max_id
        max_id += 1

    return new_id, id_dict, max_id


def reset_id():
    user_id_dict = {}
    item_id_dict = {}

    user_id_max = 0
    item_id_max = 0

    for j in range(9):
        df = pd.read_csv(f"{DATA_PATH}/batch/movie_amz_{j}.csv")
        print(f'reset id chunk {j}th')
        userids=[]
        itemids=[]
        df.rename(columns = {'Unnamed: 0' : 'id'}, inplace = True)

        for i in tqdm(range(len(df))):
            origin_user_id = df.reviewerID[i]
            origin_item_id = df.asin[i]

            new_user_id, user_id_dict, user_id_max = convert_newid(origin_user_id, user_id_dict, user_id_max)
            new_item_id, item_id_dict, item_id_max = convert_newid(origin_item_id, item_id_dict, item_id_max)

            userids.append(new_user_id)
            itemids.append(new_item_id)

        df['user_id'] = userids
        df['item_id'] = itemids

        df = df[['id', 'overall', 'user_id', 'item_id', 'unixReviewTime', 'reviewText']]
        df.columns = ['id', 'rating', 'user_id', 'item_id', 'unixReviewTime', 'reviewText']
        df.to_csv(f"{DATA_PATH}data_newid/movie_amz_{j}.csv", index=False)


if __name__ == '__main__':
    data_path

    print('start preprocessing')
    split_origin_data()
    reset_id()
    print('preprocessing end')
