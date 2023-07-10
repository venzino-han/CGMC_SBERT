import numpy as np
import pandas as pd
from tqdm import tqdm

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

BATCH_SIZE = 64


if __name__ == '__main__':

    for DATA_NAME in ['music', 'clothing', 'games', 'office', 'sports']:
        TRAIN_PATH = f'dataset/{DATA_NAME}/{DATA_NAME}_core_train.csv'
        VALID_PATH = f'dataset/{DATA_NAME}/{DATA_NAME}_core_valid.csv'
        TEST_PATH = f'dataset/{DATA_NAME}/{DATA_NAME}_core_test.csv'

        # read data
        train_df = pd.read_csv(TRAIN_PATH)
        valid_df = pd.read_csv(VALID_PATH)
        test_df = pd.read_csv(TEST_PATH)
        df = pd.concat([train_df, valid_df, test_df])
        # df = df.dropna()
        df['item_id'] += max(df.user_id)+1

        path = f'dataset/{DATA_NAME}/{DATA_NAME}_sbert_review_doc_whitening32.npy'
        arrays = np.load(path)

        user_item_pair_array = []
        for uid, iid in zip(df.user_id, df.item_id):
            u_arr = arrays[uid]
            i_arr = arrays[iid]
            # doc_mul = u_arr*i_arr
            doc_mul = np.concatenate([u_arr,i_arr], axis=0)
            user_item_pair_array.append(doc_mul)

        arrays = np.array(user_item_pair_array)
        print(arrays.shape)
        with open(f'dataset/{DATA_NAME}/{DATA_NAME}_sbert_review_doc_cat_whitening32.npy', 'wb') as f:
            np.save(f, arrays)
