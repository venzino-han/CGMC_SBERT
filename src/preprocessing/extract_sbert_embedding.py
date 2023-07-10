import numpy as np
import pandas as pd
from tqdm import tqdm

import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertPreTrainedModel
from transformers import AutoConfig, AutoTokenizer


BATCH_SIZE = 256


class ReviewDataset(Dataset):
    def __init__(self, reviews):
        self.reviews = reviews

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        item = self.reviews[idx]
        return item


if __name__ == '__main__':
    for DATA_NAME in ['games', 'movie', 'music', 'office', 'sports']:
        TRAIN_PATH = f'dataset/{DATA_NAME}/{DATA_NAME}_train.csv'
        VALID_PATH = f'dataset/{DATA_NAME}/{DATA_NAME}_valid.csv'
        TEST_PATH = f'dataset/{DATA_NAME}/{DATA_NAME}_test.csv'

        # read data
        train_df = pd.read_csv(TRAIN_PATH)
        valid_df = pd.read_csv(VALID_PATH)
        test_df = pd.read_csv(TEST_PATH)
        df = pd.concat([train_df, valid_df, test_df])
        # df = df.dropna()
        df['item_id'] += max(df.user_id)+1

        device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        # model = SentenceTransformer('msmarco-distilbert-dot-v5')
        model = SentenceTransformer('all-mpnet-base-v2')
        
        model = model.to(device)

        try:
            dataset = ReviewDataset(df.review.tolist())
        except: 
            dataset = ReviewDataset(df.reviewText.tolist())
        dataloader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)
        arrays = []
        for batch in tqdm(dataloader):
            embeddings = model.encode(batch)
            arrays.append(embeddings)

        arrays = np.concatenate(arrays, axis=0)
        with open(f'dataset/{DATA_NAME}/{DATA_NAME}_sbert_mpnet.npy', 'wb') as f:
            np.save(f, arrays)
        # with open(f'dataset/{DATA_NAME}/{DATA_NAME}_bert_base.npy', 'wb') as f:
        #     np.save(f, arrays)
