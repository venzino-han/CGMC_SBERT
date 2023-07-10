import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import copy
import os
import sys
import torch as th
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from config import *
from preprocessing.feature_extractor import FeatureExtractor
from preprocessing.utils import convert_newid, reset_id
from preprocessing.bert_regresser import BertRegresser

from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer

BATCH_SIZE=256

class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: th.tensor(val[idx]) for key, val in self.encodings.items()}
        # item['labels'] = th.tensor(self.labels[idx])
        return item


def compute_kernel_bias(vecs, vec_dim):
    """
    kernel bias
    y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :vec_dim], -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5



def main():
    MODEL_NAME = 'bert-base-uncased'

    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=True, model_max_length=512)
    model = BertRegresser.from_pretrained(MODEL_NAME, config=config)
    for DATA_NAME in [
                      'office',
                      'sports',
                      'movie',
                    #   'yelp'
                      ]:
        TRAIN_PATH = f'dataset/{DATA_NAME}/{DATA_NAME}_train.csv'
        VALID_PATH = f'dataset/{DATA_NAME}/{DATA_NAME}_valid.csv'
        TEST_PATH = f'dataset/{DATA_NAME}/{DATA_NAME}_test.csv'

        # read data
        train_df = pd.read_csv(TRAIN_PATH)
        valid_df = pd.read_csv(VALID_PATH)
        test_df = pd.read_csv(TEST_PATH)
        df = pd.concat([train_df, valid_df, test_df])
        print(len(df))
        print('load df')
        review_texts = [ str(t) for t in df.reviewText.tolist()]
        print(len(review_texts))
        encodings = tokenizer(review_texts, padding=True, max_length=512, truncation=True,)
        labels = df.rating.tolist()

        dataset = ReviewDataset(encodings, labels)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE)

        device = 0
        embeddings = []
        model.to(device)
        model.eval()
        for item in tqdm(dataloader):
            input_ids, attention_mask  = item['input_ids'], item['attention_mask']
            input_ids, attention_mask  = input_ids.to(device), attention_mask.to(device)
            output = model(input_ids, attention_mask)
            # embedding = model.latent_vector.cpu()
            embedding = np.round(output.cpu().numpy(), 4)
            embeddings.append(embedding)

        embeddings = np.concatenate(embeddings, axis=0)
        print(embeddings.shape)
        kernel, bias = compute_kernel_bias(embeddings, 32)
        vecs = transform_and_normalize(embeddings, kernel, bias)
        vecs = th.from_numpy(vecs)


        with open(f'dataset/{DATA_NAME}/{DATA_NAME}_bert_whitening32_2.npy', 'wb') as f:
            np.save(f, vecs)


if __name__ == "__main__":
  main()