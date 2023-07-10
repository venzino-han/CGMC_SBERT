import numpy as np
import torch as th
# Bert-Whitening

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


if __name__=='__main__':
    # DATA_NAME = 'music'
    pt_type = 'mpnet'
    dim=32
    for DATA_NAME in ['games', 'movie', 'music', 'office', 'sports']:
    # for DATA_NAME in [ 'yelp', 'epinions_3c']:
    # for DATA_NAME in ['movie']:
        # path = f'dataset/{DATA_NAME}/{DATA_NAME}_sbert_msmarco_review.npy'
        path = f'dataset/{DATA_NAME}/{DATA_NAME}_sbert_{pt_type}.npy'
        data = np.load(path)
        print(data.shape)
        kernel, bias = compute_kernel_bias(data, dim)
        vecs = transform_and_normalize(data, kernel, bias)
        vecs = th.from_numpy(vecs)
        print(vecs.shape)
        # with open(f'dataset/{DATA_NAME}/{DATA_NAME}_sbert_review_msmarco_whitening{dim}.npy', 'wb') as f:
        with open(f'dataset/{DATA_NAME}/{DATA_NAME}_sbert_{pt_type}_whitening{dim}.npy', 'wb') as f:
            np.save(f, vecs)