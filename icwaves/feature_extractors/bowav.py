from pprint import pprint

import numpy as np
from threadpoolctl import threadpool_info
from tqdm import tqdm

from icwaves.sikmeans.shift_kmeans import _asignment_step


def bag_of_waves(raw_ics, codebooks, metric='cosine', n_jobs=1, ord=None):
    """Flattened bag of words

    Parameters
    ----------
    raw_ics(array):
        A matrix of shape (m, n, w), where m, n and w are the number of
        idependent components (ICs), windows per IC, and window length.
    codebooks(sequence):
        A sequence of codebooks. C[i].shape = (k, P), with k and P being the number of centroids and centroid lenght, respectively.
    metric (str): metric used to perform assignment of signals (ICs) to centroids.
    n_jobs (int): number of joblib jobs used to perform assignment
    ord (non-zero int, inf, -inf, ‘fro’, ‘nuc’, None): `ord` argument passed to np.linalg.norm to perform instance-wise normalization of each
    BoWav from each codebook before concatenation. If None, don't perform normalization.
    """

    # Sanity check: use all threads possible in this function
    # You might need to unset all OMP-related variables
    # See joblib documentation
    pprint(threadpool_info())

    n_codebooks = len(codebooks)
    n_centroids = codebooks[0].shape[0]
    n_ics = raw_ics.shape[0]  # Number of segments
    n_features = n_centroids * n_codebooks

    X = np.zeros((n_ics, n_features), dtype=codebooks[0].dtype)
    x_squared_norms = None
    for i_ic in tqdm(range(n_ics)):
        for r in np.arange(n_codebooks):
            nu, _, _ = _asignment_step(
                raw_ics[i_ic], codebooks[r], metric,
                x_squared_norms, n_jobs=n_jobs)
            nu, counts = np.unique(nu, return_counts=True)
            # centroid index->feature index
            i_feature = nu + r * n_centroids
            if ord:
                X[i_ic, i_feature] = counts / np.linalg.norm(counts, ord=ord)
            else:
                X[i_ic, i_feature] = counts

    return X
