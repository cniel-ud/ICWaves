import copy
from pprint import pprint

import numpy as np
from threadpoolctl import threadpool_info
from tqdm import tqdm
from icwaves.data_loaders import load_codebooks, load_raw_set

from icwaves.sikmeans.shift_kmeans import _asignment_step
from icwaves.utils import _build_centroid_assignments_file

BOWAV_NORM_MAP = {
    'none': None,
    'l_1': 1,
    'l_2': 2,
    'l_inf': np.inf,
}

def _compute_centroid_assignments(X, codebooks, metric='cosine', n_jobs=1):
    """Shift-invariant assignment of centroids

    Parameters
    ----------
    X(array):
        A matrix of shape (m, n, w), where m, n and w are the number of
        instances, windows per instance, and window length.
    codebooks(sequence):
        A sequence of codebooks. C[i].shape = (k, P), with k and P being the
        number of centroids and centroid lenght, respectively.
    metric (str): metric used to perform assignment of windows to centroids.
    n_jobs (int): number of joblib jobs used to perform assignment

    Return
    ------
    centroid_assignments(array):
        A matrix of shape (m, k, n), where m, k and n are the number of
        instances, codebooks and windows per instance, respectively.
        centroid_assignments[i, j, k] contains the index of the centroid
        of the j-th codebook assigned to the k-th window of the i-th instance.
    """

    # Sanity check: use all threads possible in this function
    # You might need to unset all OMP-related variables
    # See joblib documentation
    pprint(threadpool_info())

    n_codebooks = len(codebooks)
    n_instances, n_windows_per_instance, _  = X.shape

    centroid_assignments = np.zeros((n_instances, n_codebooks, n_windows_per_instance), dtype=int)
    x_squared_norms = None
    for i_inst in tqdm(range(n_instances)):
        for r in np.arange(n_codebooks):
            nu, _, _ = _asignment_step(
                X[i_inst], codebooks[r], metric,
                x_squared_norms, n_jobs=n_jobs)

            centroid_assignments[i_inst, r, :] = nu

    return centroid_assignments


def build_or_load_centroid_assignments(args, windowed_ics, codebooks):

    centroid_assignments_file = _build_centroid_assignments_file(args)
    if centroid_assignments_file.is_file():
        centroid_assignments = np.load(centroid_assignments_file)
    else:
        centroid_assignments = _compute_centroid_assignments(windowed_ics, codebooks)
        with centroid_assignments_file.open('wb') as f:
            np.save(f, centroid_assignments, allow_pickle=False)

    return centroid_assignments


def build_bowav_from_centroid_assignments(centroid_assignments, n_centroids, n_windows_per_segment, ord_str):
    """Build flattened bag of waves from centroid assignments. Use all windows on each time series.

    Parameters
    ----------
    centroid_assignments:
        A matrix of shape (m, c, n), where m, c and n are the number of
        time series, codebooks, and windows per time series, respectively.
        centroid_assignments[i, j, k] contains the index of the centroid
        assigned to the k-th window of the i-th time series when using the
        j-th codebook.
    n_centroids:
        The number of centroids on each codebook. This is used to compute
        the number of features in the BoWav.
    n_windows_per_segment:
        The number of windows per segment. This is the number of
        windows/assignments counted to compute the BoWav vector.
    ord_str (str):
        BOWAV_NORM_MAP[ord_str] is the `ord` argument passed to np.linalg.norm
        to perform instance-wise normalization of each BoWav from each codebook
        before concatenation. If None, don't perform normalization.
    """
    n_time_series, n_codebooks, n_windows_per_time_series  = centroid_assignments
    n_features = n_codebooks * n_centroids

    if n_windows_per_segment:
        n_segments_per_time_series = n_windows_per_time_series // n_windows_per_segment
    else:
        n_segments_per_time_series = 1
        n_windows_per_segment = n_windows_per_time_series

    bowav = np.zeros((n_time_series, n_segments_per_time_series, n_features), dtype=np.float32)
    for i_ts in tqdm(range(n_time_series)):
        for i_seg in range(n_segments_per_time_series):
            start_ind = i_seg * n_windows_per_segment
            end_ind = start_ind + n_windows_per_segment
            for r in np.arange(n_codebooks):
                nu, counts = np.unique(
                    centroid_assignments[i_ts, r, start_ind:end_ind],
                    return_counts=True
                )
                # centroid index->feature index
                i_feature = nu + r * n_centroids

                if BOWAV_NORM_MAP[ord_str]:
                    # instance-wise normalization
                    bowav[i_ts, i_seg, i_feature] = counts / np.linalg.norm(counts, ord=ord)
                else:
                    bowav[i_ts, i_seg, i_feature] = counts

    return bowav


def bag_of_waves(X, codebooks, metric='cosine', n_jobs=1, ord=None):
    """Flattened bag of words

    Parameters
    ----------
    X(array):
        A matrix of shape (m, n, w), where m, n and w are the number of
        instances, windows per instance, and window length.
    codebooks(sequence):
        A sequence of codebooks. C[i].shape = (k, P), with k and P being the number of centroids and centroid lenght, respectively.
    metric (str): metric used to perform assignment of windows to centroids.
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
    n_instances = X.shape[0]
    n_features = n_centroids * n_codebooks

    bowav = np.zeros((n_instances, n_features), dtype=codebooks[0].dtype)
    x_squared_norms = None
    for i_inst in tqdm(range(n_instances)):
        for r in np.arange(n_codebooks):
            nu, _, _ = _asignment_step(
                X[i_inst], codebooks[r], metric,
                x_squared_norms, n_jobs=n_jobs)
            nu, counts = np.unique(nu, return_counts=True)
            # centroid index->feature index
            i_feature = nu + r * n_centroids
            if ord:
                bowav[i_inst, i_feature] = counts / np.linalg.norm(counts, ord=ord)
            else:
                bowav[i_inst, i_feature] = counts

    return bowav
