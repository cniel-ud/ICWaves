from pathlib import Path
from pprint import pprint

from argparse import Namespace
import numpy as np
from threadpoolctl import threadpool_info
from tqdm import tqdm

from icwaves.data_loaders import load_codebooks_wrapper
from icwaves.preprocessing import load_or_build_preprocessed_data
from icwaves.sikmeans.shift_kmeans import _asignment_step
from icwaves.file_utils import _build_centroid_assignments_file
from icwaves.data.types import DataBundle


def _compute_centroid_assignments(X, codebooks, metric="cosine", n_jobs=1):
    """Shift-invariant assignment of centroids

    Parameters
    ----------
    X(array):
        A matrix of shape (m, n, w), where m, n and w are the number of
        time series, windows per time series, and window length.
    codebooks(sequence):
        A sequence of codebooks. C[i].shape = (k, P), with k and P being the
        number of centroids and centroid lenght, respectively.
    metric (str): metric used to perform assignment of windows to centroids.
    n_jobs (int): number of joblib jobs used to perform assignment

    Return
    ------
    centroid_assignments(array):
        A matrix of shape (m, k, n), where m, k and n are the number of
        time series, codebooks and windows per time series, respectively.
        centroid_assignments[i, j, k] contains the index of the centroid
        of the j-th codebook assigned to the k-th window of the i-th time series.
    """

    # Sanity check: use all threads possible in this function
    # You might need to unset all OMP-related variables
    # See joblib documentation
    pprint(threadpool_info())

    n_codebooks = len(codebooks)
    n_time_series, n_windows_per_instance, _ = X.shape

    centroid_assignments = np.zeros(
        (n_time_series, n_codebooks, n_windows_per_instance), dtype=int
    )
    x_squared_norms = None
    for i_inst in tqdm(range(n_time_series)):
        for r in np.arange(n_codebooks):
            nu, _, _ = _asignment_step(
                X[i_inst], codebooks[r], metric, x_squared_norms, n_jobs=n_jobs
            )

            centroid_assignments[i_inst, r, :] = nu

    return centroid_assignments


def build_or_load_centroid_assignments_and_labels(args: Namespace) -> DataBundle:

    data_folder = Path(args.path_to_centroid_assignments)
    data_folder.mkdir(exist_ok=True, parents=True)
    centroid_assignments_file = _build_centroid_assignments_file(args)
    centroid_assignments_file = data_folder.joinpath(centroid_assignments_file)
    codebooks = load_codebooks_wrapper(args)
    n_centroids = codebooks[0].shape[0]

    if centroid_assignments_file.is_file():
        # Load labels
        db = load_or_build_preprocessed_data(args)
        with centroid_assignments_file.open("rb") as f:
            db.data = np.load(f)
    else:
        # Load or build preprocessed data
        db = load_or_build_preprocessed_data(args)
        db.data = _compute_centroid_assignments(db.data, codebooks)
        with centroid_assignments_file.open("wb") as f:
            np.save(f, db.data, allow_pickle=False)

    db.n_centroids = n_centroids

    return db


def build_bowav_from_centroid_assignments(
    centroid_assignments, n_centroids, n_windows_per_segment, normalize_by_segments=True
):
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
    normalize_by_segments:
        If True, normalize counts by the number of segments to get rates.
        This addresses the document count disparity between training and
        validation/test when using different segment lengths.
    """
    n_time_series, n_codebooks, n_windows_per_time_series = centroid_assignments.shape
    n_features = n_codebooks * n_centroids

    if n_windows_per_segment:
        n_segments_per_time_series = n_windows_per_time_series // n_windows_per_segment
    else:
        n_segments_per_time_series = 1
        n_windows_per_segment = n_windows_per_time_series

    # Use float32 if normalizing, int32 otherwise
    dtype = np.float32 if normalize_by_segments else np.int32
    bowav = np.zeros(
        (n_time_series, n_segments_per_time_series, n_features), dtype=dtype
    )
    
    for i_ts in range(n_time_series):
        for i_seg in range(n_segments_per_time_series):
            start_ind = i_seg * n_windows_per_segment
            end_ind = start_ind + n_windows_per_segment
            for r in np.arange(n_codebooks):
                nu, counts = np.unique(
                    centroid_assignments[i_ts, r, start_ind:end_ind], return_counts=True
                )
                # centroid index->feature index
                i_feature = nu + r * n_centroids
                if normalize_by_segments:
                    # Normalize by number of segments to get rates
                    bowav[i_ts, i_seg, i_feature] = counts / n_segments_per_time_series
                else:
                    bowav[i_ts, i_seg, i_feature] = counts

    return bowav


def bag_of_waves(X, codebooks, metric="cosine", n_jobs=1, ord=None):
    """Flattened bag of words

    Parameters
    ----------
    X(array):
        A matrix of shape (m, n, w), where m, n and w are the number of
        time series, windows per time series, and window length.
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
    n_time_series = X.shape[0]
    n_features = n_centroids * n_codebooks

    bowav = np.zeros((n_time_series, n_features), dtype=codebooks[0].dtype)
    x_squared_norms = None
    for i_inst in tqdm(range(n_time_series)):
        for r in np.arange(n_codebooks):
            nu, _, _ = _asignment_step(
                X[i_inst], codebooks[r], metric, x_squared_norms, n_jobs=n_jobs
            )
            nu, counts = np.unique(nu, return_counts=True)
            # centroid index->feature index
            i_feature = nu + r * n_centroids
            if ord:
                bowav[i_inst, i_feature] = counts / np.linalg.norm(counts, ord=ord)
            else:
                bowav[i_inst, i_feature] = counts

    return bowav
