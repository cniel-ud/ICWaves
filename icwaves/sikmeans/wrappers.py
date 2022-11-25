"""
Numpy ans scikit-learn wrappers
"""
import sys
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms


def si_euclidean_distances(centroids, X, X_norm_squared,
                           squared=False):
    """
    Shift-invariant wrapper of euclidean_distances()

    Rows of `centroids` are shorter than rows of X. Rows of `X` are
    windowed at each shift to compute the distances to the rows of `centroids`.

    Parameters
    ----------
    centroids (numpy.ndarray):
        centroids[i] is a centroid with length `centroid_length`. Shape: (n_centroids, centroid_length)
    X (numpy.ndarray):
        X[i] is a sample with length `n_features`. Shape: (n_samples, n_features).
        `centroid_length` < `n_features`.
    X_norm_squared (numpy.ndarray):
        Precomputed squared euclidean norm of rows of windowed `X`. Shape: (n_shifts, n_samples).
    squared (bool):
        If True, the euclidean distance is squared.

    Returns
    -------
    distances (numpy.ndarray):
        distances[i][j][k] is the euclidean distance from centroids[j] to the
        windowed sample X[k, i:i+centroid_length].
    """

    n_centroids, centroid_length = centroids.shape
    n_samples, n_features = X.shape
    n_shifts = n_features - centroid_length + 1

    distances = np.empty((n_shifts, n_centroids, n_samples))

    # Use euclidean_distances() to find the distance from each windowed X to
    # all the centroids
    for shift in range(n_shifts):
        distances[shift] = euclidean_distances(
            X=centroids,
            Y=X[:, shift:shift+centroid_length],
            Y_norm_squared=X_norm_squared[shift],
            squared=True)

    return distances


def si_pairwise_distances_argmin_min(X, centroids, metric, x_squared_norms):
    """
    Shift-invariant wrapper of http://bit.ly/argmin_min_sklearn
    """

   # euclidean_distances() requires 2D
    if metric == 'euclidean' and x_squared_norms.ndim == 1:
        x_squared_norms = x_squared_norms.reshape(1, -1)
    if centroids.ndim == 1:
        centroids = centroids.reshape(1, -1)

    n_samples, sample_length = X.shape
    centroid_length = centroids.shape[1]
    n_shifts = sample_length - centroid_length + 1

    best_labels = np.empty((n_shifts, n_samples), dtype=np.int)
    best_distances = np.empty((n_shifts, n_samples))

    if metric == 'euclidean':
        for shift in range(n_shifts):
            # A bug on sklearn enforces a 2D array
            XX = x_squared_norms[shift].reshape((n_samples, 1))
            best_labels[shift], best_distances[shift] = \
                pairwise_distances_argmin_min(
                    X=X[:, shift:shift+centroid_length],
                    Y=centroids,
                    metric_kwargs={'squared': True,
                                   'X_norm_squared': XX})
    elif metric == 'cosine':
        for shift in range(n_shifts):
            best_labels[shift], best_distances[shift] = \
                pairwise_distances_argmin_min(
                    X=X[:, shift:shift+centroid_length],
                    Y=centroids,
                    metric=metric)
    else:
        sys.exit('%s metric not implemented' % metric)

    # For each sample, find best shift
    best_shifts = np.argmin(best_distances, axis=0)
    best_labels = best_labels[best_shifts, np.arange(n_samples)]
    best_distances = best_distances[best_shifts, np.arange(n_samples)]

    return best_labels, best_shifts, best_distances


def si_row_norms(X, centroid_length, squared=False):
    """
    Shift-invariant wrapper of row_norms()
    """

    n_samples, sample_length = X.shape
    n_shifts = sample_length - centroid_length + 1

    x_squared_norms = np.empty((n_shifts, n_samples))
    for shift in range(n_shifts):
        x_squared_norms[shift] = row_norms(
            X[:, shift:shift+centroid_length], squared=squared)

    return x_squared_norms
