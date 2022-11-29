import datetime
import time
from pprint import pprint

import numpy as np
from threadpoolctl import threadpool_info
from tqdm import tqdm

from icwaves.sikmeans.shift_kmeans import _asignment_step


def bag_of_waves(raw_ics, codebooks, metric='cosine', n_jobs=1):
    """Flattened bag of words

    Parameters
    ----------
    raw_ics(array):
        A matrix of shape (m, n, w), where m, n and w are the number of
        idependent components (ICs), windows per IC, and window length.
    codebooks(sequence):
        A sequence of codebooks. C[i].shape = (k, P), with k and P being the number of centroids and centroid lenght, respectively.
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
    total_elapsed_time  = 0
    for i_ic in tqdm(range(n_ics)):
        t_start_ic = time.time()
        for r in np.arange(n_codebooks):
            nu, _, _ = _asignment_step(
                raw_ics[i_ic], codebooks[r], metric,
                x_squared_norms, n_jobs=n_jobs)
            nu, counts = np.unique(nu, return_counts=True)
            # centroid index->feature index
            i_feature = nu + r * n_centroids
            X[i_ic, i_feature] = counts

        t_end_ic = time.time()
        t_elapsed_ic = t_end_ic-t_start_ic
        total_elapsed_time += t_elapsed_ic
        t_elapsed_ic_str = str(datetime.timedelta(seconds=t_elapsed_ic))
        print(f'Time spent encoding IC-{i_ic}: {t_elapsed_ic_str} [hh:mm:ss]')
    print(
        f'Total time spent encoding the ICs with Bag-of-Waves: {i_ic}:'
        f'{str(datetime.timedelta(seconds=total_elapsed_time))} [hh:mm:ss]'
        )

    return X
