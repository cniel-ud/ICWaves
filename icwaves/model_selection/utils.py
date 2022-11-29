import numpy as np
from scipy.stats import rankdata
import warnings


def _store(key_name, array, n_splits, n_candidates, weights=None, splits=False, rank=False):
    """A small helper to store the scores/times to the cv_results_"""
    # When iterated first by splits, then by parameters
    # We want `array` to have `n_candidates` rows and `n_splits` cols.

    results = {}
    array = np.array(array, dtype=np.float64).reshape(
        n_candidates, n_splits)
    if splits:
        for split_idx in range(n_splits):
            # Uses closure to alter the results
            results["split%d_%s" %
                    (split_idx, key_name)] = array[:, split_idx]

    array_means = np.average(array, axis=1, weights=weights)
    results["mean_%s" % key_name] = array_means

    if key_name.startswith(("train_", "test_")) and np.any(
        ~np.isfinite(array_means)
    ):
        warnings.warn(
            f"One or more of the {key_name.split('_')[0]} scores "
            f"are non-finite: {array_means}",
            category=UserWarning,
        )

    # Weighted std is not directly available in numpy
    array_stds = np.sqrt(
        np.average(
            (array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights
        )
    )
    results["std_%s" % key_name] = array_stds

    if rank:
        results["rank_%s" % key_name] = np.asarray(
            rankdata(-array_means, method="min"), dtype=np.int32
        )

    return results
