import copy
import time
from collections import defaultdict
from functools import partial
from itertools import product

import numpy as np
from joblib import Parallel, delayed, parallel_backend
from numpy.ma import MaskedArray
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection._search import ParameterGrid
from sklearn.model_selection._validation import _aggregate_score_dicts
from icwaves.feature_extractors.bowav import build_bowav_from_centroid_assignments

from icwaves.model_selection.utils import _store
from icwaves.model_selection.validation import _fit_and_score


def grid_search_cv(
    estimator,
    feature_extractor,
    candidate_params,
    X,
    y,
    groups,
    expert_label_mask,
    cv,
    n_jobs,
):

    n_splits = cv.get_n_splits(X, y, groups=groups)

    with parallel_backend("loky", inner_max_num_threads=1):
        parallel = Parallel(n_jobs=n_jobs, verbose=42)
        with parallel:

            candidate_params = list(ParameterGrid(candidate_params))
            n_candidates = len(candidate_params)
            all_out = []

            out = parallel(
                delayed(_fit_and_score)(
                    clone(estimator),
                    X,
                    y,
                    expert_label_mask,
                    train=train,
                    test=test,
                    parameters=parameters,
                    scorer=balanced_accuracy_score,
                    feature_extractor=feature_extractor,
                    split_progress=(split_idx, n_splits),
                    candidate_progress=(cand_idx, n_candidates),
                )
                for (cand_idx, parameters), (split_idx, (train, test)) in product(
                    enumerate(candidate_params), enumerate(cv.split(X, y, groups))
                )
            )
            all_out.extend(out)

    all_out = _aggregate_score_dicts(all_out)
    fit_time_dict = _store("fit_time", all_out["fit_time"], n_splits, n_candidates)
    test_time_dict = _store("score_time", all_out["score_time"], n_splits, n_candidates)

    param_results = defaultdict(
        partial(
            MaskedArray,
            np.empty(
                n_candidates,
            ),
            mask=True,
            dtype=object,
        )
    )
    for cand_idx, params in enumerate(candidate_params):
        for name, value in params.items():
            # An all masked empty array gets created for the key
            # `"param_%s" % name` at the first occurrence of `name`.
            # Setting the value at an index also unmasks that index
            param_results["param_%s" % name][cand_idx] = value

    results = {**fit_time_dict, **test_time_dict, **param_results}
    results["params"] = candidate_params
    # TODO: We save back candidate parameters in the "parameters" key of `result`
    # in validation.py, but we never access that key here. Maybe remove that in
    # validation.py?
    test_scores_dict = {"scores": all_out["test_scores"]}
    # Computed the (weighted) mean and std for test scores alone
    results.update(
        _store(
            "test_scores",
            test_scores_dict["scores"],
            n_splits,
            n_candidates,
            splits=True,
            rank=True,
            weights=None,
        )
    )
    best_index = results["rank_test_scores"].argmin()
    best_score = results[f"mean_test_scores"][best_index]
    best_params = copy.deepcopy(results["params"][best_index])
    best_expert_weight = best_params.pop("expert_weight", 1)
    best_n_training_windows_per_segment = best_params.pop(
        "n_training_windows_per_segment"
    )
    del best_params["n_validation_windows_per_segment"]
    del best_params["input_or_output_aggregation_method"]
    best_estimator = clone(clone(estimator).set_params(**best_params))

    refit_start_time = time.time()
    sample_weight = np.ones(X.shape[0])
    sample_weight[expert_label_mask] = best_expert_weight

    # Build train BoWav vector for a given segment length
    X = feature_extractor(X, best_n_training_windows_per_segment)

    n_segments = X.shape[1]
    # vertically concatenate train BoWav vectors: (m, n, p) -> (m*n, p)
    X = np.vstack(X)
    # expand train labels to match train BoWav vectors
    y = np.repeat(y, n_segments)
    # expand train sample weights to match train BoWav vectors
    sample_weight = np.repeat(sample_weight, n_segments)

    named_steps = getattr(estimator, "named_steps", None)
    if named_steps is not None:
        best_estimator.fit(X, y, clf__sample_weight=sample_weight)
    else:
        best_estimator.fit(X, y, sample_weight=sample_weight)

    refit_end_time = time.time()
    refit_time = refit_end_time - refit_start_time

    results.update(
        {
            "best_estimator": best_estimator,
            "best_score": best_score,
            "refit_time": refit_time,
        }
    )

    return results
