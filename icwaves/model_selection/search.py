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

from icwaves.model_selection.utils import _store
from icwaves.model_selection.validation import _fit_and_score


def grid_search_cv(
    estimator,
    candidate_params,
    X,
    y,
    groups,
    expert_label_mask,
    cv,
    n_jobs
    ):

    n_splits = cv.get_n_splits(X, y, groups=groups)

    with parallel_backend("loky", inner_max_num_threads=1):
        parallel = Parallel(n_jobs=n_jobs)
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
                    split_progress=(split_idx, n_splits),
                    candidate_progress=(cand_idx, n_candidates)
                )
                for (cand_idx, parameters), (split_idx, (train, test)) in product(
                    enumerate(candidate_params), enumerate(cv.split(X, y, groups))
                )
            )
            all_out.extend(out)

    all_out = _aggregate_score_dicts(all_out)
    fit_time_dict = _store('fit_time', all_out['fit_time'], n_splits, n_candidates)
    test_time_dict = _store(
        'score_time', all_out['score_time'], n_splits, n_candidates)

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
    test_scores_dict = {'scores': all_out['test_scores']}
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
    best_expert_weight = best_params.pop('expert_weight', 1)
    best_estimator = clone(
        clone(estimator).set_params(**best_params)
    )

    refit_start_time = time.time()
    sample_weight = np.ones(X.shape[0])
    sample_weight[expert_label_mask] = best_expert_weight
    best_estimator.fit(X, y, clf__sample_weight=sample_weight)
    refit_end_time = time.time()
    refit_time = refit_end_time - refit_start_time

    results.update({
        'best_estimator': best_estimator,
        'best_score': best_score,
        'refit_time': refit_time
    })

    return results