# %%
import copy
import logging
import pickle
from pathlib import Path
import time

import numpy as np
from numpy.random import default_rng
import scipy
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._validation import _aggregate_score_dicts

from icwaves.feature_extractors.iclabel_features import get_iclabel_features_per_segment
from icwaves.file_utils import build_results_file, read_args_from_file
from icwaves.model_selection.split import LeaveOneSubjectOutExpertOnly
from icwaves.model_selection.utils import _store
from icwaves.argparser import (
    create_argparser_aggregate_results,
    create_argparser_all_params,
)
from icwaves.preprocessing import load_or_build_ics_and_labels


if __name__ == "__main__":

    logging.basicConfig(
        format="%(levelname)s:%(filename)s:%(message)s", level=logging.DEBUG
    )
    logging.info("Started")

    parser = create_argparser_aggregate_results()
    one_run_args = parser.parse_args()

    args_list = read_args_from_file(one_run_args.path_to_config_file)

    all_params_parser = create_argparser_all_params()
    args = all_params_parser.parse_args(args_list)

    new_rng = default_rng(13)
    # scikit-learn doesn't support the new numpy Generator:
    old_rng = np.random.RandomState(13)

    # Load data
    X_train, y_train, srate, expert_label_mask, subj_ind, noisy_labels = (
        load_or_build_ics_and_labels(args)
    )

    input_or_output_aggregation_method = ["count_pooling", "majority_vote"]
    training_segment_length = [int(l * srate) for l in args.training_segment_length]
    validation_segment_length = args.validation_segment_length
    if validation_segment_length == -1:
        validation_segment_length = None
    else:
        validation_segment_length = int(validation_segment_length * srate)
    validation_segment_length = [validation_segment_length]

    cv = LeaveOneSubjectOutExpertOnly(expert_label_mask)

    # ICLabel max-scale PSDs and autocorrelation functions
    # Do we still need to scale the features here?
    # My intuition tells me not...I checked and although the features are not
    # zero mean, their mean is not widely different, and the variances are
    # rather small.
    clf = RandomForestClassifier()

    clf_params = dict(
        n_estimators=300,
        random_state=old_rng,
        class_weight="balanced",
    )
    clf.set_params(**clf_params)

    candidate_params = dict(
        min_samples_split=args.min_samples_split,
        expert_weight=args.expert_weight,
        input_or_output_aggregation_method=input_or_output_aggregation_method,
        training_segment_length=training_segment_length,
        validation_segment_length=validation_segment_length,
    )

    candidate_params = list(ParameterGrid(candidate_params))
    n_candidates = len(candidate_params)
    n_splits = cv.get_n_splits(X_train, y_train, groups=subj_ind)

    all_out = []
    results_path = Path(args.path_to_results, "tmp_random_forest_PSD_autocorr")
    for candidate_idx in range(n_candidates):
        for split_idx in range(n_splits):
            file = results_path.joinpath(
                f"candidate_{candidate_idx}_split_{split_idx}.pkl"
            )
            with open(file, "rb") as f:
                result = pickle.load(f)
            all_out.append(result)

    all_out = _aggregate_score_dicts(all_out)
    fit_time_dict = _store("fit_time", all_out["fit_time"], n_splits, n_candidates)
    test_time_dict = _store("score_time", all_out["score_time"], n_splits, n_candidates)

    results = {**fit_time_dict, **test_time_dict}
    results["params"] = candidate_params
    test_scores_dict = {"scores": all_out["test_scores"]}
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
    best_score = results["mean_test_scores"][best_index]
    best_params = copy.deepcopy(results["params"][best_index])
    logging.info("Best score: %s", best_score)
    logging.info("Best params: %s", best_params)

    best_expert_weight = best_params.pop("expert_weight", 1)
    best_training_segment_length = best_params.pop("training_segment_length")
    del best_params["validation_segment_length"]
    del best_params["input_or_output_aggregation_method"]
    best_estimator = clone(clone(clf).set_params(**best_params))

    refit_start_time = time.time()
    sample_weight_train = np.ones(X_train.shape[0])
    sample_weight_train[expert_label_mask] = best_expert_weight

    # Build feature vector
    X_train = get_iclabel_features_per_segment(
        signal=X_train,
        sfreq=srate,
        use_autocorr=True,
        segment_len=best_training_segment_length,
    )

    # X_train.shape = (n_time_series, n_segments, n_features)
    n_segments = X_train.shape[1]
    # vertically concatenate train feature vectors: (m, n, p) -> (m*n, p)
    X_train = np.vstack(X_train)
    # expand train labels to match train feature vectors
    y_train = np.repeat(y_train, n_segments)
    # expand train sample weights to match train feature vectors
    sample_weight_train = np.repeat(sample_weight_train, n_segments)

    logging.info("Fitting best estimator")

    named_steps = getattr(best_estimator, "named_steps", None)
    if named_steps is not None:
        best_estimator.fit(X_train, y_train, clf__sample_weight=sample_weight_train)
    else:
        best_estimator.fit(X_train, y_train, sample_weight=sample_weight_train)

    refit_end_time = time.time()
    refit_time = refit_end_time - refit_start_time

    results.update(
        {
            "best_estimator": best_estimator,
            "best_score": best_score,
            "refit_time": refit_time,
        }
    )

    results_folder = Path(args.path_to_results)
    results["sklearn_version"] = sklearn.__version__
    results["numpy_version"] = np.__version__
    results["scipy_version"] = scipy.__version__

    # create results file name using best hyperparameters
    # TODO: improve the creation of this file name
    results_file = f"PSD_autocorr_rf_trainseg{best_training_segment_length}_valseglen300_expw{int(best_expert_weight)}.pkl"
    results_file = results_folder.joinpath(results_file)
    with results_file.open("wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    logging.info("Finished")
