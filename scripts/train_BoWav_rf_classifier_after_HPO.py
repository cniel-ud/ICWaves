import copy
import logging
import time
from collections import defaultdict
from functools import partial

import numpy as np
from numpy.ma import MaskedArray
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection._search import ParameterGrid
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.pipeline import Pipeline
from icwaves.argparser import (
    create_argparser_aggregate_results,
    create_argparser_all_params,
)
from icwaves.feature_extractors.bowav import (
    build_bowav_from_centroid_assignments,
    build_or_load_centroid_assignments_and_labels,
)

from icwaves.model_selection.split import LeaveOneSubjectOutExpertOnly
from icwaves.model_selection.utils import _store
from icwaves.file_utils import build_results_file, read_args_from_file
from pathlib import Path
import pickle
import sklearn
import scipy

TF_IDF_NORM_MAP = {
    "none": None,
    "l1": "l1",
    "l2": "l2",
}

if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s:%(filename)s:%(message)s", level=logging.DEBUG
    )
    logging.info("Started")

    parser = create_argparser_aggregate_results()
    agg_args = parser.parse_args()
    args_list = read_args_from_file(agg_args.path_to_config_file)
    all_params_parser = create_argparser_all_params()
    args = all_params_parser.parse_args(args_list)

    # Load or build centroid assignments
    centroid_assignments, labels, expert_label_mask, subj_ind, _, n_centroids = (
        build_or_load_centroid_assignments_and_labels(args)
    )
    input_or_output_aggregation_method = ["count_pooling", "majority_vote"]
    training_segment_length = args.training_segment_length
    # Compute training_segment_length
    training_segment_length = [
        int(s / args.window_length) for s in training_segment_length
    ]
    validation_segment_length = args.validation_segment_length
    # Compute validation_segment_length
    if validation_segment_length == -1:
        validation_segment_length = None
    else:
        validation_segment_length = int(validation_segment_length / args.window_length)
    validation_segment_length = [validation_segment_length]
    if not isinstance(args.tf_idf_norm, list):
        args.tf_idf_norm = [args.tf_idf_norm]

    cv = LeaveOneSubjectOutExpertOnly(expert_label_mask)

    clf = RandomForestClassifier()

    rng = np.random.RandomState(13)
    clf_params = dict(
        n_estimators=300,
        random_state=rng,
        class_weight="balanced",
    )
    clf.set_params(**clf_params)

    candidate_params = dict(
        min_samples_split=args.min_samples_split,
        # min_samples_leaf=args.min_samples_leaf,
        expert_weight=args.expert_weight,
        input_or_output_aggregation_method=input_or_output_aggregation_method,
        training_segment_length=training_segment_length,
        validation_segment_length=validation_segment_length,
    )
    candidate_params = list(ParameterGrid(candidate_params))
    n_candidates = len(candidate_params)
    all_out = []

    estimator = clf
    X = centroid_assignments
    y = labels
    n_splits = cv.get_n_splits(centroid_assignments, labels, groups=subj_ind)

    # TODO: use a folder name that is parametrized by the specific HPO that was used,
    # instead of "temporal_results"
    results_path = Path(args.path_to_results, "tmp_random_forest_bowav")

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
    best_score = results["mean_test_scores"][best_index]
    best_params = copy.deepcopy(results["params"][best_index])
    logging.info("Best score: %s", best_score)
    logging.info("Best params: %s", best_params)
    best_expert_weight = best_params.pop("expert_weight", 1)
    best_training_segment_length = best_params.pop("training_segment_length")
    del best_params["validation_segment_length"]
    del best_params["input_or_output_aggregation_method"]
    best_estimator = clone(clone(estimator).set_params(**best_params))

    refit_start_time = time.time()
    sample_weight = np.ones(X.shape[0])
    sample_weight[expert_label_mask] = best_expert_weight

    # Build train BoWav vector for a given segment length
    bowav = build_bowav_from_centroid_assignments(
        X, n_centroids, best_training_segment_length
    )
    del X

    n_segments_per_time_series = bowav.shape[1]
    # vertically concatenate train BoWav vectors: (m, n, p) -> (m*n, p)
    bowav = np.vstack(bowav)
    # expand train labels to match train BoWav vectors
    y = np.repeat(y, n_segments_per_time_series)
    # expand train sample weights to match train BoWav vectors
    sample_weight = np.repeat(sample_weight, n_segments_per_time_series)

    logging.info("Fitting best estimator")
    named_steps = getattr(estimator, "named_steps", None)
    if named_steps is not None:
        best_estimator.fit(bowav, y, clf__sample_weight=sample_weight)
    else:
        best_estimator.fit(bowav, y, sample_weight=sample_weight)

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

    # TODO: Use a simpler file name. Maybe using only the best parameters, and
    # create a corresponging README or log file with the rest of the information,
    # like the full HPO grid.
    results_file = f"BoWav_random_forest_seglen{best_training_segment_length}_expw{int(best_expert_weight)}.pkl"
    results_file = results_folder.joinpath(results_file)
    with results_file.open("wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    logging.info("Finished")
