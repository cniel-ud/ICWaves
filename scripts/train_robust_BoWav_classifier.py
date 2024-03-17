# %%
from itertools import product
import logging
from pathlib import Path
import pickle

import numpy as np
from numpy.random import default_rng
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from icwaves.data_loaders import load_codebooks_wrapper
from icwaves.feature_extractors.bowav import (
    build_or_load_centroid_assignments,
)
from icwaves.model_selection.split import LeaveOneSubjectOutExpertOnly
from icwaves.model_selection.validation import _fit_and_score
from icwaves.preprocessing import load_or_build_preprocessed_data
from icwaves.utils import read_args_from_file
from icwaves.argparser import (
    create_argparser_all_params,
    create_argparser_one_parameter_one_split,
)
import sklearn
import scipy


BOWAV_NORM_MAP = {
    "none": None,
    "l_1": 1,
    "l_2": 2,
    "l_inf": np.inf,
}

if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s:%(filename)s:%(message)s", level=logging.DEBUG
    )
    logging.info("Started")

    parser = create_argparser_one_parameter_one_split()
    one_run_args = parser.parse_args()

    args_list = read_args_from_file(one_run_args.path_to_config_file)
    job_id = one_run_args.job_id

    all_params_parser = create_argparser_all_params()
    args = all_params_parser.parse_args(args_list)

    new_rng = default_rng(13)
    # scikit-learn doesn't support the new numpy Generator:
    old_rng = np.random.RandomState(13)

    # Load or build preprocessed data
    (
        windowed_ics,
        labels,
        srate,
        expert_label_mask,
        subj_ind,
    ) = load_or_build_preprocessed_data(args)

    # Load codebooks
    codebooks = load_codebooks_wrapper(args, srate)
    n_centroids = codebooks[0].shape[0]

    # Load or build centroid assignments
    centroid_assignments = build_or_load_centroid_assignments(
        args, windowed_ics, codebooks
    )
    del windowed_ics, codebooks
    logging.info(f"centroid_assignments.shape: {centroid_assignments.shape}")
    logging.info(f"centroid_assignments.dtype: {centroid_assignments.dtype}")
    logging.info(f"centroid_assignments[13, 0]: {centroid_assignments[13, 0]}")

    input_or_output_aggregation_method = ["count_pooling", "majority_vote"]

    training_segment_length = args.training_segment_length
    # Compute n_training_windows_per_segment
    n_training_windows_per_segment = [
        int(s / args.window_length) for s in training_segment_length
    ]
    logging.info(f"n_training_windows_per_segment: {n_training_windows_per_segment}")

    validation_segment_length = args.validation_segment_length
    # Compute n_validation_windows_per_segment
    if validation_segment_length == -1:
        n_validation_windows_per_segment = None
    else:
        n_validation_windows_per_segment = int(
            validation_segment_length / args.window_length
        )
    logging.info(
        f"n_validation_windows_per_segment: {n_validation_windows_per_segment}"
    )

    cv = LeaveOneSubjectOutExpertOnly(expert_label_mask)

    pipe = Pipeline([("scaler", TfidfTransformer()), ("clf", LogisticRegression())])

    clf_params = dict(
        clf__class_weight="balanced",
        clf__solver="saga",
        clf__penalty=args.penalty,
        clf__random_state=old_rng,
        clf__multi_class="multinomial",
        clf__warm_start=True,
        clf__max_iter=args.max_iter,
    )
    pipe.set_params(**clf_params)

    # Wrap single-value params as a list. TODO: check and convert for all params
    n_centroids = [n_centroids]
    n_validation_windows_per_segment = [n_validation_windows_per_segment]

    candidate_params = dict(
        clf__C=args.regularization_factor,
        clf__l1_ratio=args.l1_ratio,
        expert_weight=args.expert_weight,
        bowav_norm=args.bowav_norm,
        input_or_output_aggregation_method=input_or_output_aggregation_method,
        n_training_windows_per_segment=n_training_windows_per_segment,
        n_validation_windows_per_segment=n_validation_windows_per_segment,
        n_centroids=n_centroids,
    )

    candidate_params = list(ParameterGrid(candidate_params))
    n_candidates = len(candidate_params)
    list_of_candidate_params_and_splits = list(
        product(
            enumerate(candidate_params),
            enumerate(cv.split(centroid_assignments, labels, subj_ind)),
        )
    )
    (cand_idx, parameters), (split_idx, (train, test)) = (
        list_of_candidate_params_and_splits[job_id]
    )

    n_splits = cv.get_n_splits(centroid_assignments, labels, groups=subj_ind)

    result = _fit_and_score(
        pipe,
        centroid_assignments,
        labels,
        expert_label_mask,
        train=train,
        test=test,
        parameters=parameters,
        scorer=balanced_accuracy_score,
        split_progress=(split_idx, n_splits),
        candidate_progress=(cand_idx, n_candidates),
    )

    results_folder = Path(args.path_to_results, "temporal_results")
    results_folder.mkdir(exist_ok=True, parents=True)
    results_file = f"candidate_{cand_idx}_split_{split_idx}.pkl"
    results_file = results_folder.joinpath(results_file)

    # Add to results the version of scikit-learn, numpy, and
    # scipy to improve reproducibility
    result["sklearn_version"] = sklearn.__version__
    result["numpy_version"] = np.__version__
    result["scipy_version"] = scipy.__version__

    with results_file.open("wb") as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    logging.info("Finished")
