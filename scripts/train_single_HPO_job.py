# scripts/train_classifier_single_HPO_job.py
from itertools import product
import logging
from pathlib import Path
import pickle
from typing import Dict
import numpy as np
from numpy.random import default_rng
import scipy
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

from icwaves.factories import create_estimator, create_feature_extractor
from icwaves.feature_extractors.bowav import (
    build_or_load_centroid_assignments_and_labels,
)
from icwaves.model_selection.split import LeaveOneSubjectOutExpertOnly
from icwaves.model_selection.validation import _fit_and_score
from icwaves.model_selection.job_utils import get_job_parameters
from icwaves.file_utils import read_args_from_file
from icwaves.argparser import (
    create_argparser_all_params,
    create_argparser_one_parameter_one_split,
)
from icwaves.preprocessing import load_or_build_ics_and_labels
from icwaves.data.loading import get_data_and_feature_extractor
from icwaves.feature_extractors.utils import calculate_segment_length

TF_IDF_NORM_MAP = {
    "none": None,
    "l1": "l1",
    "l2": "l2",
}


def get_base_parameters(args, rng):
    if args.classifier_type == "logistic":
        params = dict(
            penalty=args.penalty,
            max_iter=args.max_iter,
            random_state=rng,
            class_weight="balanced",
            solver="saga",
            warm_start=True,
            multi_class="multinomial",
        )
    elif args.classifier_type == "random_forest":
        params = dict(
            n_estimators=300,
            random_state=rng,
            class_weight="balanced",
        )
    return params


def build_grid_parameters(args, srate):
    candidate_params = {}
    candidate_params["input_or_output_aggregation_method"] = [
        "count_pooling",
        "majority_vote",
    ]
    candidate_params["training_segment_length"] = calculate_segment_length(
        args, srate, train=True
    )
    candidate_params["validation_segment_length"] = calculate_segment_length(
        args, srate, train=False
    )
    candidate_params["expert_weight"] = args.expert_weight
    if args.classifier_type == "logistic":
        if args.feature_extractor == "bowav":
            candidate_params["clf__C"] = args.regularization_factor
            candidate_params["clf__l1_ratio"] = args.l1_ratio
            candidate_params["scaler__norm"] = [
                TF_IDF_NORM_MAP[norm] for norm in args.tf_idf_norm
            ]
        else:
            candidate_params["C"] = args.regularization_factor
            candidate_params["l1_ratio"] = args.l1_ratio
    elif args.classifier_type == "random_forest":
        candidate_params["min_samples_split"] = args.min_samples_split

    return candidate_params


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s:%(filename)s:%(message)s", level=logging.DEBUG
    )
    logging.info("Started")

    # Parse arguments
    parser = create_argparser_one_parameter_one_split()
    one_run_args = parser.parse_args()
    args_list = read_args_from_file(one_run_args.path_to_config_file)
    job_id = one_run_args.job_id

    all_params_parser = create_argparser_all_params(one_run_args.feature_extractor)
    args = all_params_parser.parse_args(args_list)
    args.feature_extractor = one_run_args.feature_extractor

    # Setup RNG
    new_rng = default_rng(13)
    old_rng = np.random.RandomState(13)

    # Load or prepare data based on feature extractor type
    data_bundle, feature_extractor = get_data_and_feature_extractor(args)

    # Create cross-validation splitter
    cv = LeaveOneSubjectOutExpertOnly(data_bundle.expert_label_mask)
    params = get_base_parameters(args, old_rng)
    clf = create_estimator(args.classifier_type, args.feature_extractor, **params)
    logging.info(f"clf: {clf}")
    candidate_params = build_grid_parameters(args, data_bundle.srate)

    job_params = get_job_parameters(job_id, data_bundle, cv, candidate_params)
    # '0' is the 'brain' class. We want to compute the F1-score for this class only.
    job_params.parameters["scorer_kwargs"] = {"labels": [0], "average": None}

    # log candidate and split id
    logging.info(f"candidate_index: {job_params.candidate_index}")
    logging.info(f"split_index: {job_params.split_index}")

    result = _fit_and_score(
        clf,
        data_bundle.data,
        data_bundle.labels,
        data_bundle.expert_label_mask,
        train=job_params.train_indices,
        test=job_params.test_indices,
        parameters=job_params.parameters,
        scorer=f1_score,
        feature_extractor=feature_extractor,
        split_progress=(job_params.split_index, job_params.n_splits),
        candidate_progress=(job_params.candidate_index, job_params.n_candidates),
    )

    valseglen = (
        "none"
        if args.validation_segment_length == -1
        else int(args.validation_segment_length)
    )
    results_folder = Path(
        args.path_to_results,
        f"{args.classifier_type}_{args.feature_extractor}_valSegLen{valseglen}",
    )
    results_folder.mkdir(exist_ok=True, parents=True)

    results_file = results_folder.joinpath(
        f"candidate_{job_params.candidate_index}_split_{job_params.split_index}.pkl"
    )

    # Add to results the version of scikit-learn, numpy, and
    # scipy to improve reproducibility
    result["sklearn_version"] = sklearn.__version__
    result["numpy_version"] = np.__version__
    result["scipy_version"] = scipy.__version__

    with results_file.open("wb") as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    logging.info("Finished")
