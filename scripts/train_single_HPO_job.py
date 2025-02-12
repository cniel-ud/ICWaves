import logging
from pathlib import Path
import pickle
import shutil
import numpy as np
from numpy.random import default_rng
import scipy
import sklearn
from sklearn.metrics import f1_score

from icwaves.factories import create_estimator
from icwaves.model_selection.split import LeaveOneSubjectOutExpertOnly
from icwaves.model_selection.validation import _fit_and_score
from icwaves.model_selection.job_utils import get_job_parameters
from icwaves.file_utils import read_args_from_file
from icwaves.argparser import (
    create_argparser_all_params,
    create_argparser_one_parameter_one_split,
)
from icwaves.data.loading import get_data_and_feature_extractor
from icwaves.model_selection.hpo_utils import get_base_parameters, build_grid_parameters


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
    data_bundles, feature_extractor = get_data_and_feature_extractor(args)

    # the DataBundle for bowav and psd_autocorr only differs in the `data` attribute
    # TODO: find a more efficient way of doing this
    data_bundle = (
        data_bundles["bowav"]
        if "bowav" in data_bundles
        else data_bundles["psd_autocorr"]
    )

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

    X = {k: v.data for k, v in data_bundles.items()}
    result = _fit_and_score(
        clf,
        X=X,
        y=data_bundle.labels,
        expert_label_mask=data_bundle.expert_label_mask,
        train=job_params.train_indices,
        test=job_params.test_indices,
        parameters=job_params.parameters,
        scorer=f1_score,
        feature_extractor=feature_extractor,
        split_progress=(job_params.split_index, job_params.n_splits),
        candidate_progress=(job_params.candidate_index, job_params.n_candidates),
    )

    valseglen = (
        "None"
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

    if job_id == 0:
        # Save a copy of the config file in the results directory
        shutil.copy(one_run_args.path_to_config_file, results_folder)

    # Add to results the version of scikit-learn, numpy, and
    # scipy to improve reproducibility
    result["sklearn_version"] = sklearn.__version__
    result["numpy_version"] = np.__version__
    result["scipy_version"] = scipy.__version__

    with results_file.open("wb") as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    logging.info("Finished")
