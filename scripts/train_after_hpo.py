import copy
from pathlib import Path
import logging
import pickle
import time
import numpy as np
from numpy.random import default_rng
import scipy
import sklearn
from sklearn.base import clone

from icwaves.factories import create_estimator
from icwaves.file_utils import (
    get_cmmn_suffix,
    get_validation_segment_length_string,
    read_args_from_file,
)
from icwaves.model_selection.hpo_utils import (
    process_candidate_results,
    get_base_parameters,
)
from icwaves.model_selection.split import LeaveOneSubjectOutExpertOnly
from icwaves.argparser import (
    create_argparser_aggregate_results,
    create_argparser_all_params,
)
from icwaves.data.loading import get_feature_extractor, load_data_bundles


def train_final_model(
    estimator, X, y, expert_label_mask, best_params, feature_extractor=None
):
    """Train the final model using the provided best parameters.

    Parameters
    ----------
    estimator : estimator object
        The base estimator to be trained
    X : Dict[str, array-like]
        Training data for each feature in {'bowav', 'psd_autocorr'}
    y : array-like
        Target values
    expert_label_mask : array-like
        Boolean mask indicating expert-labeled samples
    best_params : dict
        Dictionary containing the best parameters from HPO
    feature_extractor : callable, optional
        Function to extract features from X. If None, X is used directly

    Returns
    -------
    estimator
        The fitted estimator with best parameters
    """
    # Make a deep copy of the best parameters
    best_params = copy.deepcopy(best_params)
    # Extract special parameters
    best_expert_weight = best_params.pop("expert_weight", 1)
    best_training_segment_length = best_params.pop("training_segment_length")

    # Remove unused parameters if they exist
    for param in [
        "validation_segment_length",
        "input_or_output_aggregation_method",
    ]:
        best_params.pop(param, None)

    # Create and configure best estimator
    best_estimator = clone(clone(estimator).set_params(**best_params))

    # Prepare sample weights
    feature_extractors = list(X.keys())
    sample_weight = np.ones(X[feature_extractors[0]].shape[0])
    sample_weight[expert_label_mask] = best_expert_weight

    # Feature extraction if needed
    if feature_extractor is not None:
        X = feature_extractor(X, best_training_segment_length)

    # Reshape data
    n_segments = X.shape[1]
    X = np.vstack(X)
    y = np.repeat(y, n_segments)
    sample_weight = np.repeat(sample_weight, n_segments)

    # Fit the model
    logging.info("Fitting best estimator")
    refit_start_time = time.time()
    named_steps = getattr(estimator, "named_steps", None)
    if named_steps is not None:
        best_estimator.fit(X, y, clf__sample_weight=sample_weight)
    else:
        best_estimator.fit(X, y, sample_weight=sample_weight)
    refit_end_time = time.time()
    refit_time = refit_end_time - refit_start_time

    return best_estimator, refit_time


if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s:%(filename)s:%(message)s", level=logging.DEBUG
    )
    logging.info("Started")

    # Parse arguments from config file
    parser = create_argparser_aggregate_results()
    agg_args = parser.parse_args()
    args_list = read_args_from_file(agg_args.path_to_config_file)
    all_params_parser = create_argparser_all_params(agg_args.feature_extractor)
    args, _ = all_params_parser.parse_known_args(args_list)
    args.feature_extractor = agg_args.feature_extractor

    # Setup RNG
    new_rng = default_rng(13)
    old_rng = np.random.RandomState(13)

    # Load or prepare data based on feature extractor type
    data_bundles = load_data_bundles(args)
    feature_extractor = get_feature_extractor(args.feature_extractor, data_bundles)

    data_bundle = (
        data_bundles["bowav"]
        if "bowav" in data_bundles
        else data_bundles["psd_autocorr"]
    )

    # Create cross-validation splitter and estimator
    cv = LeaveOneSubjectOutExpertOnly(data_bundle.expert_label_mask)
    params = get_base_parameters(args, old_rng)
    clf = create_estimator(args.classifier_type, args.feature_extractor, **params)
    logging.info(f"clf: {clf}")

    # Get best parameters from HPO results
    best_params, results = process_candidate_results(
        args, cv, data_bundle.srate, data_bundle.subj_ind
    )

    # Train final model with best parameters
    X = {k: v.data for k, v in data_bundles.items()}
    best_estimator, refit_time = train_final_model(
        clf,
        X,
        data_bundle.labels,
        data_bundle.expert_label_mask,
        best_params,
        feature_extractor,
    )

    results.update(
        {
            "best_params": best_params,
            "refit_time": refit_time,
            "best_estimator": best_estimator,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        }
    )

    # Save final model and results
    valseglen = get_validation_segment_length_string(
        int(args.validation_segment_length)
    )
    cmmn_suffix = get_cmmn_suffix(args.cmmn_filter)
    results_file = Path(
        args.path_to_results,
        f"train_{args.classifier_type}_{args.feature_extractor}_valSegLen{valseglen}{cmmn_suffix}.pkl",
    )
    with results_file.open("wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    logging.info(f"Final model saved to {results_file}")
    logging.info("Finished")
