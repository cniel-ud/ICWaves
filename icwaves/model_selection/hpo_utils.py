import logging
import copy
import pickle
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._validation import _aggregate_score_dicts
from icwaves.feature_extractors.utils import calculate_segment_length
from icwaves.model_selection.utils import _store

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

    candidate_params = list(ParameterGrid(candidate_params))

    return candidate_params


def get_grid_size(candidate_params, cv, data_bundle):
    n_candidates = len(candidate_params)
    n_splits = cv.get_n_splits(
        data_bundle.data,
        data_bundle.labels,
        groups=data_bundle.subj_ind,
    )
    return n_candidates, n_splits


def load_candidate_results(results_path, n_candidates, n_splits):
    """Load and aggregate results from hyperparameter optimization runs.

    Args:
        results_path (Path): Path to the directory containing results
        n_candidates (int): Number of hyperparameter candidates
        n_splits (int): Number of cross-validation splits

    Returns:
        dict: Aggregated results from all runs
    """
    all_out = []
    for candidate_idx in range(n_candidates):
        for split_idx in range(n_splits):
            file = results_path.joinpath(
                f"candidate_{candidate_idx}_split_{split_idx}.pkl"
            )
            with open(file, "rb") as f:
                all_out.append(pickle.load(f))
    return _aggregate_score_dicts(all_out)


def get_best_parameters(results):
    """Extract and process the best parameters from results.

    Args:
        results (dict): Dictionary containing evaluation results

    Returns:
        tuple: (best_params, best_expert_weight, best_training_segment_length, n_centroids)
    """
    best_index = results["rank_test_scores"].argmin()
    best_params = copy.deepcopy(results["params"][best_index])
    best_score = results["mean_test_scores"][best_index]

    logging.info("Best score: %s", best_score)
    logging.info("Best params: %s", best_params)

    return best_params


def process_candidate_results(args, n_candidates, n_splits, candidate_params):
    """Process all candidate results and return the best estimator configuration.

    Args:
        args: Arguments containing path information
        estimator: Base estimator to clone
        n_candidates (int): Number of hyperparameter candidates
        n_splits (int): Number of cross-validation splits
        candidate_params (list): List of parameter dictionaries

    Returns:
        tuple: (best_params, results)
    """
    valseglen = (
        "none"
        if args.validation_segment_length == -1
        else int(args.validation_segment_length)
    )
    results_path = Path(
        args.path_to_results,
        f"{args.classifier_type}_{args.feature_extractor}_valSegLen{valseglen}",
    )
    all_out = load_candidate_results(results_path, n_candidates, n_splits)

    timing_results = {
        **_store("fit_time", all_out["fit_time"], n_splits, n_candidates),
        **_store("score_time", all_out["score_time"], n_splits, n_candidates),
    }

    results = {**timing_results, "params": candidate_params}

    results.update(
        _store(
            "test_scores",
            all_out["test_scores"],
            n_splits,
            n_candidates,
            splits=True,
            rank=True,
            weights=None,
        )
    )

    best_params = get_best_parameters(results)

    return best_params, results
