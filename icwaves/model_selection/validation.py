import time

import numpy as np
import scipy
from joblib import logger

from icwaves.feature_extractors.bowav import build_bowav_from_centroid_assignments


def _fit_and_score(
    estimator,
    X,
    y,
    expert_label_mask,
    train,
    test,
    parameters,
    scorer,
    split_progress,
    candidate_progress,
):
    progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
    progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"
    sorted_keys = sorted(parameters)  # Ensure deterministic o/p
    params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    start_msg = f"[CV{progress_msg}] START {params_msg}"
    print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Build sample_weights
    expert_weight = parameters.pop("expert_weight", 1)
    sample_weight = np.ones(X.shape[0])
    sample_weight[expert_label_mask] = expert_weight

    # Get bowav norm
    bowav_norm = parameters.pop("bowav_norm")

    # Get input/output aggregation method
    input_or_output_aggregation_method = parameters.pop(
        "input_or_output_aggregation_method"
    )

    # Get n_training_windows_per_segment
    n_training_windows_per_segment = parameters.pop("n_training_windows_per_segment")

    # Get n_validation_windows_per_segment
    n_validation_windows_per_segment = parameters.pop(
        "n_validation_windows_per_segment"
    )

    # Get n_centroids
    n_centroids = parameters.pop("n_centroids")

    # Set classifier params
    estimator.set_params(**parameters)

    result = {}
    start_time = time.time()
    X_train, y_train = X[train], y[train]
    sample_weight_train = sample_weight[train]

    # Build train BoWav vector for a given segment length
    bowav_train = build_bowav_from_centroid_assignments(
        X_train, n_centroids, n_training_windows_per_segment, bowav_norm
    )
    n_segments_per_time_series = bowav_train.shape[1]
    # vertically concatenate train BoWav vectors: (m, n, p) -> (m*n, p)
    bowav_train = np.vstack(bowav_train)
    # expand train labels to match train BoWav vectors
    y_train = np.repeat(y_train, n_segments_per_time_series)
    # expand train sample weights to match train BoWav vectors
    sample_weight_train = np.repeat(sample_weight_train, n_segments_per_time_series)

    named_steps = getattr(estimator, "named_steps", None)
    if named_steps is not None:
        estimator.fit(bowav_train, y_train, clf__sample_weight=sample_weight_train)
    else:
        estimator.fit(bowav_train, y_train, sample_weight=sample_weight_train)

    fit_time = time.time() - start_time

    X_test, y_test = X[test], y[test]
    sample_weight_test = sample_weight[test]

    # Aggregate input at either training segment length or validation segment length
    if input_or_output_aggregation_method == "count_pooling":
        bowav_test = build_bowav_from_centroid_assignments(
            X_test, n_centroids, n_validation_windows_per_segment, bowav_norm
        )
    else:
        bowav_test = build_bowav_from_centroid_assignments(
            X_test, n_centroids, n_training_windows_per_segment, bowav_norm
        )

    n_segments_per_time_series = bowav_test.shape[1]
    # vertically concatenate test BoWav vectors: (m, n, p) -> (m*n, p)
    bowav_test = np.vstack(bowav_test)

    y_pred = estimator.predict(bowav_test)

    # Maybe aggregate output
    if input_or_output_aggregation_method == "majority_vote":
        y_pred = y_pred.reshape(-1, n_segments_per_time_series)
        y_pred = scipy.stats.mode(y_pred, axis=1)[0]
    else:
        # expand test labels to match test BoWav vectors
        y_test = np.repeat(y_test, n_segments_per_time_series)
        # expand test sample weights to match test BoWav vectors
        sample_weight_test = np.repeat(sample_weight_test, n_segments_per_time_series)

    test_scores = scorer(y_test, y_pred, sample_weight=sample_weight_test)
    score_time = time.time() - start_time - fit_time

    total_time = score_time + fit_time
    end_msg = f"[CV{progress_msg}] END "
    result_msg = params_msg + (";" if params_msg else "")
    result_msg += ", score="
    result_msg += f"{test_scores:.3f}"
    result_msg += f" total time={logger.short_format_time(total_time)}"
    end_msg += "." * (80 - len(end_msg) - len(result_msg))
    end_msg += result_msg
    print(end_msg)

    parameters.update({"expert_weight": expert_weight})
    parameters.update({"bowav_norm": bowav_norm})
    parameters.update(
        {"input_or_output_aggregation_method": input_or_output_aggregation_method}
    )
    parameters.update(
        {"n_training_windows_per_segment": n_training_windows_per_segment}
    )
    parameters.update(
        {"n_validation_windows_per_segment": n_validation_windows_per_segment}
    )
    result["test_scores"] = test_scores
    result["fit_time"] = fit_time
    result["score_time"] = score_time
    result["parameters"] = parameters

    return result
