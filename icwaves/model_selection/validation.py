import time

import numpy as np
import scipy
from joblib import logger
import logging
from icwaves.model_selection.utils import _gather


def _fit_and_score(
    estimator,
    X,
    y,
    expert_label_mask,
    train,
    test,
    parameters,
    scorer,
    feature_extractor,
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
    feature_extractors = list(X.keys())
    sample_weight = np.ones(X[feature_extractors[0]].shape[0])
    sample_weight[expert_label_mask] = expert_weight

    # Get input/output aggregation method
    input_or_output_aggregation_method = parameters.pop(
        "input_or_output_aggregation_method"
    )

    # Get training_segment_length
    train_segment_length = parameters.pop("training_segment_length")

    # Get validation_segment_length
    validation_segment_length = parameters.pop("validation_segment_length")

    # Get scorer_kwargs
    scorer_kwargs = parameters.pop("scorer_kwargs", {})

    # Set classifier params
    estimator.set_params(**parameters)

    result = {}
    start_time = time.time()
    X_train, y_train = _gather(X, y, train)
    sample_weight_train = sample_weight[train]

    # Build train feature vector for a given segment length
    X_train = feature_extractor(X_train, train_segment_length)

    # X_train.shape = (n_time_series, n_segments, n_features)
    n_segments = X_train.shape[1]
    # vertically concatenate train feature vectors: (m, n, p) -> (m*n, p)
    X_train = np.vstack(X_train)
    # expand train labels to match train feature vectors
    y_train = np.repeat(y_train, n_segments)
    # expand train sample weights to match train feature vectors
    sample_weight_train = np.repeat(sample_weight_train, n_segments)

    named_steps = getattr(estimator, "named_steps", None)
    if named_steps is not None:
        estimator.fit(X_train, y_train, clf__sample_weight=sample_weight_train)
    else:
        estimator.fit(X_train, y_train, sample_weight=sample_weight_train)

    del X_train, y_train, sample_weight_train

    fit_time = time.time() - start_time

    X_test, y_test = _gather(X, y, test)
    sample_weight_test = sample_weight[test]

    # Aggregate input at either training segment length or validation segment length
    # TODO: rename `count_pooling` to `pooling`
    if input_or_output_aggregation_method == "count_pooling":
        X_test = feature_extractor(X_test, validation_segment_length)
    else:
        X_test = feature_extractor(X_test, train_segment_length)

    n_segments = X_test.shape[1]
    # vertically concatenate test BoWav vectors: (m, n, p) -> (m*n, p)
    X_test = np.vstack(X_test)

    y_pred = estimator.predict(X_test)
    del X_test

    # TODO: improve this:
    validation_segment_length = (
        validation_segment_length["bowav"]
        if "bowav" in validation_segment_length
        else validation_segment_length["psd_autocorr"]
    )
    train_segment_length = (
        train_segment_length["bowav"]
        if "bowav" in train_segment_length
        else train_segment_length["psd_autocorr"]
    )

    # Predictions were made on segments of length training_segment_length.
    if input_or_output_aggregation_method == "majority_vote":
        # Aggregate all the predictions
        if validation_segment_length is None:
            y_pred = y_pred.reshape(-1, n_segments)
            y_pred = scipy.stats.mode(y_pred, axis=1)[0]
            n_segments = 1
        # Aggregate predictions every validation_segment_length > training_segment_length
        else:
            n_validation_segments = (
                n_segments * train_segment_length
            ) // validation_segment_length

            n_train_segments_per_validation_segment = (
                n_segments // n_validation_segments
            )

            # Discard some predictions if the number of training segments is not
            # divisible by the number of validation segments.
            if n_segments % n_validation_segments:
                trimmed_n_segments = (
                    n_train_segments_per_validation_segment * n_validation_segments
                )
                y_pred = y_pred.reshape(-1, n_segments)
                y_pred = y_pred[:, :trimmed_n_segments]
                y_pred = y_pred.reshape(-1)
                logging.warning(
                    f"Trimming y_pred from {n_segments} to "
                    f"{trimmed_n_segments} segments per time series."
                )

            y_pred = y_pred.reshape(-1, n_train_segments_per_validation_segment)
            y_pred = scipy.stats.mode(y_pred, axis=1)[0]
            n_segments = n_validation_segments

    # expand test labels to match test BoWav vectors
    y_test = np.repeat(y_test, n_segments)
    # expand test sample weights to match test BoWav vectors
    sample_weight_test = np.repeat(sample_weight_test, n_segments)

    test_scores = scorer(
        y_test, y_pred, sample_weight=sample_weight_test, **scorer_kwargs
    )

    # This accounts for cases where we want to compute a metric on a single class
    # e.g., F1-score on brain
    if isinstance(test_scores, np.ndarray) and len(test_scores) == 1:
        test_scores = test_scores[0]

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
    parameters.update(
        {"input_or_output_aggregation_method": input_or_output_aggregation_method}
    )
    parameters.update({"training_segment_length": train_segment_length})
    parameters.update({"validation_segment_length": validation_segment_length})
    result["test_scores"] = test_scores
    result["fit_time"] = fit_time
    result["score_time"] = score_time
    result["parameters"] = parameters

    return result
