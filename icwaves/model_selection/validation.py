import time

import numpy as np
from joblib import logger


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
    candidate_progress
):

    progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
    progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"
    sorted_keys = sorted(parameters)  # Ensure deterministic o/p
    params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    start_msg = f"[CV{progress_msg}] START {params_msg}"
    print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Build sample_weights
    expert_weight = parameters.pop('expert_weight', 1)
    sample_weight = np.ones(X.shape[0])
    sample_weight[expert_label_mask] = expert_weight

    # Get input/output aggregation method
    input_or_output_aggregation_method = parameters.pop(
        'input_or_output_aggregation_method')

    # Get segment length
    segment_length = parameters.pop('segment_length')

    # Get minutes for validation
    minutes_for_validation = parameters.pop('minutes_for_validation')

    # Set classifier params
    estimator.set_params(**parameters)

    result = {}
    start_time = time.time()
    X_train, y_train = X[train], y[train]
    sample_weight_train = sample_weight[train]

    # Get (BoWav) counts per segment
    # TODO:
    #  * Create function in bowav.py that takes X_test and return the counts
    #    for each segment (a BoWav vector).
    #  * If the input_or_output_aggregation_method is 'count_pooling', then
    #    aggregate the BoWav vectors until completing minutes_for_validation,
    #    or aggregate all BoWav vectors if minutes_for_validation==-1
    #  * If the input_or_output_aggregation_method is 'majority_vote', then
    #    perform prediction, and then aggregate the predictions until completing
    #    minutes_for_validation, or aggregate all predictions if minutes_for_validation==-1

    named_steps = getattr(estimator, 'named_steps', None)
    if named_steps is not None:
        estimator.fit(X_train, y_train, clf__sample_weight=sample_weight_train)
    else:
        estimator.fit(X_train, y_train, sample_weight=sample_weight_train)

    fit_time = time.time() - start_time

    X_test, y_test = X[test], y[test]

    # Maybe aggregate input
    if input_or_output_aggregation_method == 'count_pooling':
        X_test = np.sum(X_test, axis=1)
        X_test = X_test.reshape(-1, 1)


    y_pred = estimator.predict(X_test)
    sample_weight_test = sample_weight[test]
    test_scores = scorer(
        y_test, y_pred, sample_weight=sample_weight_test)
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

    parameters.update({'expert_weight': expert_weight})
    result["test_scores"] = test_scores
    result["fit_time"] = fit_time
    result["score_time"] = score_time
    result["parameters"] = parameters

    return result
