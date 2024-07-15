import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import f1_score

from icwaves.feature_extractors.bowav import build_bowav_from_centroid_assignments


def build_bowav_based_on_aggregation_method(
    centroid_assignments,
    n_centroids,
    agg_method,
    n_validation_windows_per_segment,
    n_training_windows_per_segment,
    subj_mask=slice(None),
):
    if agg_method == "count_pooling":
        bowav_test = build_bowav_from_centroid_assignments(
            centroid_assignments[subj_mask],
            n_centroids,
            n_validation_windows_per_segment,
        )
        bowav_test = np.expand_dims(bowav_test[:, 0, :], axis=1)
    else:
        # TODO: using `n_training_windows_per_segment` does not make sense,
        # as the test segment length might be smaller. Discuss this with AJB.
        bowav_test = build_bowav_from_centroid_assignments(
            centroid_assignments[subj_mask],
            n_centroids,
            n_training_windows_per_segment,
        )
        n_segments_per_time_series = bowav_test.shape[1]
        if n_validation_windows_per_segment is not None:
            n_validation_segments_per_time_series = (
                n_segments_per_time_series * n_training_windows_per_segment
            ) // n_validation_windows_per_segment

            n_train_segments_per_validation_segment = (
                n_segments_per_time_series // n_validation_segments_per_time_series
            )
            bowav_test = bowav_test[:, :n_train_segments_per_validation_segment, :]

    return bowav_test


def compute_brain_F1_score_per_subject(
    clf,
    centroid_assignments,
    labels,
    expert_label_mask,
    n_centroids,
    agg_method,
    n_validation_windows_per_segment,
    n_training_windows_per_segment,
    subj_mask=slice(None),
):

    bowav_test = build_bowav_based_on_aggregation_method(
        centroid_assignments,
        n_centroids,
        agg_method,
        n_validation_windows_per_segment,
        n_training_windows_per_segment,
        subj_mask,
    )

    n_segments_per_time_series = bowav_test.shape[1]
    # vertically concatenate test BoWav vectors: (m, n, p) -> (m*n, p)
    bowav_test = np.vstack(bowav_test)
    y_pred = clf.predict(bowav_test)

    # Maybe aggregate output
    if agg_method == "majority_vote":
        # Aggregate all the predictions
        # TODO: include None in n_validation_windows_per_segment_arr
        # to get all the time series
        if n_validation_windows_per_segment is None:
            y_pred = y_pred.reshape(-1, n_segments_per_time_series)
            y_pred = scipy.stats.mode(y_pred, axis=1)[0]
            n_segments_per_time_series = 1

    # expand labels and expert mask to match test BoWav vectors
    y = np.repeat(labels[subj_mask], n_segments_per_time_series)
    ext_expert_label_mask = np.repeat(
        expert_label_mask[subj_mask], n_segments_per_time_series
    )

    y_expert = y[ext_expert_label_mask]
    y_pred_expert = y_pred[ext_expert_label_mask]
    brain_f1_score = f1_score(y_expert, y_pred_expert, labels=[0], average=None)

    return brain_f1_score.item()


def jackknife_stddev(group):
    subject_ids = group["Subject ID"].unique()
    n = len(subject_ids)
    means = []

    # Perform jackknife resampling
    for subject_id in subject_ids:
        jackknife_sample = group[group["Subject ID"] != subject_id]
        means.append(jackknife_sample["Brain F1 score [holdout]"].mean())

    # Calculate the jackknife estimate of the mean
    jackknife_mean = np.mean(means)

    # Calculate the jackknife estimate of the standard deviation
    squared_diffs = [(mean - jackknife_mean) ** 2 for mean in means]
    jackknife_variance = (n - 1) / n * np.sum(squared_diffs)
    jackknife_std = np.sqrt(jackknife_variance)

    return pd.Series(
        {"Jackknife Mean": jackknife_mean, "Jackknife StdDev": jackknife_std}
    )
