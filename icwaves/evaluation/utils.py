import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import f1_score


def build_features_based_on_aggregation_method(
    feature_extractor,
    X,
    validation_segment_length,
    training_segment_length,
    agg_method,
    subj_mask=slice(None),
):
    """
    Arguments
    ---------
    feature_extractor: function used to extract features from X
    X: input data. Shape and meaning depend on the feature extractor used,
                   but X.shape[0] is the number of samples, and X.shape[-1]
                   is the length of each sample.
    validation_segment_length: length of the validation segment. If None,
                               use the entire sample. If not None, it
                               should be larger than or equal to the training segment length.
    training_segment_length: length of the training segment
    agg_method: method used to aggregate the features or the predictions.
                With "count_pooling", the features are computed over a segment of length
                validation_segment_length, and predictions over such segment.
                With "majority_vote", the features are computed over segments of length
                training_segment_length, predictions over all such segments, and the mode
                of the predictions is taken.
    subj_mask: mask to select subjects from X

    Returns
    -------
    features: features computed according to agg_method. Shape is (n_samples, n_segments, n_features),
              where n_segments is equal 1 if agg_method is "count_pooling", or equal to
              k = floor(X.shape[-1] / training_segment_length) if agg_method is "majority_vote".
    """
    # X is a dict. Make a deep copy to avoid messing with upstream data.
    X = {k: np.copy(v) for k, v in X.items()}
    for k in X.keys():
        X[k] = X[k][subj_mask, ..., slice(0, validation_segment_length[k])]

    if agg_method == "count_pooling":
        features = feature_extractor(
            X,
            validation_segment_length,
        )
    else:
        for k in X.keys():
            if validation_segment_length[k] is not None:
                assert training_segment_length[k] <= validation_segment_length[k]

        features = feature_extractor(
            X,
            training_segment_length,
        )

    return features


def compute_brain_F1_score_per_subject(
    clf,
    X,
    labels,
    expert_label_mask,
    agg_method,
    feature_extractor,
    validation_segment_length,
    training_segment_length,
    subj_mask=slice(None),
):
    """Compute the brain F1 score for each subject. It assumes that the label for the 'brain' ICLabel
    class is 0.

    Arguments
    ---------
    clf: classifier used to predict the labels
    X: input data. Shape and meaning depend on the feature extractor used,
                   but X.shape[0] is the number of samples, and X.shape[-1]
                   is the length of each sample.
    labels: labels of the input data
    expert_label_mask: mask to select the expert labels
    agg_method: method used to aggregate the features or the predictions.
                With "count_pooling", the features are computed over a segment of length
                validation_segment_length, and predictions over such segment.
                With "majority_vote", the features are computed over segments of length
                training_segment_length, predictions over all such segments, and the mode
                of the predictions is taken.
    feature_extractor: function used to extract features from X
    validation_segment_length: length of the validation segment. If None,
                               use the entire sample. If not None, it
                               should be larger than or equal to the training segment length.
    training_segment_length: length of the training segment
    subj_mask: mask to select subjects from X

    Returns
    -------
    brain_f1_score: brain F1 score for each subject.
    """
    X = build_features_based_on_aggregation_method(
        feature_extractor,
        X,
        validation_segment_length,
        training_segment_length,
        agg_method,
        subj_mask,
    )

    n_segments = X.shape[1]
    # vertically concatenate test BoWav vectors: (m, n, p) -> (m*n, p)
    X = np.vstack(X)
    y_pred = clf.predict(X)

    # Maybe aggregate output
    if agg_method == "majority_vote":
        # Aggregate all the predictions
        # TODO: include None in validation_segment_length_arr
        # to get all the time series
        y_pred = y_pred.reshape(-1, n_segments)
        y_pred = scipy.stats.mode(y_pred, axis=1)[0]

    # expand labels and expert mask to match test BoWav vectors
    y = labels[subj_mask]
    ext_expert_label_mask = expert_label_mask[subj_mask]

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
