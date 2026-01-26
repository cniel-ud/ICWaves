from copy import deepcopy
from typing import Callable
import numpy as np
import scipy
from sklearn.metrics import f1_score

from icwaves.evaluation.config import EvalConfig
from icwaves.feature_extractors.tfidf_rate_scaler import TfidfRateScaler
from icwaves.file_utils import build_base_classifier_name


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
    feature_types = list(feature_extractor.keys())
    # X is a dict. Make a deep copy to avoid messing with upstream data.
    X = {k: np.copy(v) for k, v in X.items()}
    for feature_type in feature_types:
        X[feature_type] = X[feature_type][
            subj_mask, ..., slice(0, validation_segment_length[feature_type])
        ]

    if "zero_window_mask" in X and X["zero_window_mask"] is not None:
        X["zero_window_mask"] = X["zero_window_mask"][
            subj_mask, ..., slice(0, validation_segment_length["bowav"])
        ]

    features = {}
    seg_len_keys = list(validation_segment_length.keys())
    if len(feature_types) == len(seg_len_keys):
        """Individual features (e.g., 'bowav', 'psd_autocorr')"""
        for feature_type in feature_types:
            if agg_method[feature_type] == "count_pooling":
                features[feature_type] = feature_extractor[feature_type](
                    X,
                    validation_segment_length,
                )
            else:
                if validation_segment_length[feature_type] is not None:
                    assert (
                        training_segment_length[feature_type]
                        <= validation_segment_length[feature_type]
                    )

                features[feature_type] = feature_extractor[feature_type](
                    X,
                    training_segment_length,
                )
    else:
        """Concatenated features (e.g., 'bowav_psd_autocorr')"""
        extractor_key = feature_types[0]
        if agg_method[extractor_key] == "count_pooling":
            features[extractor_key] = feature_extractor[extractor_key](
                X,
                validation_segment_length,
            )
        else:
            for feature_type in seg_len_keys:
                if validation_segment_length[feature_type] is not None:
                    assert (
                        training_segment_length[feature_type]
                        <= validation_segment_length[feature_type]
                    )

            features[extractor_key] = feature_extractor[extractor_key](
                X,
                training_segment_length,
            )

    return features


def compute_brain_F1_score_per_subject(
    clf,
    features_dict,
    labels,
    expert_label_mask,
    agg_method,
    feature_extractor,
    validation_segment_length,
    training_segment_length,
    subj_mask=slice(None),
    calibrate_idf=None,
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
    features_dict = build_features_based_on_aggregation_method(
        feature_extractor,
        features_dict,
        validation_segment_length,
        training_segment_length,
        agg_method,
        subj_mask,
    )

    n_feature_types = len(features_dict.keys())
    y_preds_log_proba_agg = 0

    for feature_type, feature_array in features_dict.items():
        n_time_series, n_segments, n_features = feature_array.shape
        # vertically concatenate test BoWav vectors: (m, n, p) -> (m*n, p)
        feature_array = np.vstack(feature_array)
        if calibrate_idf is not None:
            clf = {k: v for k, v in clf.items()}
            calibrated_clf = calibrate_idf(clf[feature_type], subj_mask)
            clf[feature_type] = calibrated_clf
        y_pred = clf[feature_type].predict(feature_array)
        y_preds_proba = clf[feature_type].predict_proba(feature_array)
        y_preds_log_proba = np.log(y_preds_proba + 1e-12)

        # Maybe aggregate output
        if agg_method[feature_type] == "majority_vote":
            # Aggregate all the predictions
            # TODO: include None in validation_segment_length_arr
            # to get all the time series
            y_pred = y_pred.reshape(-1, n_segments)
            y_pred = scipy.stats.mode(y_pred, axis=1)[0]
            n_classes = y_preds_proba.shape[1]
            y_preds_log_proba = y_preds_log_proba.reshape(
                n_time_series, n_segments, n_classes
            )
            y_preds_log_proba = np.sum(y_preds_log_proba, axis=1)
            y_preds_log_proba = y_preds_log_proba - scipy.special.logsumexp(
                y_preds_log_proba, axis=1, keepdims=True
            )

        y_preds_log_proba_agg += y_preds_log_proba

    y_preds_log_proba_agg_norm = y_preds_log_proba_agg - scipy.special.logsumexp(
        y_preds_log_proba_agg, axis=1, keepdims=True
    )
    y_preds_proba_agg = np.exp(y_preds_log_proba_agg_norm)
    y_preds_agg = np.argmax(y_preds_proba_agg, axis=1)

    # if we passed two classifiers (each trained on a different feature type)
    # use the aggregated predictions to compute the F1 score
    if n_feature_types == 2:
        feature_types = list(features_dict.keys())
        print(
            f"Using aggregated predictions from {feature_types[0]} and {feature_types[1]}..."
        )
        y_pred = y_preds_agg

    # expand labels and expert mask to match test BoWav vectors
    y = labels[subj_mask]
    ext_expert_label_mask = expert_label_mask[subj_mask]

    y_expert = y[ext_expert_label_mask]
    y_pred_expert = y_pred[ext_expert_label_mask]
    brain_f1_score = f1_score(y_expert, y_pred_expert, labels=[0], average=None)

    return brain_f1_score.item()


def sl2min(sl):
    return "50min" if sl == -1 else "5min"


def get_base_results_filename(config: EvalConfig) -> str:
    base_clf_name = build_base_classifier_name(config)
    if config.train_config.cmmn_filter is not None:
        base_clf_name += "_clf-trained-on-filtered-data"
    return base_clf_name


def make_calibrate_idf_fn(
    config: EvalConfig,
) -> Callable:
    def calibrate_idf(pipeline, subj_mask):
        assert hasattr(pipeline, "named_steps")
        assert "scaler" in pipeline.named_steps
        assert "clf" in pipeline.named_steps
        assert isinstance(pipeline["scaler"], TfidfRateScaler)

        # For the source dataset (emotion_study), the cmmn_filter used to
        # create these bowav features we are loading here is the same
        # one used during training
        config_train = deepcopy(config)
        config_train.cmmn_filter = config_train.train_config.cmmn_filter

        train_dir = config_train.root / "data/emotion_study/bowav/train"
        output_base_filename = get_base_results_filename(config_train)
        train_path = train_dir / f"{output_base_filename}.npz"

        with np.load(train_path, allow_pickle=True) as f:
            bowav_train = f["bowav"]

        bowav_train = bowav_train.reshape(-1, bowav_train.shape[-1])

        # TODO: compute this on the go?
        test_dir = config.root / f"data/{config.eval_dataset}/bowav/full"
        test_path = test_dir / f"{output_base_filename}.npz"
        with np.load(test_path, allow_pickle=True) as f:
            bowav_full_test = f["bowav"]

        train_mean_vals = np.mean(bowav_train, axis=0)
        test_mean_vals = np.mean(bowav_full_test[subj_mask], axis=(0, 1))
        tf_ratio = np.divide(
            train_mean_vals,
            test_mean_vals,
            out=np.ones_like(train_mean_vals),  # Default to 1 for zero division
            where=test_mean_vals != 0,
        )
        clipped_tf_ratio = np.clip(tf_ratio, 0.1, 10.0)
        calibrated_idf = pipeline["scaler"].idf_ * clipped_tf_ratio
        calibrated_pipeline = deepcopy(pipeline)
        calibrated_pipeline["scaler"].idf_ = calibrated_idf
        calibrated_pipeline["scaler"]._tfidf_transformer.idf_ = calibrated_idf

        return calibrated_pipeline

    return calibrate_idf


def get_eval_cmmn_filter_options(eval_dataset, train_cmmn_filter):
    assert train_cmmn_filter in [None, "normed-barycenter"]
    if eval_dataset == "emotion_study":
        cmmn_filter_options = [train_cmmn_filter]
    else:
        if train_cmmn_filter is None:
            cmmn_filter_options = [
                None,
                "unnormed-barycenter",
                "normed-barycenter",
                "subj_to_subj",
            ]
        else:
            cmmn_filter_options = [train_cmmn_filter]

    return cmmn_filter_options
