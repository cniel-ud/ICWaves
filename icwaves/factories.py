# icwaves/factories.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Callable, Dict, Optional, Union

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from .feature_extractors.bowav import build_bowav_from_centroid_assignments
from .feature_extractors.iclabel_features import get_iclabel_features_per_segment
import numpy as np
import numpy.typing as npt
from sklearn.compose import ColumnTransformer


def _get_classifier(classifier_name: str):
    """Create a classifier based on name."""
    return (
        LogisticRegression()
        if classifier_name == "logistic"
        else RandomForestClassifier()
    )


def _filter_classifier_kwargs(kwargs: dict) -> dict:
    """Remove parameters not meant for the classifier."""
    return {k: v for k, v in kwargs.items() if k not in ["n_codebooks", "n_centroids"]}


def _create_pipeline(scaler, classifier, clf_kwargs: dict) -> Pipeline:
    """Create and configure a pipeline with a scaler and classifier."""
    pipeline = Pipeline(
        [
            ("scaler", scaler),
            ("clf", classifier),
        ]
    )
    params = {f"clf__{k}": v for k, v in clf_kwargs.items()}
    pipeline.set_params(**params)
    return pipeline


def create_estimator(
    classifier_name: str, feature_extractor: str, **kwargs
) -> Union[Pipeline, LogisticRegression, RandomForestClassifier]:
    """Create a classifier instance based on name."""
    if classifier_name not in ["logistic", "random_forest"]:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    classifier = _get_classifier(classifier_name)
    clf_kwargs = _filter_classifier_kwargs(kwargs)

    if feature_extractor == "bowav":
        # Use TfidfTransformer for both logistic regression and random forest with bowav
        clf = _create_pipeline(TfidfTransformer(), classifier, clf_kwargs)
    elif feature_extractor == "psd_autocorr":
        clf = classifier
        clf.set_params(**clf_kwargs)
    elif feature_extractor == "bowav_psd_autocorr":
        # bowav_psd_autocorr is the concatenation of bowav and psd_autocorr
        # we want to apply the TfidfTransformer only to bowav
        # x = [x_{bowav} x_{psd_autocorr}], and len(x_{bowav}) = n_codebooks * n_centroids
        bowav_feat_len = kwargs["n_codebooks"] * kwargs["n_centroids"]
        scaler = ColumnTransformer(
            [
                ("bowav", TfidfTransformer(), slice(0, bowav_feat_len)),
            ],
            remainder="passthrough",
        )

        clf = _create_pipeline(scaler, classifier, clf_kwargs)
    else:
        raise ValueError(f"Unsupported feature extractor: {feature_extractor}")

    return clf


def create_feature_extractor(feature_type: str, **kwargs) -> Callable:
    """Create a feature extractor function based on type.

    For bowav, time_series are the centroid assignments and have shape
    (n_ics, 7, n_win_per_ic), and the output of the extractor has shape
    (n_ics, n_segments, n_centroids*7). 7 is the number of ICLabel classes.

    For psd_autocorr, time_series are ICs, and have shape (n_ics, n_samples),
    and the output of the extractor has shape (n_ics, n_segments, 200). 200
    comes from the fact that we use 100 samples for the PSD and 100 samples
    for the autocorrelation.
    """

    def bowav(
        time_series: dict[str, npt.ArrayLike],
        segment_len: Dict[str, Optional[int]],
    ):
        return build_bowav_from_centroid_assignments(
            time_series["bowav"], kwargs["n_centroids"], segment_len["bowav"]
        )

    def psd_autocorr(
        time_series: dict[str, npt.ArrayLike], segment_len: Dict[str, Optional[int]]
    ):
        return get_iclabel_features_per_segment(
            signal=time_series["psd_autocorr"],
            sfreq=kwargs["srate"],
            use_autocorr=True,
            segment_len=segment_len["psd_autocorr"],
        )

    def bowav_psd_autocorr(time_series: dict[str, npt.ArrayLike], segment_len: int):
        bowav_features = bowav(time_series, segment_len)
        psd_autocorr_features = psd_autocorr(time_series, segment_len)
        return np.concatenate([bowav_features, psd_autocorr_features], axis=2)

    feature_extractors = {
        "bowav": bowav,
        "psd_autocorr": psd_autocorr,
        "bowav_psd_autocorr": bowav_psd_autocorr,
    }

    if feature_type not in feature_extractors:
        raise ValueError(f"Unsupported feature extractor: {feature_type}")

    return feature_extractors[feature_type]
