# icwaves/factories.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Callable, Dict, Any, Union

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from .feature_extractors.bowav import build_bowav_from_centroid_assignments
from .feature_extractors.iclabel_features import get_iclabel_features_per_segment


def create_estimator(
    classifier_name: str, feature_extractor: str, **kwargs
) -> Union[Pipeline, RandomForestClassifier]:
    """Create a classifier instance based on name."""
    if classifier_name == "logistic":
        if feature_extractor == "bowav":
            clf = Pipeline(
                [
                    ("scaler", TfidfTransformer()),
                    ("clf", LogisticRegression()),
                ]
            )
            clf_params = dict(
                clf__penalty=kwargs["penalty"],
                clf__max_iter=kwargs["max_iter"],
                clf__random_state=kwargs["random_state"],
                clf__class_weight=kwargs["class_weight"],
                clf__solver=kwargs["solver"],
                clf__warm_start=kwargs["warm_start"],
                clf__multi_class=kwargs["multi_class"],
            )
            clf.set_params(**clf_params)
        elif feature_extractor == "psd_autocorr":
            clf = LogisticRegression(**kwargs)
        else:
            raise ValueError(f"Unsupported feature extractor: {feature_extractor}")
    elif classifier_name == "random_forest":
        clf = RandomForestClassifier(**kwargs)
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    return clf


def create_feature_extractor(feature_type: str, **kwargs) -> Callable:
    """Create a feature extractor function based on type."""
    feature_extractors = {
        "bowav": lambda time_series, segment_len: build_bowav_from_centroid_assignments(
            time_series, kwargs["n_centroids"], segment_len
        ),
        "psd_autocorr": lambda time_series, segment_len: get_iclabel_features_per_segment(
            signal=time_series,
            sfreq=kwargs["srate"],
            use_autocorr=True,
            segment_len=segment_len,
        ),
    }

    if feature_type not in feature_extractors:
        raise ValueError(f"Unsupported feature extractor: {feature_type}")

    return feature_extractors[feature_type]
