# icwaves/factories.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Callable, Dict, Union

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from .feature_extractors.bowav import build_bowav_from_centroid_assignments
from .feature_extractors.iclabel_features import get_iclabel_features_per_segment
import numpy as np
import numpy.typing as npt


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

    def bowav(
        time_series: dict[str, npt.ArrayLike], segment_len: Dict[str, int | None]
    ):
        return build_bowav_from_centroid_assignments(
            time_series["bowav"], kwargs["n_centroids"], segment_len["bowav"]
        )

    def psd_autocorr(
        time_series: dict[str, npt.ArrayLike], segment_len: Dict[str, int | None]
    ):
        return get_iclabel_features_per_segment(
            signal=time_series["psd_autocorr"],
            sfreq=kwargs["srate"],
            use_autocorr=True,
            segment_len=segment_len["psd_autocorr"],
        )

    # TODO: this concatenation is not performing any pre-scaling of bowav or psd_autocorr.
    # For random forest, that is OK, but we will need to improve this if we want to use logistic regression.
    # TODO:
    #   * test concatenation
    #   * document shape of extractor outputs
    def bowav_psd_autocorr(time_series: dict[str, npt.ArrayLike], segment_len: int):
        bowav_features = bowav(time_series["bowav"], segment_len["bowav"])
        psd_autocorr_features = psd_autocorr(
            time_series["psd_autocorr"], segment_len["psd_autocorr"]
        )
        return np.concatenate([bowav_features, psd_autocorr_features], axis=2)

    feature_extractors = {
        "bowav": bowav,
        "psd_autocorr": psd_autocorr,
        "bowav_psd_autocorr": bowav_psd_autocorr,
    }

    if feature_type not in feature_extractors:
        raise ValueError(f"Unsupported feature extractor: {feature_type}")

    return feature_extractors[feature_type]
