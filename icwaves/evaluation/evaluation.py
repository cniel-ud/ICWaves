import copy
import pickle
from typing import Callable, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from icwaves.data.types import DataBundle
from icwaves.evaluation.config import EvalConfig
from icwaves.evaluation.utils import compute_brain_F1_score_per_subject


def load_classifier(config: EvalConfig) -> Tuple[BaseEstimator, dict]:
    """Load trained classifier and its best parameters."""
    clf_path = config.paths["results_file"]
    with clf_path.open("rb") as f:
        results = pickle.load(f)

    clf = (
        results["best_estimator"]["clf"]
        if "clf" in results["best_estimator"]
        else results["best_estimator"]
    )

    best_params = results["best_params"]

    return clf, best_params


def evaluate_classifier(
    config: EvalConfig,
    clf: BaseEstimator,
    feature_extractor: Callable,
    validation_segment_lengths: np.ndarray,
    data_bundle: DataBundle,
    input_or_output_aggregation_method: str,
    training_segment_length: int,
) -> pd.DataFrame:
    """Evaluate classifier performance across different time windows."""
    columns = [
        "Prediction window [minutes]",
        "Subject ID",
        "Brain F1 score",
        "Number of ICs",
    ]
    results_df = pd.DataFrame(columns=columns)

    if config.feature_extractor == "psd_autocorr":
        conversion_factor = 1 / data_bundle.srate / 60
    elif config.feature_extractor == "bowav":
        conversion_factor = 1 / 60
    else:
        raise ValueError(f"Unknown feature extractor {config.feature_extractor}")

    for val_segment_len in validation_segment_lengths:
        for subj_id in config.subj_ids:
            print(f"Subject {subj_id}")
            subj_mask = data_bundle.subj_ind == subj_id
            score = compute_brain_F1_score_per_subject(
                clf,
                data_bundle.data,
                data_bundle.labels,
                data_bundle.expert_label_mask,
                input_or_output_aggregation_method,
                feature_extractor,
                val_segment_len,
                training_segment_length,
                subj_mask,
            )

            results_df.loc[len(results_df)] = {
                "Prediction window [minutes]": val_segment_len * conversion_factor,
                "Subject ID": subj_id,
                "Brain F1 score": score,
                "Number of ICs": data_bundle.expert_label_mask.sum(),
            }

    return results_df
