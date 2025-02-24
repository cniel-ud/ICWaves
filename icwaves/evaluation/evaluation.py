import copy
from pathlib import Path
import pickle
from typing import Callable, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from icwaves.data.types import DataBundle
from icwaves.evaluation.config import EvalConfig
from icwaves.evaluation.utils import compute_brain_F1_score_per_subject
from icwaves.model_selection.hpo_utils import get_best_parameters
from icwaves.feature_extractors.utils import _get_conversion_factor
from icwaves.file_utils import get_validation_segment_length_string


def load_classifier(path: Path) -> Tuple[BaseEstimator, dict]:
    """Load trained classifier and its best parameters."""
    with path.open("rb") as f:
        results = pickle.load(f)

    clf = (
        results["best_estimator"]["clf"]
        if isinstance(results["best_estimator"], Pipeline)
        else results["best_estimator"]
    )

    best_params = get_best_parameters(results)

    return clf, best_params


def eval_classifier_per_subject_brain_F1(
    config: EvalConfig,
    clf: BaseEstimator,
    feature_extractor: Callable,
    validation_segment_lengths: np.ndarray,
    data_bundle: DataBundle,
    input_or_output_aggregation_method: str,
    training_segment_length: int,
) -> pd.DataFrame:
    """Evaluate classifier performance across different time windows.

    Args:
        config: Evaluation configuration.
        clf: Trained classifier.
        feature_extractor: Feature extractor.
        validation_segment_lengths: Array of validation segment lengths in seconds.
        data_bundle: Data bundle.
        input_or_output_aggregation_method: Input or output aggregation method.
        training_segment_length: Training segment length in either number of
                                 windows (BoWav) or number of samples (other features).

    Returns:
        A data frame with evaluation results.
    """
    results_path = config.root / "results" / config.eval_dataset / "evaluation"
    valseglen = get_validation_segment_length_string(
        int(config.validation_segment_length)
    )
    results_file = (
        results_path
        / f"eval_brain_f1_{config.classifier_type}_{config.feature_extractor}_{valseglen}.csv"
    )

    # Try to load cached results if they exist
    if results_file.exists():
        print(f"Loading cached results from {results_file}")
        results_df = pd.read_csv(results_file)
    else:
        # Create directories if they don't exist
        results_path.mkdir(parents=True, exist_ok=True)

        columns = [
            "Prediction window [minutes]",
            "Subject ID",
            "Brain F1 score",
            "Number of ICs",
        ]
        results_df = pd.DataFrame(columns=columns)


        converted_val_segment_lengths = convert_segment_length(
            validation_segment_lengths.tolist(),
            config.feature_extractor,
            data_bundle.srate,
            config.window_length,
        )
        total_iterations = len(validation_segment_lengths) * len(config.subj_ids)
        with tqdm(total=total_iterations) as pbar:
            for val_segment_len in validation_segment_lengths:
                converted_val_segment_len = int(val_segment_len * conversion_factor)
                # TODO: move this logic inside compute_brain_F1_score_per_subject?
                if input_or_output_aggregation_method == "majority_vote":
                    if converted_val_segment_len < training_segment_length:
                        continue
                for subj_id in config.subj_ids:
                    subj_mask = data_bundle.subj_ind == subj_id
                    score = compute_brain_F1_score_per_subject(
                        clf,
                        data_bundle.data,
                        data_bundle.labels,
                        data_bundle.expert_label_mask,
                        input_or_output_aggregation_method,
                        feature_extractor,
                        converted_val_segment_len,
                        training_segment_length,
                        subj_mask,
                    )

                    results_df.loc[len(results_df)] = {
                        "Prediction window [minutes]": val_segment_len / 60,
                        "Subject ID": subj_id,
                        "Brain F1 score": score,
                        "Number of ICs": data_bundle.expert_label_mask.sum(),
                    }
                    pbar.update(1)
        results_df.to_csv(results_file, index=False)

    std_df = results_df.groupby("Prediction window [minutes]")["Brain F1 score"].std()
    std_df = std_df.rename(f"StdDev - {config.feature_extractor}").reset_index()
    mean_df = (
        results_df.groupby("Prediction window [minutes]")["Brain F1 score"]
        .mean()
        .rename(f"Brain F1 score - {config.feature_extractor}")
        .reset_index()
    )
    mean_and_std_df = pd.merge(std_df, mean_df, on="Prediction window [minutes]")

    return mean_and_std_df
