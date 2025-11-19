from pathlib import Path
import pickle
from typing import Callable, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from icwaves.data.types import DataBundle
from icwaves.evaluation.config import EvalConfig
from icwaves.evaluation.utils import (
    compute_brain_F1_score_per_subject,
    get_base_results_filename,
)
from icwaves.model_selection.hpo_utils import get_best_parameters
from icwaves.feature_extractors.utils import convert_segment_length


def load_estimator(path: Path) -> Tuple[Union[BaseEstimator, Pipeline], dict]:
    """Load trained classifier and its best parameters."""
    with path.open("rb") as f:
        results = pickle.load(f)

    best_estimator = results["best_estimator"]

    best_params = get_best_parameters(results)

    return best_estimator, best_params


def _should_skip_segment(val_segment_len, train_segment_len, agg_method):
    """
    The keys in val_segment_len and train_segment_len will always be individual feature
    types (e.g., "bowav", "psd_autocorr"). The keys in agg_method can be either individual
    feature types or concatenated (e.g., "bowav_psd_autocorr").
    """
    seg_len_keys = list(val_segment_len.keys())
    agg_method_keys = list(agg_method.keys())
    if len(seg_len_keys) == len(agg_method_keys):
        for k in seg_len_keys:
            if agg_method[k] == "majority_vote":
                if val_segment_len[k] < train_segment_len[k]:
                    return True

    else:  # len(seg_len_keys) > len(agg_method_keys)
        agg_key = agg_method_keys[0]
        for k in seg_len_keys:
            if agg_method[agg_key] == "majority_vote":
                if val_segment_len[k] < train_segment_len[k]:
                    return True

    return False


def get_results_filepath(config: EvalConfig) -> Path:
    """
    Generate the path for saving/loading evaluation results.

    Args:
        config: Evaluation configuration.

    Returns:
        Path to the results CSV file
    """
    results_path = config.root / "results" / config.eval_dataset / "evaluation"
    base_clf_name = get_base_results_filename(config)
    base_clf_name += ".csv"

    results_file = results_path / base_clf_name

    # Create directories if they don't exist
    results_path.mkdir(parents=True, exist_ok=True)

    return results_file


def _load_cached_results(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Attempt to load cached results from file.

    Args:
        filepath: Path to the results file.

    Returns:
        DataFrame with results if file exists, None otherwise.
    """
    if filepath.exists():
        print(f"Loading cached results from {filepath}")
        return pd.read_csv(filepath)
    return None


def _evaluate_subject(
    subj_id: int,
    subj_mask: np.ndarray,
    val_segment_len: float,
    converted_val_segment_len: Dict[str, int],
    clf: Dict[str, BaseEstimator],
    X: Dict[str, np.ndarray],
    data_bundle: DataBundle,
    feature_extractor: Dict[str, Callable],
    input_or_output_aggregation_method: Dict[str, str],
    training_segment_length: Dict[str, int],
    calibrate_idf: Optional[Callable] = None,
) -> Dict:
    """
    Evaluate classifier performance for a single subject.

    Args:
        subj_id: Subject identifier
        subj_mask: Boolean mask for selecting subject's data
        val_segment_len: Validation segment length in seconds
        converted_val_segment_len: Converted validation segment length
        clf: Classifier(s)
        X: Feature data
        data_bundle: Data bundle
        feature_extractor: Feature extractor(s)
        input_or_output_aggregation_method: Aggregation method
        training_segment_length: Training segment length

    Returns:
        Dictionary with evaluation results for this subject
    """
    score = compute_brain_F1_score_per_subject(
        clf,
        X,
        data_bundle.labels,
        data_bundle.expert_label_mask,
        input_or_output_aggregation_method,
        feature_extractor,
        converted_val_segment_len,
        training_segment_length,
        subj_mask,
        calibrate_idf,
    )

    return {
        "Prediction window [minutes]": val_segment_len / 60,
        "Subject ID": subj_id,
        "Brain F1 score": score,
        "Number of ICs": data_bundle.expert_label_mask.sum(),
    }


def _compute_summary_statistics(
    results_df: pd.DataFrame, feature_extractor_name: str, cmmn_filter: str
) -> pd.DataFrame:
    """
    Compute mean and standard deviation of F1 scores.

    Args:
        results_df: DataFrame with per-subject results
        feature_extractor_name: Name of the feature extractor
        cmmn_filter: CMMN filter type

    Returns:
        DataFrame with mean and standard deviation
    """
    # Compute standard deviation
    std_df = results_df.groupby("Prediction window [minutes]")["Brain F1 score"].std()
    std_df = std_df.rename(
        f"StdDev - {feature_extractor_name} - cmmn-{cmmn_filter}"
    ).reset_index()

    # Compute mean
    mean_df = (
        results_df.groupby("Prediction window [minutes]")["Brain F1 score"]
        .mean()
        .rename(f"Brain F1 score - {feature_extractor_name} - cmmn-{cmmn_filter}")
        .reset_index()
    )

    # Merge mean and standard deviation
    return pd.merge(std_df, mean_df, on="Prediction window [minutes]")


def eval_classifier_per_subject_brain_F1(
    config: EvalConfig,
    clf: Dict[str, BaseEstimator],
    feature_extractor: Dict[str, Callable],
    validation_segment_lengths: np.ndarray,
    data_bundles: Dict[str, DataBundle],
    input_or_output_aggregation_method: Dict[str, str],
    training_segment_length: Dict[str, int],
    results_file: Path,
    calibrate_idf: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Evaluate classifier performance across different time windows.

    Args:
        config: Evaluation configuration
        clf: Trained classifier(s)
        feature_extractor: Feature extractor(s)
        validation_segment_lengths: Array of validation segment lengths in seconds
        data_bundles: Data bundles
        input_or_output_aggregation_method: Input or output aggregation method
        training_segment_length: Training segment length
        results_file: Path to file where results are to be saved
        calibrate_idf: If not None, a function used to calibrate the idf term

    Returns:
        DataFrame with mean and standard deviation of F1 scores
    """

    # Try to load cached results
    results_df = _load_cached_results(results_file)

    if results_df is None:
        # Initialize results DataFrame
        columns = [
            "Prediction window [minutes]",
            "Subject ID",
            "Brain F1 score",
            "Number of ICs",
        ]
        results_df = pd.DataFrame(columns=columns)

        # Get a reference data bundle
        data_bundle = next(iter(data_bundles.values()))

        # Convert validation segment lengths
        window_length = (
            config.window_length if "bowav" in config.feature_extractor else None
        )
        # TODO: improve pattern in convert_segment_length:
        #   - handling of window_length
        converted_val_segment_lengths = convert_segment_length(
            validation_segment_lengths.tolist(),
            config.feature_extractor,
            data_bundle.srate,
            window_length,
        )

        # Extract feature data
        X = {k: v.data for k, v in data_bundles.items()}

        # Add zero_window_mask if it exists (for handling all-zero windows in bowav)
        # TODO: That is a hacky way of adding this mask without changing too much the
        # existing code, since the keys in `X` are supposed to be the names of feature
        # types. It would be better to have a more elegant solution.
        if "bowav" in config.feature_extractor and hasattr(
            data_bundles["bowav"], "zero_window_mask"
        ):
            X["zero_window_mask"] = data_bundles["bowav"].zero_window_mask

        # Calculate total iterations for progress bar
        total_iterations = len(validation_segment_lengths) * len(config.subj_ids)

        # Evaluate each segment length and subject
        with tqdm(total=total_iterations) as pbar:
            for converted_val_segment_len, val_segment_len in zip(
                converted_val_segment_lengths, validation_segment_lengths
            ):
                # Skip if segment should be skipped
                if _should_skip_segment(
                    converted_val_segment_len,
                    training_segment_length,
                    input_or_output_aggregation_method,
                ):
                    continue

                # Evaluate each subject
                for subj_id in config.subj_ids:
                    subj_mask = data_bundle.subj_ind == subj_id

                    # Get evaluation results for this subject
                    result = _evaluate_subject(
                        subj_id,
                        subj_mask,
                        val_segment_len,
                        converted_val_segment_len,
                        clf,
                        X,
                        data_bundle,
                        feature_extractor,
                        input_or_output_aggregation_method,
                        training_segment_length,
                        calibrate_idf,
                    )

                    # Add to results DataFrame
                    results_df.loc[len(results_df)] = result
                    pbar.update(1)

        # Save results to file
        results_df.to_csv(results_file, index=False)

    # Compute summary statistics
    return _compute_summary_statistics(
        results_df, config.feature_extractor, str(config.cmmn_filter)
    )
