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
from icwaves.feature_extractors.utils import convert_segment_length
from icwaves.file_utils import get_validation_segment_length_string, get_cmmn_suffix


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


def eval_classifier_per_subject_brain_F1(
    config: EvalConfig,
    clf: dict[str, BaseEstimator],
    feature_extractor: dict[str, Callable],
    validation_segment_lengths: np.ndarray,
    data_bundles: dict[str, DataBundle],
    input_or_output_aggregation_method: dict[str, str],
    training_segment_length: dict[str, int],
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
    cmmn_suffix = get_cmmn_suffix(config.cmmn_filter)
    results_file = (
        results_path
        / f"eval_brain_f1_{config.classifier_type}_{config.feature_extractor}_{valseglen}{cmmn_suffix}.csv"
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

        # the DataBundle for bowav and psd_autocorr only differs in the `data` attribute
        # TODO: find a more efficient way of doing this
        data_bundle = (
            data_bundles["bowav"]
            if "bowav" in data_bundles
            else data_bundles["psd_autocorr"]
        )

        converted_val_segment_lengths = convert_segment_length(
            validation_segment_lengths.tolist(),
            config.feature_extractor,
            data_bundle.srate,
            config.window_length,
        )
        total_iterations = len(validation_segment_lengths) * len(config.subj_ids)
        X = {k: v.data for k, v in data_bundles.items()}
        with tqdm(total=total_iterations) as pbar:
            for converted_val_segment_len, val_segment_len in zip(
                converted_val_segment_lengths, validation_segment_lengths
            ):

                if _should_skip_segment(
                    converted_val_segment_len,
                    training_segment_length,
                    input_or_output_aggregation_method,
                ):
                    continue

                for subj_id in config.subj_ids:
                    subj_mask = data_bundle.subj_ind == subj_id
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
    std_df = std_df.rename(
        f"StdDev - {config.feature_extractor} - cmmn-{config.cmmn_filter}"
    ).reset_index()
    mean_df = (
        results_df.groupby("Prediction window [minutes]")["Brain F1 score"]
        .mean()
        .rename(
            f"Brain F1 score - {config.feature_extractor} - cmmn-{config.cmmn_filter}"
        )
        .reset_index()
    )
    mean_and_std_df = pd.merge(std_df, mean_df, on="Prediction window [minutes]")

    return mean_and_std_df
