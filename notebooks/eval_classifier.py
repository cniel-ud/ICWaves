# %%
# Set OMP constants to use only 8 CPUs
import os

os.environ["OMP_NUM_THREADS"] = "8"

# Imports and setup
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from icwaves.evaluation.evaluation import (
    get_results_filepath,
    load_classifier,
    eval_classifier_per_subject_brain_F1,
)
from icwaves.evaluation.config import EvalConfig
from icwaves.data.loading import get_feature_extractor, load_data_bundles
from icwaves.viz import (
    create_comparison_plot,
)
from icwaves.evaluation.iclabel import compute_iclabel_scores_for_dataset


# %%
def get_cmmn_filter_options(eval_dataset, is_classifier_trained_on_normalized_data):
    if eval_dataset == "emotion_study":
        # TODO: remove "normed-barycenter"
        if is_classifier_trained_on_normalized_data:
            cmmn_filter_options = ["normed-barycenter"]
        else:
            cmmn_filter_options = [None]
    else:
        if is_classifier_trained_on_normalized_data:
            cmmn_filter_options = ["normed-barycenter"]
        else:
            cmmn_filter_options = [
                None,
                "unnormed-barycenter",
                "subj_to_subj",
            ]
    return cmmn_filter_options


def run_evaluation_and_collect_results(
    eval_dataset: str,
    cmmn_filter: Union[str, None],
    feature_extractor_str: str,
    classifier_type: str,
    validation_segment_len: int,
    is_classifier_trained_on_normalized_data: bool,
    is_cmmn_filter_resampled: bool,
    root: Path,
    validation_times: np.ndarray,
) -> pd.DataFrame:
    """
    Run evaluation for a specific configuration and return results in a flat DataFrame format.

    Args:
        eval_dataset: Dataset to evaluate on ("emotion_study" or "cue")
        cmmn_filter: CMMN filter type (None, "barycenter", "subj_to_subj")
        feature_extractor_str: Feature extractor type ("bowav" or "psd_autocorr")
        classifier_type: Classifier type ("logistic" or "random_forest")
        validation_segment_len: Validation segment length (300 or -1)
        is_classifier_trained_on_normalized_data: Whether the classifier was trained on normalized data
        is_cmmn_filter_resampled: Whether the CMMN filter was resampled
        root: Root path
        validation_times: Array of validation times in seconds

    Returns:
        DataFrame: Results in flat format with columns for all configuration dimensions
    """
    # Create evaluation configuration
    config = EvalConfig(
        eval_dataset=eval_dataset,
        feature_extractor=feature_extractor_str,
        classifier_type=classifier_type,
        validation_segment_length=validation_segment_len,
        root=root,
        cmmn_filter=cmmn_filter,
        is_cmmn_filter_resampled=is_cmmn_filter_resampled,
        is_classifier_trained_on_normalized_data=is_classifier_trained_on_normalized_data,
    )

    # Load and prepare data
    print(
        f"Getting data from {eval_dataset}, using CMMN filter {cmmn_filter}, and building feature extractor for {feature_extractor_str}..."
    )
    data_bundles = load_data_bundles(config)
    feature_extractor = get_feature_extractor(feature_extractor_str, data_bundles)
    feature_extractor_dict = {feature_extractor_str: feature_extractor}

    # Load classifier and get parameters
    clf, best_params = load_classifier(config.path_to_classifier[feature_extractor_str])
    clf_dict = {feature_extractor_str: clf}
    agg_method = {
        feature_extractor_str: best_params["input_or_output_aggregation_method"]
    }

    # Handle conversion for PSD autocorr segment length
    if (
        eval_dataset == "cue"
        and "psd_autocorr" in best_params["training_segment_length"]
    ):
        best_params["training_segment_length"]["psd_autocorr"] = int(
            best_params["training_segment_length"]["psd_autocorr"] / 256 * 500
        )

    # Run evaluation
    print("Computing F1 score...")
    # Get results file path
    results_file = get_results_filepath(config)
    results = eval_classifier_per_subject_brain_F1(
        config,
        clf_dict,
        feature_extractor_dict,
        validation_times,
        data_bundles,
        agg_method,
        best_params["training_segment_length"],
        results_file,
    )

    # Convert evaluation results to flat format
    flat_results = []
    for _, row in results.iterrows():
        prediction_window = row["Prediction window [minutes]"]
        mean_f1 = row[f"Brain F1 score - {feature_extractor_str} - cmmn-{cmmn_filter}"]
        std_f1 = row[f"StdDev - {feature_extractor_str} - cmmn-{cmmn_filter}"]

        flat_results.append(
            {
                "eval_dataset": eval_dataset,
                "cmmn_filter": str(cmmn_filter),
                "feature_extractor": feature_extractor_str,
                "classifier_type": classifier_type,
                "is_normalized": is_classifier_trained_on_normalized_data,
                "validation_segment_len": validation_segment_len,
                "prediction_window": prediction_window,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
            }
        )

    return pd.DataFrame(flat_results)


# %%

# Initialize the flat results DataFrame
all_results = pd.DataFrame(
    columns=[
        "eval_dataset",
        "cmmn_filter",
        "feature_extractor",
        "classifier_type",
        "is_normalized",
        "validation_segment_len",
        "prediction_window",
        "mean_f1",
        "std_f1",
    ]
)

# Define validation times
validation_times = np.r_[
    [9.0, 19.5, 30.0, 39.0, 49.5, 60],
    np.arange(2 * 60, 5 * 60 + 1, 60),
    np.arange(5 * 60, 5 * 60 + 1, 60),
    np.arange(10 * 60, 50 * 60 + 1, 5 * 60),
].astype(float)

# Define the root path
root = Path().absolute().parent

# Define the configuration options
is_classifier_trained_on_normalized_data = False
eval_datasets = ["cue", "emotion_study"]
feature_extractors = ["bowav", "psd_autocorr"]
classifier_types = ["random_forest"]  # , "logistic"]  # Both classifier types included
validation_segment_lens = [300, -1]

# Run evaluation for each configuration

# Loop through configurations and run evaluations
for eval_dataset in eval_datasets:
    is_cmmn_filter_resampled = True if eval_dataset == "cue" else False
    cmmn_filter_options = get_cmmn_filter_options(
        eval_dataset, is_classifier_trained_on_normalized_data
    )

    for cmmn_filter in cmmn_filter_options:
        cmmn_filter_str = str(cmmn_filter)

        for feature_extractor_str in feature_extractors:

            for classifier_type in classifier_types:

                for validation_segment_len in validation_segment_lens:
                    print(
                        f"\nRunning evaluation for: dataset={eval_dataset}, filter={cmmn_filter}, "
                        f"feature={feature_extractor_str}, classifier={classifier_type}, "
                        f"validation_segment_len={validation_segment_len}"
                    )

                    # Run evaluation and get results
                    results = run_evaluation_and_collect_results(
                        eval_dataset=eval_dataset,
                        cmmn_filter=cmmn_filter,
                        feature_extractor_str=feature_extractor_str,
                        classifier_type=classifier_type,
                        validation_segment_len=validation_segment_len,
                        is_classifier_trained_on_normalized_data=is_classifier_trained_on_normalized_data,
                        is_cmmn_filter_resampled=is_cmmn_filter_resampled,
                        root=root,
                        validation_times=validation_times,
                    )

                    # Append directly to the master results DataFrame
                    all_results = pd.concat([all_results, results], ignore_index=True)

# Compute and add ICLabel scores for each dataset
print("\nComputing ICLabel scores...")
for eval_dataset in eval_datasets:
    print(f"Computing ICLabel scores for {eval_dataset}...")
    mean_std_f1_iclabel, per_subject_f1_iclabel = compute_iclabel_scores_for_dataset(
        eval_dataset, validation_times, root
    )
    all_results = pd.concat([all_results, mean_std_f1_iclabel], ignore_index=True)
    per_subject_f1_iclabel.to_csv(
        root / "results" / f"{eval_dataset}" / "evaluation" / "ICLabel.csv", index=False
    )

# Save the results to a CSV file
results_dir = root / "results"
results_dir.mkdir(exist_ok=True)
train_data_str = "normalized" if is_classifier_trained_on_normalized_data else "raw"
all_results.to_csv(results_dir / f"mean_std_brain_f1_{train_data_str}.csv", index=False)

# %%
# Example usage of the new plotting function

# Create results directory for saving plots
plot_dir = results_dir / "plots"
plot_dir.mkdir(exist_ok=True)

# Plot 1: Compare feature extractors (bowav vs psd_autocorr) with all filter options
dataset = "cue"
validation_segment_len = -1
classifier_type = "random_forest"
validation_segment_len_str = "5min" if validation_segment_len == 300 else "50min"
save_path = (
    plot_dir
    / f"feature_comparison_{dataset}_{validation_segment_len_str}_{classifier_type}_{train_data_str}.pdf"
)
fig1, ax1 = create_comparison_plot(
    all_results,
    fixed_params={
        "eval_dataset": dataset,  # "emotion_study", "cue"
        "validation_segment_len": validation_segment_len,
        "classifier_type": classifier_type,
    },
    vary_by="feature_extractor",
    save_path=save_path,
    add_title=False,
)

# %%
