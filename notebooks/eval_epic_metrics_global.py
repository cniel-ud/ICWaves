"""
Compute GLOBAL sensitivity, specificity, and geometric mean (GMean) for EPIC dataset.

This script evaluates classifiers on the EPIC dataset and computes:
- Sensitivity (TPR): TP / (TP + FN) for brain class
- Specificity (TNR): TN / (TN + FP) for brain class
- GMean: sqrt(Sensitivity * Specificity)

Results are computed GLOBALLY across all subjects (not averaged per-subject).
All predictions and labels are pooled together before computing metrics.
"""

import os

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from icwaves.evaluation.config import EvalConfig
from icwaves.evaluation.evaluation import load_estimator
from icwaves.evaluation.utils import (
    build_features_based_on_aggregation_method,
    get_eval_cmmn_filter_options,
    make_calibrate_idf_fn,
    sl2min,
)
from icwaves.data.loading import get_feature_extractor, load_data_bundles
from icwaves.file_utils import parse_config_file_args, get_cmmn_suffix
from scipy.io import loadmat


def get_predictions_for_subject(
    clf,
    features_dict,
    labels,
    expert_label_mask,
    agg_method,
    feature_extractor,
    validation_segment_length,
    training_segment_length,
    subj_mask,
    calibrate_idf=None,
):
    """
    Get predictions and true labels for a single subject.

    Returns:
        tuple: (y_true_expert, y_pred_expert) - arrays of true labels and predictions
               for expert-labeled ICs only
    """
    import scipy

    # Build features
    features_dict = build_features_based_on_aggregation_method(
        feature_extractor,
        features_dict,
        validation_segment_length,
        training_segment_length,
        agg_method,
        subj_mask,
    )

    n_feature_types = len(features_dict.keys())

    # Get predictions from each feature type
    for feature_type, feature_array in features_dict.items():
        n_time_series, n_segments, n_features = feature_array.shape
        feature_array = np.vstack(feature_array)

        # Apply IDF calibration if needed
        if calibrate_idf is not None:
            clf_copy = {k: v for k, v in clf.items()}
            calibrated_clf = calibrate_idf(clf_copy[feature_type], subj_mask)
            clf_to_use = calibrated_clf
        else:
            clf_to_use = clf[feature_type]

        y_pred = clf_to_use.predict(feature_array)

        # Aggregate if using majority vote
        if agg_method[feature_type] == "majority_vote":
            y_pred = y_pred.reshape(-1, n_segments)
            y_pred = scipy.stats.mode(y_pred, axis=1)[0]

    # Get true labels and predictions for expert-labeled ICs only
    y_true = labels[subj_mask]
    expert_mask = expert_label_mask[subj_mask]
    y_true_expert = y_true[expert_mask]
    y_pred_expert = y_pred[expert_mask]

    return y_true_expert, y_pred_expert


def compute_global_metrics(y_true_all, y_pred_all):
    """
    Compute global sensitivity, specificity, and GMean from all predictions.

    For brain class (label 0):
    - Sensitivity = TP / (TP + FN) where TP=predicted brain & actual brain
    - Specificity = TN / (TN + FP) where TN=predicted non-brain & actual non-brain
    - GMean = sqrt(Sensitivity * Specificity)

    Args:
        y_true_all: Array of all true labels across all subjects
        y_pred_all: Array of all predictions across all subjects

    Returns:
        dict with sensitivity, specificity, gmean, and counts
    """
    # Convert to binary: brain (label 0) = 1 (positive), non-brain (labels 1-6) = 0 (negative)
    y_true_binary = (y_true_all == 0).astype(int)
    y_pred_binary = (y_pred_all == 0).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    # For brain as positive class: non-brain=0 (negative), brain=1 (positive)
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    # Extract confusion matrix values
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # This shouldn't happen with labels parameter, but handle it anyway
        tn = fp = fn = tp = 0
        if cm.shape == (1, 1):
            if y_true_binary[0] == 0 and y_pred_binary[0] == 0:
                tn = cm[0, 0]
            elif y_true_binary[0] == 1 and y_pred_binary[0] == 1:
                tp = cm[0, 0]

    # Compute metrics (with brain as positive class)
    # Sensitivity = TP / (TP + FN) = correctly identified brain / all actual brain
    # Specificity = TN / (TN + FP) = correctly rejected non-brain / all actual non-brain
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = np.sqrt(sensitivity * specificity)

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "gmean": gmean,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n_total_ics": len(y_true_all),
        "n_brain_ics": int((y_true_all == 0).sum()),
        "n_nonbrain_ics": int((y_true_all != 0).sum()),
    }


def evaluate_iclabel_epic(root):
    """
    Evaluate ICLabel on EPIC dataset with GLOBAL metrics.

    Returns:
        DataFrame with global ICLabel results
    """
    print("\nEvaluating ICLabel on EPIC dataset...")

    # ICLabel data directory
    iclabel_data_dir = root / "data/epic/ICLabels"

    # Validation time for EPIC (10 minutes = 600 seconds)
    validation_time = 600.0

    # Get subdirectory for this validation time
    subdir = iclabel_data_dir / f"IC_labels_at_{validation_time:.1f}_seconds"

    if not subdir.exists():
        print(f"ICLabel data directory not found: {subdir}")
        return None

    # Find all subject files
    files = list(subdir.glob("*.mat"))
    if not files:
        print(f"No ICLabel .mat files found in: {subdir}")
        return None

    # Collect ALL predictions and labels across all subjects
    all_y_true = []
    all_y_pred = []

    print(f"Processing {len(files)} subjects...")
    for file in tqdm(files):
        # Load ICLabel predictions and expert labels
        with file.open("rb") as f:
            data = loadmat(f)
            expert_label_mask = data["expert_label_mask"].flatten().astype(bool)
            labels = data["labels"] - 1  # Convert from 1-indexed to 0-indexed
            noisy_labels = data["noisy_labels"]

        # Get ICLabel predictions (argmax of the probability matrix)
        iclabel_predictions = np.argmax(noisy_labels, axis=1)

        # Filter to expert-labeled ICs only
        y_true_expert = labels[expert_label_mask]
        y_pred_expert = iclabel_predictions[expert_label_mask]

        all_y_true.append(y_true_expert)
        all_y_pred.append(y_pred_expert)

    # Concatenate all predictions and labels
    if all_y_true:
        all_y_true = np.concatenate(all_y_true)
        all_y_pred = np.concatenate(all_y_pred)

        # Compute GLOBAL metrics
        global_metrics = compute_global_metrics(all_y_true, all_y_pred)

        result_row = {
            "feature_extractor": "iclabel",
            "classifier_type": "iclabel",
            "validation_segment_len": -1,  # Not applicable
            "cmmn_filter": "None",
            **global_metrics,
        }

        # Save ICLabel results
        results_dir = root / "results" / "epic" / "evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)

        iclabel_df = pd.DataFrame([result_row])
        filepath = results_dir / "ICLabel_global_metrics.csv"
        iclabel_df.to_csv(filepath, index=False)

        print(f"Saved ICLabel global metrics to: ICLabel_global_metrics.csv")
        print(f"  Sensitivity: {global_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {global_metrics['specificity']:.4f}")
        print(f"  GMean: {global_metrics['gmean']:.4f}")
        print(f"  Total ICs: {global_metrics['n_total_ics']}")
        print(f"  Brain ICs: {global_metrics['n_brain_ics']}")
        print(f"  Non-brain ICs: {global_metrics['n_nonbrain_ics']}")

        return iclabel_df

    return None


def evaluate_epic_dataset(
    feature_extractor_str,
    classifier_type,
    validation_segment_len,
    root,
):
    """
    Evaluate a single configuration on EPIC dataset with GLOBAL metrics.

    Returns:
        DataFrame with global results per CMMN filter
    """
    # Load configuration
    train_cmmn_filter = None
    train_use_idf = True
    cmmn_str = get_cmmn_suffix(train_cmmn_filter)
    idf_str = "_idf" if train_use_idf and "bowav" in feature_extractor_str else ""
    config_file_name = (
        f"{classifier_type}_{sl2min(validation_segment_len)}{cmmn_str}{idf_str}.txt"
    )

    config_folder = root / "config/_private"
    config_path = config_folder / config_file_name

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return None

    train_config = parse_config_file_args(config_path, feature_extractor_str)

    # Get CMMN filter options for EPIC
    eval_cmmn_filter_options = get_eval_cmmn_filter_options("epic", train_cmmn_filter)

    # Setup results directory
    results_dir = root / "results" / "epic" / "evaluation"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for eval_cmmn_filter in eval_cmmn_filter_options:
        print(
            f"\nEvaluating: {feature_extractor_str}, {classifier_type}, "
            f"val_seg_len={validation_segment_len}, cmmn_filter={eval_cmmn_filter}"
        )

        # Create evaluation configuration
        config = EvalConfig(
            eval_dataset="epic",
            train_config=train_config,
            root=root,
            cmmn_filter=eval_cmmn_filter,
            minutes_per_ic=10,  # EPIC uses 10 minutes
        )

        # Load data
        data_bundles = load_data_bundles(config)
        feature_extractor = get_feature_extractor(feature_extractor_str, data_bundles)
        feature_extractor_dict = {feature_extractor_str: feature_extractor}

        # Load classifier
        clf, best_params = load_estimator(
            config.path_to_classifier[feature_extractor_str]
        )
        clf_dict = {feature_extractor_str: clf}

        agg_method = {
            feature_extractor_str: best_params["input_or_output_aggregation_method"]
        }
        training_segment_length = best_params["training_segment_length"]

        # Validation time for EPIC (10 minutes)
        validation_time = 10 * 60  # seconds

        # Convert validation segment length
        data_bundle = next(iter(data_bundles.values()))
        window_length = (
            config.window_length if "bowav" in feature_extractor_str else None
        )

        from icwaves.feature_extractors.utils import convert_segment_length

        converted_val_segment_len = convert_segment_length(
            [validation_time],
            feature_extractor_str,
            data_bundle.srate,
            window_length,
        )[0]

        # Setup IDF calibration if needed
        if "bowav" in feature_extractor_str:
            calibrate_idf_fn = make_calibrate_idf_fn(config)
        else:
            calibrate_idf_fn = None

        # Extract feature data
        X = {k: v.data for k, v in data_bundles.items()}
        if "bowav" in feature_extractor_str and hasattr(
            data_bundles["bowav"], "zero_window_mask"
        ):
            X["zero_window_mask"] = data_bundles["bowav"].zero_window_mask

        # Collect ALL predictions and labels across all subjects
        all_y_true = []
        all_y_pred = []

        # Evaluate each subject and accumulate predictions
        print(f"Processing {len(config.subj_ids)} subjects...")
        for subj_id in tqdm(config.subj_ids):
            subj_mask = data_bundle.subj_ind == subj_id

            # Skip if no data for this subject
            if not subj_mask.any():
                continue

            y_true_expert, y_pred_expert = get_predictions_for_subject(
                clf_dict,
                X,
                data_bundle.labels,
                data_bundle.expert_label_mask,
                agg_method,
                feature_extractor_dict,
                converted_val_segment_len,
                training_segment_length,
                subj_mask,
                calibrate_idf_fn,
            )

            all_y_true.append(y_true_expert)
            all_y_pred.append(y_pred_expert)

        # Concatenate all predictions and labels
        if all_y_true:
            all_y_true = np.concatenate(all_y_true)
            all_y_pred = np.concatenate(all_y_pred)

            # Compute GLOBAL metrics
            global_metrics = compute_global_metrics(all_y_true, all_y_pred)

            result_row = {
                "feature_extractor": feature_extractor_str,
                "classifier_type": classifier_type,
                "validation_segment_len": validation_segment_len,
                "cmmn_filter": str(eval_cmmn_filter),
                **global_metrics,
            }

            all_results.append(result_row)

            # Save individual configuration result
            config_df = pd.DataFrame([result_row])

            # Generate filename
            val_seg_str = (
                "valSegLenNone"
                if validation_segment_len == -1
                else f"valSegLen{validation_segment_len}"
            )
            cmmn_filter_str = f"cmmn-{eval_cmmn_filter}"

            filename = f"{classifier_type}_{feature_extractor_str}_{val_seg_str}_{cmmn_filter_str}_global_metrics{idf_str}.csv"
            filepath = results_dir / filename

            config_df.to_csv(filepath, index=False)
            print(f"Saved global metrics to: {filename}")
            print(f"  Sensitivity: {global_metrics['sensitivity']:.4f}")
            print(f"  Specificity: {global_metrics['specificity']:.4f}")
            print(f"  GMean: {global_metrics['gmean']:.4f}")
            print(f"  Total ICs: {global_metrics['n_total_ics']}")
            print(f"  Brain ICs: {global_metrics['n_brain_ics']}")
            print(f"  Non-brain ICs: {global_metrics['n_nonbrain_ics']}")

    return pd.DataFrame(all_results) if all_results else None


def main():
    """Main evaluation function."""
    # Set root path
    root = Path(__file__).parents[1]

    # Define configurations to evaluate
    feature_extractors = ["bowav", "psd_autocorr"]
    classifier_types = ["random_forest", "logistic"]
    validation_segment_lens = [-1, 300]

    # Collect all results
    all_results = []

    for feat in feature_extractors:
        for clf_type in classifier_types:
            for val_seg_len in validation_segment_lens:
                df = evaluate_epic_dataset(
                    feat,
                    clf_type,
                    val_seg_len,
                    root,
                )
                if df is not None and not df.empty:
                    all_results.append(df)

    # Evaluate ICLabel
    iclabel_df = evaluate_iclabel_epic(root)
    if iclabel_df is not None and not iclabel_df.empty:
        all_results.append(iclabel_df)

    # Combine all results
    if all_results:
        full_results = pd.concat(all_results, ignore_index=True)

        # Save combined results
        results_dir = root / "results" / "epic" / "evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save global results
        global_file = results_dir / "global_metrics_summary.csv"
        full_results.to_csv(global_file, index=False)
        print(f"\nGlobal metrics summary saved to: {global_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY OF GLOBAL RESULTS")
        print("=" * 80)
        print(full_results.to_string())
        print("\nMetrics computed GLOBALLY across all subjects:")
        print("- Sensitivity (TPR): Proportion of brain ICs correctly identified")
        print("- Specificity (TNR): Proportion of non-brain ICs correctly identified")
        print("- GMean: Geometric mean of sensitivity and specificity")
        print(
            "\nNote: These metrics are computed on the full dataset (all subjects pooled),"
        )
        print("      not averaged per-subject.")

    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
