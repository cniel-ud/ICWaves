"""
Compute sensitivity, specificity, and geometric mean (GMean) for EPIC dataset.

This script evaluates classifiers on the EPIC dataset and computes:
- Sensitivity (TPR): TP / (TP + FN) for brain class
- Specificity (TNR): TN / (TN + FP) for brain class
- GMean: sqrt(Sensitivity * Specificity)

Results are computed per subject and averaged across subjects.
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


def compute_metrics_per_subject(
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
    Compute sensitivity, specificity, and GMean for a single subject.

    For brain class (label 0):
    - Sensitivity = TP / (TP + FN) where TP=predicted brain & actual brain
    - Specificity = TN / (TN + FP) where TN=predicted non-brain & actual non-brain
    - GMean = sqrt(Sensitivity * Specificity)

    Returns:
        dict with sensitivity, specificity, and gmean
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

    # Compute confusion matrix for binary classification (brain vs non-brain)
    # Convert to binary: brain (label 0) = 1 (positive), non-brain (labels 1-6) = 0 (negative)
    y_true_binary = (y_true_expert == 0).astype(int)
    y_pred_binary = (y_pred_expert == 0).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    # For brain as positive class: non-brain=0 (negative), brain=1 (positive)
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    # Handle cases where confusion matrix might not be complete
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
        "n_ics": expert_mask.sum(),
    }


def evaluate_epic_dataset(
    feature_extractor_str,
    classifier_type,
    validation_segment_len,
    root,
):
    """
    Evaluate a single configuration on EPIC dataset and save results per CMMN filter.

    Returns:
        DataFrame with per-subject results
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

        # # TODO: remove after debugging
        # if eval_cmmn_filter != "unnormed-barycenter":
        #     continue

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

        # Collect results for this specific configuration
        config_results = []

        # Evaluate each subject
        print(f"Evaluating {len(config.subj_ids)} subjects...")
        for subj_id in tqdm(config.subj_ids):
            subj_mask = data_bundle.subj_ind == subj_id

            # Skip if no data for this subject
            if not subj_mask.any():
                continue

            metrics = compute_metrics_per_subject(
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

            result_row = {
                "subject_id": subj_id,
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "gmean": metrics["gmean"],
                "n_ics": metrics["n_ics"],
            }

            config_results.append(result_row)
            all_results.append(
                {
                    **result_row,
                    "feature_extractor": feature_extractor_str,
                    "classifier_type": classifier_type,
                    "validation_segment_len": validation_segment_len,
                    "cmmn_filter": str(eval_cmmn_filter),
                }
            )

        # Save results for this specific configuration
        if config_results:
            config_df = pd.DataFrame(config_results)

            # Generate filename following the same pattern as F1 evaluation
            val_seg_str = (
                "valSegLenNone"
                if validation_segment_len == -1
                else f"valSegLen{validation_segment_len}"
            )
            cmmn_filter_str = f"cmmn-{eval_cmmn_filter}"

            filename = f"{classifier_type}_{feature_extractor_str}_{val_seg_str}_{cmmn_filter_str}_metrics{idf_str}.csv"
            filepath = results_dir / filename

            config_df.to_csv(filepath, index=False)
            print(f"Saved {len(config_df)} subject results to: {filename}")

    return pd.DataFrame(all_results)


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

    # Combine all results
    if all_results:
        full_results = pd.concat(all_results, ignore_index=True)

        # Compute average across subjects for each configuration
        summary_results = (
            full_results.groupby(
                [
                    "feature_extractor",
                    "classifier_type",
                    "validation_segment_len",
                    "cmmn_filter",
                ]
            )
            .agg(
                {
                    "sensitivity": ["mean", "std"],
                    "specificity": ["mean", "std"],
                    "gmean": ["mean", "std"],
                    "n_ics": "sum",
                }
            )
            .reset_index()
        )

        # Flatten column names
        summary_results.columns = [
            "_".join(col).strip("_") if col[1] else col[0]
            for col in summary_results.columns.values
        ]

        # Save results
        results_dir = root / "results" / "epic" / "evaluation"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save per-subject results
        per_subject_file = results_dir / "metrics_per_subject.csv"
        full_results.to_csv(per_subject_file, index=False)
        print(f"\nPer-subject results saved to: {per_subject_file}")

        # Save summary results
        summary_file = results_dir / "metrics_summary.csv"
        summary_results.to_csv(summary_file, index=False)
        print(f"Summary results saved to: {summary_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY OF RESULTS")
        print("=" * 80)
        print(summary_results.to_string())
        print("\nMetrics computed:")
        print("- Sensitivity (TPR): Proportion of brain ICs correctly identified")
        print("- Specificity (TNR): Proportion of non-brain ICs correctly identified")
        print("- GMean: Geometric mean of sensitivity and specificity")

    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
