#!/usr/bin/env python3
"""
Simplified SHAP Feature Importance Analysis for Random Forest BoWav Classifier

Usage:
    python scripts/feature_importance_analysis_simplified.py [--output_dir results/feature_importance]
"""

import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from icwaves.evaluation.evaluation import load_estimator
from icwaves.evaluation.utils import make_calibrate_idf_fn
from icwaves.evaluation.config import EvalConfig
from icwaves.file_utils import parse_config_file_args

os.environ["OMP_NUM_THREADS"] = "8"

CLASS_LABELS = [
    "Brain",
    "Muscle",
    "Eye",
    "Heart",
    "Line Noise",
    "Channel Noise",
    "Other",
]


def load_codebooks(eval_config):
    """Load codebooks from dictionaries directory."""
    codebooks = []
    for class_idx in range(1, 8):  # Classes 1-7
        dict_file = (
            eval_config.path_to_codebooks
            / f"sikmeans_P-1.0_k-128_class-{class_idx}_minutesPerIC-50.0_icsPerSubj-2.npz"
        )
        data = np.load(dict_file)
        codebook = data["centroids"]
        codebooks.append(codebook)

    return np.array(codebooks)


def load_data(eval_config, background_size=100, sample_size=1000):
    """Load classifier, background data, and evaluation data."""
    # Load classifier
    best_estimator, best_params = load_estimator(
        eval_config.path_to_classifier["bowav"]
    )

    # Load background data (training)
    output_base_filename = "random_forest_bowav_valSegLen300_cmmn-None_idf"
    train_file = (
        eval_config.root
        / "data/emotion_study/bowav/train"
        / f"{output_base_filename}.npz"
    )
    train_data = np.load(train_file)
    train_bowav = train_data["bowav"]
    n_ics, n_seg, n_feats = train_bowav.shape
    train_bowav = train_bowav.reshape(n_ics * n_seg, n_feats)
    background_indices = np.random.choice(
        len(train_bowav), min(background_size, len(train_bowav) // 10), replace=False
    )
    background_data = train_bowav[background_indices]

    # Load evaluation data (cue)
    eval_file = (
        eval_config.root / "data/cue/bowav/test_segment" / f"{output_base_filename}.npz"
    )
    eval_data = np.load(eval_file)
    eval_bowav = eval_data["bowav"]
    eval_subj_ind = eval_data["subj_ind"]
    n_ics, n_seg, n_feats = eval_bowav.shape
    eval_bowav = eval_bowav.reshape(n_ics * n_seg, n_feats)
    eval_subj_ind = np.repeat(eval_subj_ind, n_seg)

    # Sample if needed
    if eval_bowav.shape[0] > sample_size:
        sample_indices = np.random.choice(
            eval_bowav.shape[0], sample_size, replace=False
        )
        eval_bowav = eval_bowav[sample_indices]
        eval_subj_ind = eval_subj_ind[sample_indices]

    return best_estimator, background_data, eval_bowav, eval_subj_ind


def compute_shap_importance(
    estimator, background_data, eval_data, eval_subj_ind, calibrate_idf_fn
):
    """Compute SHAP feature importance for all classes."""
    clf = (
        estimator.named_steps["clf"] if hasattr(estimator, "named_steps") else estimator
    )

    unique_subj = np.unique(eval_subj_ind)
    subj_shap_values = []
    for subj_ind in unique_subj:
        mask = eval_subj_ind == subj_ind

        calibrated_estimator = calibrate_idf_fn(estimator, mask)
        # calibrated_background_data = (
        #     calibrated_estimator["scaler"].transform(background_data).toarray()
        # )
        explainer = shap.TreeExplainer(
            clf, background_data, feature_perturbation="interventional"
        )

        subj_eval_data = eval_data[mask]
        subj_eval_data = (
            calibrated_estimator["scaler"].transform(subj_eval_data).toarray()
        )

        subj_shap_values.append(
            explainer.shap_values(subj_eval_data, check_additivity=False)
        )

    # Concatenate across subjects and average across samples
    all_shap_values = np.concatenate(subj_shap_values, axis=0)
    # Return mean absolute importance for each class: shape (n_classes, n_features)
    return np.mean(np.abs(all_shap_values), axis=0).T


def create_feature_dataframe(importances, eval_config):
    """Create structured DataFrame with feature importance for all classes."""
    codebooks = load_codebooks(eval_config)
    n_centroids = codebooks.shape[1]
    n_classes, n_features = importances.shape

    data = []
    for class_idx in range(n_classes):
        for feature_idx, importance in enumerate(importances[class_idx]):
            codebook_idx = feature_idx // n_centroids
            centroid_idx = feature_idx % n_centroids
            data.append(
                {
                    "target_class_idx": class_idx,
                    "target_class_name": CLASS_LABELS[class_idx],
                    "feature_idx": feature_idx,
                    "codebook_idx": codebook_idx,
                    "codebook_name": CLASS_LABELS[codebook_idx],
                    "centroid_idx": centroid_idx,
                    "importance": importance,
                }
            )

    df = pd.DataFrame(data)
    # Rank within each target class
    df["rank"] = (
        df.groupby("target_class_idx")["importance"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )
    return df.sort_values(
        ["target_class_idx", "importance"], ascending=[True, False]
    ).reset_index(drop=True)


def create_visualizations(df, output_dir, eval_config):
    """Create key visualization plots for multi-class feature importance."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations for each target class
    for target_class_idx in df["target_class_idx"].unique():
        class_df = df[df["target_class_idx"] == target_class_idx]
        target_class_name = class_df["target_class_name"].iloc[0]

        # 1. Top features bar plot for this class
        fig, ax = plt.subplots(figsize=(12, 8))
        top_20 = class_df.head(20)
        ax.barh(range(20), top_20["importance"])
        ax.set_yticks(range(20))
        ax.set_yticklabels(
            [
                f"C{row['codebook_idx']}_F{row['centroid_idx']}"
                for _, row in top_20.iterrows()
            ]
        )
        ax.set_xlabel("SHAP Importance")
        ax.set_title(
            f"Top 20 Most Important Features for {target_class_name} Classification"
        )
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(
            output_dir
            / f"top_features_{target_class_name.lower().replace(' ', '_')}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 2. Importance by codebook for this class
        codebook_summary = (
            class_df.groupby(["codebook_idx", "codebook_name"])["importance"]
            .sum()
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(codebook_summary["codebook_name"], codebook_summary["importance"])
        ax.set_xlabel("ICLabel Class")
        ax.set_ylabel("Total SHAP Importance")
        ax.set_title(
            f"Feature Importance by ICLabel Class for {target_class_name} Classification"
        )
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            output_dir
            / f"importance_by_class_{target_class_name.lower().replace(' ', '_')}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 3. Top centroids visualization for this class
        codebooks = load_codebooks(eval_config)
        top_10 = class_df.head(10)

        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()

        for i, (_, row) in enumerate(top_10.iterrows()):
            centroid = codebooks[row["codebook_idx"]][row["centroid_idx"]]
            time_axis = np.linspace(0, len(centroid) / 500, len(centroid))

            axes[i].plot(time_axis, centroid, "b-", linewidth=1.5)
            axes[i].set_title(
                f'Rank {i+1}: {row["codebook_name"]}\nC{row["centroid_idx"]} (Imp: {row["importance"]:.4f})'
            )
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel("Amplitude")
            axes[i].grid(True, alpha=0.3)

        plt.suptitle(
            f"Top 10 Most Important Centroids for {target_class_name} Classification",
            fontsize=16,
        )
        plt.tight_layout()
        plt.savefig(
            output_dir
            / f"top_centroids_{target_class_name.lower().replace(' ', '_')}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # 4. Overall comparison across all classes
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, target_class_idx in enumerate(sorted(df["target_class_idx"].unique())):
        class_df = df[df["target_class_idx"] == target_class_idx]
        target_class_name = class_df["target_class_name"].iloc[0]

        codebook_summary = (
            class_df.groupby(["codebook_idx", "codebook_name"])["importance"]
            .sum()
            .reset_index()
        )

        axes[i].bar(codebook_summary["codebook_name"], codebook_summary["importance"])
        axes[i].set_title(f"{target_class_name}")
        axes[i].set_ylabel("Total SHAP Importance")
        axes[i].tick_params(axis="x", rotation=45)

    plt.suptitle(
        "Feature Importance by ICLabel Class - All Target Classes", fontsize=16
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "importance_comparison_all_classes.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_results(df, output_dir):
    """Save results to CSV and summary text for multi-class analysis."""
    # Save detailed results
    df.to_csv(output_dir / "feature_importance_detailed.csv", index=False)

    # Save summary for each target class
    with open(output_dir / "summary.txt", "w") as f:
        f.write("SHAP Feature Importance Analysis - Multi-Class Classification\n")
        f.write("=" * 65 + "\n\n")

        for target_class_idx in sorted(df["target_class_idx"].unique()):
            class_df = df[df["target_class_idx"] == target_class_idx]
            target_class_name = class_df["target_class_name"].iloc[0]

            f.write(f"TARGET CLASS: {target_class_name}\n")
            f.write("-" * 40 + "\n")

            f.write("Top 10 Features:\n")
            for i, (_, row) in enumerate(class_df.head(10).iterrows()):
                f.write(
                    f"  {i+1:2d}. {row['codebook_name']} C{row['centroid_idx']:3d}: {row['importance']:.6f}\n"
                )

            # Importance by codebook for this target class
            codebook_summary = (
                class_df.groupby("codebook_name")["importance"]
                .agg(["sum", "mean", "count"])
                .round(6)
            )

            f.write("\nImportance by ICLabel Class:\n")
            for class_name, stats in codebook_summary.iterrows():
                f.write(
                    f"  {class_name:15s}: Total={stats['sum']:.6f}, Mean={stats['mean']:.6f}\n"
                )
            f.write("\n" + "=" * 65 + "\n\n")

    # Save separate CSV files for each target class
    for target_class_idx in sorted(df["target_class_idx"].unique()):
        class_df = df[df["target_class_idx"] == target_class_idx]
        target_class_name = class_df["target_class_name"].iloc[0]
        filename = (
            f"feature_importance_{target_class_name.lower().replace(' ', '_')}.csv"
        )
        class_df.to_csv(output_dir / filename, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Simplified SHAP Feature Importance Analysis"
    )
    parser.add_argument(
        "--output_dir", default="results/feature_importance", help="Output directory"
    )
    args = parser.parse_args()

    print("Starting SHAP Feature Importance Analysis...")

    # Setup root path
    root = Path(__file__).absolute().parents[1]

    # Load the train_config from the config file
    config_file = root / "config/_private/random_forest_5min_cmmn-None_idf.txt"
    train_config = parse_config_file_args(config_file, feature_extractor="bowav")

    # Create EvalConfig object
    eval_config = EvalConfig(
        eval_dataset="cue",
        train_config=train_config,
        root=root,
        cmmn_filter=None,
    )

    # Load data
    estimator, background_data, eval_data, eval_subj_ind = load_data(eval_config)
    calibrate_idf_fn = make_calibrate_idf_fn(eval_config)

    # Compute SHAP importance
    print("Computing SHAP values...")
    importances = compute_shap_importance(
        estimator, background_data, eval_data, eval_subj_ind, calibrate_idf_fn
    )

    # Create structured results
    df = create_feature_dataframe(importances, eval_config)

    # Create visualizations and save results
    output_dir = Path(args.output_dir)
    create_visualizations(df, output_dir, eval_config)
    save_results(df, output_dir)

    print(f"Analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
