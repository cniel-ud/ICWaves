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

import pickle

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


def load_codebooks(root):
    """Load codebooks from dictionaries directory."""
    codebooks = []
    dict_dir = root / "results/emotion_study/dictionaries_resampled/unfiltered"

    for class_idx in range(1, 8):  # Classes 1-7
        dict_file = (
            dict_dir
            / f"sikmeans_P-1.0_k-128_class-{class_idx}_minutesPerIC-50.0_icsPerSubj-2.npz"
        )
        data = np.load(dict_file)
        codebook = data["centroids"]  # Assuming centroids key
        codebooks.append(codebook)

    return np.array(codebooks)


def load_estimator(path):
    """Load trained classifier from pickle file."""
    with open(path, "rb") as f:
        results = pickle.load(f)
    return results["best_estimator"]


def setup_config(root=None):
    """Setup paths and configuration."""
    if root is None:
        root = (
            Path().absolute().parent
            if Path().absolute().name == "scripts"
            else Path().absolute()
        )

    classifier_path = (
        root
        / "results/emotion_study/classifier/train_random_forest_bowav_valSegLen300_cmmn-None.pkl"
    )

    return {
        "root": root,
        "classifier_path": classifier_path,
    }


def load_data(config, background_size=100, sample_size=1000):
    """Load classifier, background data, and evaluation data."""
    # Load classifier
    estimator = load_estimator(config["classifier_path"])

    # Load background data (training)
    train_file = config["root"] / "data/emotion_study/bowav/train_5min.npz"
    train_data = np.load(train_file)["bowav"]
    background_indices = np.random.choice(
        len(train_data), min(background_size, len(train_data) // 10), replace=False
    )
    background_data = train_data[background_indices]

    # Load evaluation data (cue)
    eval_file = config["root"] / "data/cue/bowav/5min.npz"
    eval_data = np.load(eval_file)["bowav"]
    if len(eval_data) > sample_size:
        sample_indices = np.random.choice(len(eval_data), sample_size, replace=False)
        eval_data = eval_data[sample_indices]

    return estimator, background_data, eval_data


def compute_shap_importance(estimator, background_data, eval_data):
    """Compute SHAP feature importance for brain classification."""
    # Extract classifier and handle pipeline
    if hasattr(estimator, "named_steps"):
        clf = estimator.named_steps["clf"]
        background_data = estimator["scaler"].transform(background_data).toarray()
        eval_data = estimator["scaler"].transform(eval_data).toarray()
    else:
        clf = estimator

    # Create SHAP explainer and compute values
    explainer = shap.TreeExplainer(
        clf, background_data, feature_perturbation="interventional"
    )
    shap_values = explainer.shap_values(eval_data)

    # Focus on brain class (class 0) and compute mean absolute importance
    return np.mean(np.abs(shap_values[:, :, 0]), axis=0)


def create_feature_dataframe(importances, root):
    """Create structured DataFrame with feature importance."""
    codebooks = load_codebooks(root)
    n_centroids = codebooks.shape[1]

    data = []
    for i, importance in enumerate(importances):
        codebook_idx = i // n_centroids
        centroid_idx = i % n_centroids
        data.append(
            {
                "feature_idx": i,
                "codebook_idx": codebook_idx,
                "codebook_name": CLASS_LABELS[codebook_idx],
                "centroid_idx": centroid_idx,
                "importance": importance,
            }
        )

    df = pd.DataFrame(data)
    df["rank"] = df["importance"].rank(ascending=False, method="dense").astype(int)
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def create_visualizations(df, output_dir, root):
    """Create key visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Top features bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    top_20 = df.head(20)
    bars = ax.barh(range(20), top_20["importance"])
    ax.set_yticks(range(20))
    ax.set_yticklabels(
        [
            f"C{row['codebook_idx']}_F{row['centroid_idx']}"
            for _, row in top_20.iterrows()
        ]
    )
    ax.set_xlabel("SHAP Importance")
    ax.set_title("Top 20 Most Important Features for Brain Classification")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / "top_features.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Importance by codebook
    codebook_summary = (
        df.groupby(["codebook_idx", "codebook_name"])["importance"].sum().reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(codebook_summary["codebook_name"], codebook_summary["importance"])
    ax.set_xlabel("ICLabel Class")
    ax.set_ylabel("Total SHAP Importance")
    ax.set_title("Feature Importance by ICLabel Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "importance_by_class.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Top centroids visualization
    codebooks = load_codebooks(root)
    top_10 = df.head(10)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i, (_, row) in enumerate(top_10.iterrows()):
        centroid = codebooks[row["codebook_idx"]][row["centroid_idx"]]
        time_axis = np.linspace(0, len(centroid) / 250, len(centroid))

        axes[i].plot(time_axis, centroid, "b-", linewidth=1.5)
        axes[i].set_title(
            f'Rank {i+1}: {row["codebook_name"]}\nC{row["centroid_idx"]} (Imp: {row["importance"]:.4f})'
        )
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude")
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Top 10 Most Important Centroids", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "top_centroids.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_results(df, output_dir):
    """Save results to CSV and summary text."""
    # Save detailed results
    df.to_csv(output_dir / "feature_importance_detailed.csv", index=False)

    # Save summary
    codebook_summary = (
        df.groupby("codebook_name")["importance"].agg(["sum", "mean", "count"]).round(6)
    )

    with open(output_dir / "summary.txt", "w") as f:
        f.write("SHAP Feature Importance Analysis - Brain Classification\n")
        f.write("=" * 55 + "\n\n")

        f.write("Top 10 Features:\n")
        for i, (_, row) in enumerate(df.head(10).iterrows()):
            f.write(
                f"  {i+1:2d}. {row['codebook_name']} C{row['centroid_idx']:3d}: {row['importance']:.6f}\n"
            )

        f.write("\nImportance by Class:\n")
        for class_name, stats in codebook_summary.iterrows():
            f.write(
                f"  {class_name:15s}: Total={stats['sum']:.6f}, Mean={stats['mean']:.6f}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Simplified SHAP Feature Importance Analysis"
    )
    parser.add_argument(
        "--output_dir", default="results/feature_importance", help="Output directory"
    )
    args = parser.parse_args()

    print("Starting SHAP Feature Importance Analysis...")

    # Setup and load data
    config = setup_config()
    estimator, background_data, eval_data = load_data(config)

    # Compute SHAP importance
    print("Computing SHAP values...")
    importances = compute_shap_importance(estimator, background_data, eval_data)

    # Create structured results
    df = create_feature_dataframe(importances, config["root"])

    # Create visualizations and save results
    output_dir = Path(args.output_dir)
    create_visualizations(df, output_dir, config["root"])
    save_results(df, output_dir)

    print(f"Analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
