#!/usr/bin/env python3
"""
SHAP Feature Importance Analysis for Random Forest BoWav Classifier

This script analyzes Brain class using two covariance-based approaches:

Plot 1 - Positive SHAP Centroids:
   - Computes covariance between SHAP values (target class Brain) and y_true == Brain
   - Computes mean SHAP across samples for target class Brain
   - Drops features with mean SHAP < 0
   - Sorts remaining centroids by covariance or max SHAP (descending) and selects top k

Plot 2 - Negative SHAP Centroids:
   - Computes covariance between SHAP values (target class Brain) and y_true == Brain
   - Computes mean SHAP across samples for target class Brain
   - Drops features with mean SHAP >= 0
   - Sorts remaining centroids by absolute covariance or min SHAP (most negative, descending) and selects top k

Usage:
    python scripts/feature_importance.py --num-centroids 10
    python scripts/feature_importance.py --num-centroids 10 --sort-by magnitude
    python scripts/feature_importance.py --num-centroids 10 --sample-size 2000
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import typer

from icwaves.evaluation.evaluation import load_estimator
from icwaves.evaluation.config import EvalConfig
from icwaves.evaluation.utils import make_calibrate_idf_fn
from icwaves.file_utils import parse_config_file_args

os.environ["OMP_NUM_THREADS"] = "8"

app = typer.Typer()

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


def analyze_brain_shap(
    shap_values,
    true_labels,
    eval_bowav,
    full_eval_bowav,
    full_eval_labels,
    num_centroids,
    sort_by="covariance",
    positive=True,
):
    """
    Analyze Brain class centroids with positive or negative mean SHAP.

    Steps:
    1. Compute covariance between SHAP values (Brain class) and y_true == Brain
    2. Compute mean SHAP across samples for Brain class
    3. Drop features based on mean SHAP sign (>= 0 for positive, < 0 for negative)
    4. Sort by covariance or max/min SHAP and get top k

    Args:
        positive: If True, analyze positive mean SHAP centroids; if False, negative
        sort_by: "covariance" or "magnitude"
        full_eval_bowav: Full evaluation dataset for computing occurrence rates
        full_eval_labels: Full evaluation labels for computing occurrence rates
    """
    # Brain class is index 0
    brain_shap = shap_values[:, :, 0]

    # Binary indicator: is true label Brain?
    y_brain = (true_labels == 0).astype(float)

    # Compute covariance between each feature's SHAP and y_brain
    brain_shap_centered = brain_shap - np.mean(brain_shap, axis=0)
    y_brain_centered = y_brain - np.mean(y_brain)
    covariances = (brain_shap_centered.T @ y_brain_centered) / (brain_shap.shape[0] - 1)

    # Compute SHAP statistics across samples
    mean_shap = np.mean(brain_shap, axis=0)
    max_shap = np.max(brain_shap, axis=0)
    min_shap = np.min(brain_shap, axis=0)

    # Filter based on mean SHAP sign
    if positive:
        mask = mean_shap >= 0
        magnitude_values = max_shap
    else:
        mask = mean_shap < 0
        magnitude_values = np.abs(min_shap)

    # Apply mask based on sort_by
    if sort_by == "covariance":
        filtered_values = np.abs(covariances)
        filtered_values = filtered_values.copy()
        filtered_values[~mask] = -np.inf
    else:  # magnitude
        filtered_values = magnitude_values.copy()
        filtered_values[~mask] = -np.inf

    # Get top k
    top_indices = np.argsort(filtered_values)[-num_centroids:][::-1]

    # Occurrence rates: Brain vs Not Brain (computed on FULL eval dataset)
    brain_mask = full_eval_labels == 0
    rate_in_brain = np.mean(full_eval_bowav[brain_mask], axis=0)
    rate_not_brain = np.mean(full_eval_bowav[~brain_mask], axis=0)

    # Build results
    results = []
    for idx in top_indices:
        results.append(
            {
                "centroid_index": idx,
                "covariance": covariances[idx],
                "mean_shap": mean_shap[idx],
                "max_shap": max_shap[idx],
                "min_shap": min_shap[idx],
                "rate_in_brain": rate_in_brain[idx],
                "rate_not_brain": rate_not_brain[idx],
                "codebook_class": idx // 128,
            }
        )
    return pd.DataFrame(results)


def plot_centroids(
    df, codebooks, class_name, output_path, num_centroids, psd_method="mtm"
):
    """Plot top centroids with time-domain waveforms and PSDs.

    Args:
        psd_method: Method for PSD computation, either "mtm" (multitaper) or "welch"
    """
    from mne.time_frequency import psd_array_multitaper
    from scipy import signal

    # Grid: k rows (one per centroid), 2 columns (time domain + PSD)
    n_rows = num_centroids
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Sampling rate for cue dataset is 500 Hz, centroid length is 1.0 second
    sampling_rate = 500

    for i, row in df.iterrows():
        idx = int(row["centroid_index"])
        codebook_idx = idx // 128
        centroid_idx = idx % 128
        waveform = codebooks[codebook_idx][centroid_idx]

        # Time domain plot (left column)
        time_axis = np.arange(len(waveform)) / sampling_rate
        axes[i, 0].plot(time_axis, waveform)
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_title(f"Centroid {idx} (Codebook: {CLASS_LABELS[codebook_idx]})")

        # Use scientific notation for small rates
        rate_brain = row["rate_in_brain"]
        rate_not_brain = row["rate_not_brain"]

        if rate_brain < 0.001:
            brain_str = f"{rate_brain:.2e}"
        else:
            brain_str = f"{rate_brain:.3f}"

        if rate_not_brain < 0.001:
            not_brain_str = f"{rate_not_brain:.2e}"
        else:
            not_brain_str = f"{rate_not_brain:.3f}"

        axes[i, 0].text(
            0.02,
            0.98,
            f"Cov: {row['covariance']:.4f}\n"
            f"Min SHAP: {row['min_shap']:.4f}\n"
            f"Max SHAP: {row['max_shap']:.4f}\n"
            f"Mean SHAP: {row['mean_shap']:.4f}\n"
            f"Rate in Brain: {brain_str}\n"
            f"Rate in Not Brain: {not_brain_str}",
            transform=axes[i, 0].transAxes,
            va="top",
            fontsize=8,
        )

        # PSD plot (right column)
        if psd_method == "mtm":
            # Thomson Multitaper Method
            psd, freqs = psd_array_multitaper(
                waveform.reshape(1, -1),
                sfreq=sampling_rate,
                fmin=0,
                fmax=sampling_rate / 2,
                bandwidth=2.0,
                verbose=False,
            )
            psd = psd.squeeze()
        else:  # welch
            # Welch's method
            nfft = 2048
            n = len(waveform)
            freqs, psd = signal.welch(
                waveform,
                fs=sampling_rate,
                nfft=nfft,
                nperseg=int(0.95 * n),
                noverlap=int(0.9 * n),
            )

        # Convert to dB
        psd_db = 10 * np.log10(psd)

        # Find frequency with max PSD (use original PSD for finding max)
        max_freq_idx = np.argmax(psd)
        max_freq = freqs[max_freq_idx]

        axes[i, 1].plot(freqs, psd_db)
        axes[i, 1].set_xlabel("Frequency (Hz)")
        axes[i, 1].set_ylabel("Power (dB/Hz)")
        method_label = "MTM" if psd_method == "mtm" else "Welch"
        axes[i, 1].set_title(f"PSD ({method_label}, Peak: {max_freq:.1f} Hz)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_data(eval_config, background_size=500, sample_size=1000):
    """Load classifier, background data, and evaluation data.

    Returns:
        best_estimator: Trained classifier
        background_data: Sampled training data for SHAP
        eval_bowav: Sampled evaluation data for SHAP
        eval_labels: Sampled evaluation labels for SHAP
        full_eval_bowav: Full evaluation data for occurrence rates
        full_eval_labels: Full evaluation labels for occurrence rates
    """
    # Set random seed for reproducibility
    np.random.seed(42)

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
    eval_file = eval_config.root / "data/cue/bowav/full" / f"{output_base_filename}.npz"
    eval_data = np.load(eval_file)
    eval_bowav = eval_data["bowav"]
    eval_labels = eval_data["labels"]
    n_ics, n_seg, n_feats = eval_bowav.shape
    eval_bowav = eval_bowav.reshape(n_ics * n_seg, n_feats)

    # Reshape labels to match bowav: repeat each label n_seg times
    eval_labels = np.repeat(eval_labels, n_seg)

    # Keep full dataset for occurrence rates
    full_eval_bowav = eval_bowav.copy()
    full_eval_labels = eval_labels.copy()

    # Sample if needed for SHAP computation
    if eval_bowav.shape[0] > sample_size:
        sample_indices = np.random.choice(
            eval_bowav.shape[0], sample_size, replace=False
        )
        eval_bowav = eval_bowav[sample_indices]
        eval_labels = eval_labels[sample_indices]

    return (
        best_estimator,
        background_data,
        eval_bowav,
        eval_labels,
        full_eval_bowav,
        full_eval_labels,
    )


def compute_shap_importance(
    estimator,
    background_data,
    eval_data,
    eval_labels,
    calibrate_idf_fn,
    cache_file=None,
):
    """
    Compute SHAP feature importance for all classes.

    Args:
        cache_file: Path to cache file. If exists, load from cache. If None, don't cache.

    Returns:
        shap_values, predictions, eval_data, eval_labels (cached eval data and labels)
    """
    # Try to load from cache
    if cache_file and cache_file.exists():
        typer.echo(f"Loading cached SHAP values from {cache_file}...")
        cached_data = np.load(cache_file)
        shap_values = cached_data["shap_values"]
        predictions = cached_data["predictions"]
        eval_data_cached = cached_data["eval_data"]
        eval_labels_cached = cached_data["eval_labels"]
        return shap_values, predictions, eval_data_cached, eval_labels_cached

    clf = (
        estimator.named_steps["clf"] if hasattr(estimator, "named_steps") else estimator
    )

    # Calibrate IDF for all eval data (use slice(None) to select all subjects)
    calibrated_estimator = calibrate_idf_fn(estimator, slice(None))

    # Transform background data with original scaler (from training dataset)
    background_data = estimator["scaler"].transform(background_data).toarray()

    # Transform eval data with calibrated scaler (for cross-dataset generalization)
    eval_data_transformed = (
        calibrated_estimator["scaler"].transform(eval_data).toarray()
    )

    # Create SHAP explainer and compute values
    explainer = shap.TreeExplainer(
        clf, background_data, feature_perturbation="interventional"
    )
    # Additivity check disabled because the difference found using a background data of
    # 500 instances and a test data of 7560 instances is small (0.510843 vs 0.486079)
    shap_values = explainer.shap_values(eval_data_transformed, check_additivity=False)

    # Get predictions
    predictions = calibrated_estimator["clf"].predict(eval_data_transformed)

    # Save to cache if requested (include eval_data and eval_labels)
    if cache_file:
        typer.echo(f"Saving SHAP values to cache: {cache_file}...")
        np.savez_compressed(
            cache_file,
            shap_values=shap_values,
            predictions=predictions,
            eval_data=eval_data,
            eval_labels=eval_labels,
        )

    # Return raw SHAP values, predictions, and eval data/labels
    return shap_values, predictions, eval_data, eval_labels


@app.command()
def main(
    output_dir: str = typer.Option(
        "results/feature_importance", help="Output directory"
    ),
    num_centroids: int = typer.Option(10, help="Number of top centroids to display"),
    sort_by: str = typer.Option(
        "covariance", help="Sort by 'covariance' or 'magnitude' (max/min SHAP values)"
    ),
    sample_size: int = typer.Option(
        1000, help="Number of samples to use for SHAP computation"
    ),
    psd_method: str = typer.Option(
        "mtm", help="PSD computation method: 'mtm' (multitaper) or 'welch'"
    ),
):
    """
    SHAP Feature Importance Analysis for Random Forest BoWav Classifier.

    Generates two plots for Brain class:
    - Plot 1: Positive mean SHAP centroids sorted by covariance or max SHAP
    - Plot 2: Negative mean SHAP centroids sorted by absolute covariance or absolute min SHAP (most negative)
    """
    # Validate arguments
    if sort_by not in ["covariance", "magnitude"]:
        typer.echo("Error: --sort-by must be 'covariance' or 'magnitude'")
        raise typer.Exit(code=1)

    if psd_method not in ["mtm", "welch"]:
        typer.echo("Error: --psd-method must be 'mtm' or 'welch'")
        raise typer.Exit(code=1)

    typer.echo(
        f"Starting SHAP Feature Importance Analysis (sort_by={sort_by}, sample_size={sample_size})..."
    )

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

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    (
        estimator,
        background_data,
        eval_data,
        eval_labels,
        full_eval_data,
        full_eval_labels,
    ) = load_data(eval_config, sample_size=sample_size)
    calibrate_idf_fn = make_calibrate_idf_fn(eval_config)

    # Load codebooks
    codebooks = load_codebooks(eval_config)

    # Define cache file path
    cache_file = output_path / "shap_cache.npz"

    # Compute SHAP importance (with caching)
    typer.echo("Computing SHAP values...")
    if cache_file.exists():
        typer.echo(f"Cache found at {cache_file}")
    shap_values, predictions, eval_data, eval_labels = compute_shap_importance(
        estimator, background_data, eval_data, eval_labels, calibrate_idf_fn, cache_file
    )

    # Plot 1: Positive mean SHAP centroids
    typer.echo(
        f"Analyzing Brain class - Positive mean SHAP centroids (sort_by={sort_by}, psd_method={psd_method})..."
    )
    positive_df = analyze_brain_shap(
        shap_values,
        eval_labels,
        eval_data,
        full_eval_data,
        full_eval_labels,
        num_centroids,
        sort_by,
        positive=True,
    )
    positive_df.to_csv(
        output_path
        / f"brain_positive_shap_{sort_by}_{psd_method}_centroid_analysis.csv",
        index=False,
    )
    plot_centroids(
        positive_df,
        codebooks,
        "Brain (Positive SHAP)",
        output_path
        / f"brain_positive_shap_{sort_by}_{psd_method}_{psd_method}_top_centroids.pdf",
        num_centroids,
        psd_method,
    )

    # Plot 2: Negative mean SHAP centroids
    typer.echo(
        f"Analyzing Brain class - Negative mean SHAP centroids (sort_by={sort_by}, psd_method={psd_method})..."
    )
    negative_df = analyze_brain_shap(
        shap_values,
        eval_labels,
        eval_data,
        full_eval_data,
        full_eval_labels,
        num_centroids,
        sort_by,
        positive=False,
    )
    negative_df.to_csv(
        output_path
        / f"brain_negative_shap_{sort_by}_{psd_method}_centroid_analysis.csv",
        index=False,
    )
    plot_centroids(
        negative_df,
        codebooks,
        "Brain (Negative SHAP)",
        output_path / f"brain_negative_shap_{psd_method}_{sort_by}_top_centroids.pdf",
        num_centroids,
        psd_method,
    )

    typer.echo(f"Analysis complete. Results saved to {output_path}")


if __name__ == "__main__":
    app()
