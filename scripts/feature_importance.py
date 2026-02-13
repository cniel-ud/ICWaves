#!/usr/bin/env python3
"""
SHAP Feature Importance Analysis for Random Forest BoWav Classifier

This script supports four SHAP computation approaches:

1. Mean Absolute SHAP - Filtered (--shap-metric mean-abs-target-class):
   - Filters samples by predicted class before computing mean absolute SHAP
   - Answers: "When the model predicts Brain, which features drive that prediction?"
   
2. Mean Absolute SHAP - Global (--shap-metric mean-abs-all):
   - Computes mean absolute SHAP across ALL samples
   - Answers: "What does the model think are key features for Brain ICs overall?"

3. Correlation SHAP - Filtered (--shap-metric corr-target-class):
   - Filters samples by predicted class, then computes Pearson correlation between
     SHAP values and true labels (one-hot encoded)
   - Answers: "When the model predicts Brain, which features correlate with actual Brain labels?"

4. Correlation SHAP - Global (--shap-metric corr-all):
   - Computes Pearson correlation between SHAP values and true labels across ALL samples
   - Answers: "Which features' SHAP values correlate with actual Brain labels overall?"

Usage:
    python scripts/feature_importance.py --shap-metric mean-abs-target-class --num-centroids 10
    python scripts/feature_importance.py --shap-metric corr-all --non-brain-class Eye
"""

import os
from pathlib import Path
from typing import Literal
import warnings
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


def analyze_class(shap_values, predictions, true_labels, eval_bowav, target_class_idx, 
                  num_centroids, shap_metric):
    """
    Analyze centroid importance for a target class.
    
    Args:
        shap_metric: 'mean-abs-target-class', 'mean-abs-all', 'corr-target-class', or 'corr-all'
    """
    if shap_metric == "mean-abs-target-class":
        # Filtered approach: only samples predicted as target class
        mask = predictions == target_class_idx
        filtered_shap = shap_values[mask, :, target_class_idx]
        mean_shap = np.mean(np.abs(filtered_shap), axis=0)
    elif shap_metric == "mean-abs-all":
        # Global approach: all samples
        class_shap = shap_values[:, :, target_class_idx]
        mean_shap = np.mean(np.abs(class_shap), axis=0)
    elif shap_metric == "corr-target-class":
        # Correlation filtered: only samples predicted as target class
        mask = predictions == target_class_idx
        filtered_shap = shap_values[mask, :, target_class_idx]
        filtered_labels = true_labels[mask]
        y = (filtered_labels == target_class_idx).astype(float)
        # Compute Pearson correlation per feature
        # Stack features and y, compute correlation matrix
        data = np.column_stack([filtered_shap, y])
        corr_matrix = np.corrcoef(data, rowvar=False)
        # Extract correlations between each feature and y (last column/row)
        mean_shap = corr_matrix[:-1, -1]
        # Replace NaN with 0 (happens when feature has zero variance)
        mean_shap = np.nan_to_num(mean_shap, nan=0.0)
    else:  # corr-all
        # Correlation global: all samples
        class_shap = shap_values[:, :, target_class_idx]
        y = (true_labels == target_class_idx).astype(float)
        # Compute Pearson correlation per feature
        # Stack features and y, compute correlation matrix
        data = np.column_stack([class_shap, y])
        corr_matrix = np.corrcoef(data, rowvar=False)
        # Extract correlations between each feature and y (last column/row)
        mean_shap = corr_matrix[:-1, -1]
        # Replace NaN with 0 (happens when feature has zero variance)
        mean_shap = np.nan_to_num(mean_shap, nan=0.0)
    
    # Occurrence rates: ALWAYS Brain (class 0) vs Not Brain (classes 1-6)
    brain_mask = true_labels == 0
    rate_in_brain = np.mean(eval_bowav[brain_mask], axis=0)
    rate_not_brain = np.mean(eval_bowav[~brain_mask], axis=0)
    
    # Top N centroids
    top_indices = np.argsort(mean_shap)[-num_centroids:][::-1]
    
    # Build results
    results = []
    for idx in top_indices:
        results.append({
            'centroid_index': idx,
            'mean_abs_shap': mean_shap[idx],
            'rate_in_brain': rate_in_brain[idx],
            'rate_not_brain': rate_not_brain[idx],
            'codebook_class': idx // 128
        })
    return pd.DataFrame(results)


def plot_centroids(df, codebooks, class_name, output_path, num_centroids):
    """Plot top centroids with time-domain waveforms and PSDs."""
    from mne.time_frequency import psd_array_multitaper
    
    # Grid: k rows (one per centroid), 2 columns (time domain + PSD)
    n_rows = num_centroids
    n_cols = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Sampling rate for cue dataset is 500 Hz, centroid length is 1.0 second
    sampling_rate = 500
    
    for i, row in df.iterrows():
        idx = int(row['centroid_index'])
        codebook_idx = idx // 128
        centroid_idx = idx % 128
        waveform = codebooks[codebook_idx][centroid_idx]
        
        # Time domain plot (left column)
        time_axis = np.arange(len(waveform)) / sampling_rate
        axes[i, 0].plot(time_axis, waveform)
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_title(f"Centroid {idx} (Codebook: {CLASS_LABELS[codebook_idx]})")
        
        # Use scientific notation for small rates
        rate_brain = row['rate_in_brain']
        rate_not_brain = row['rate_not_brain']
        
        if rate_brain < 0.001:
            brain_str = f"{rate_brain:.2e}"
        else:
            brain_str = f"{rate_brain:.3f}"
            
        if rate_not_brain < 0.001:
            not_brain_str = f"{rate_not_brain:.2e}"
        else:
            not_brain_str = f"{rate_not_brain:.3f}"
        
        axes[i, 0].text(0.02, 0.98, 
            f"SHAP: {row['mean_abs_shap']:.4f}\n"
            f"Rate in Brain: {brain_str}\n"
            f"Rate in Not Brain: {not_brain_str}",
            transform=axes[i, 0].transAxes, va='top', fontsize=8)
        
        # PSD plot (right column) - Thomson Multitaper Method
        psd, freqs = psd_array_multitaper(
            waveform.reshape(1, -1), 
            sfreq=sampling_rate, 
            fmin=0, 
            fmax=sampling_rate/2,
            bandwidth=2.0,
            verbose=False
        )
        psd = psd.squeeze()
        
        # Convert to dB
        psd_db = 10 * np.log10(psd)
        
        # Find frequency with max PSD (use original PSD for finding max)
        max_freq_idx = np.argmax(psd)
        max_freq = freqs[max_freq_idx]
        
        axes[i, 1].plot(freqs, psd_db)
        axes[i, 1].set_xlabel('Frequency (Hz)')
        axes[i, 1].set_ylabel('Power (dB/Hz)')
        axes[i, 1].set_title(f"PSD (Peak: {max_freq:.1f} Hz)")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_data(eval_config, background_size=100, sample_size=1000):
    """Load classifier, background data, and evaluation data."""
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

    # Sample if needed
    if eval_bowav.shape[0] > sample_size:
        sample_indices = np.random.choice(
            eval_bowav.shape[0], sample_size, replace=False
        )
        eval_bowav = eval_bowav[sample_indices]
        eval_labels = eval_labels[sample_indices]

    return best_estimator, background_data, eval_bowav, eval_labels


def compute_shap_importance(estimator, background_data, eval_data, calibrate_idf_fn, cache_file=None):
    """
    Compute SHAP feature importance for all classes.
    
    Args:
        cache_file: Path to cache file. If exists, load from cache. If None, don't cache.
    """
    # Try to load from cache
    if cache_file and cache_file.exists():
        typer.echo(f"Loading cached SHAP values from {cache_file}...")
        cached_data = np.load(cache_file)
        shap_values = cached_data['shap_values']
        predictions = cached_data['predictions']
        return shap_values, predictions
    
    clf = (
        estimator.named_steps["clf"] if hasattr(estimator, "named_steps") else estimator
    )

    # Calibrate IDF for all eval data (use slice(None) to select all subjects)
    calibrated_estimator = calibrate_idf_fn(estimator, slice(None))

    # Transform background data with original scaler (from training dataset)
    background_data = estimator["scaler"].transform(background_data).toarray()

    # Transform eval data with calibrated scaler (for cross-dataset generalization)
    eval_data = calibrated_estimator["scaler"].transform(eval_data).toarray()

    # Create SHAP explainer and compute values
    explainer = shap.TreeExplainer(
        clf, background_data, feature_perturbation="interventional"
    )
    shap_values = explainer.shap_values(eval_data, check_additivity=False)
    
    # Get predictions
    predictions = calibrated_estimator["clf"].predict(eval_data)

    # Save to cache if requested
    if cache_file:
        typer.echo(f"Saving SHAP values to cache: {cache_file}...")
        np.savez_compressed(cache_file, shap_values=shap_values, predictions=predictions)

    # Return raw SHAP values and predictions (don't aggregate yet)
    return shap_values, predictions


@app.command()
def main(
    output_dir: str = typer.Option("results/feature_importance", help="Output directory"),
    non_brain_class: str = typer.Option("Muscle", help="Non-brain class to analyze"),
    num_centroids: int = typer.Option(10, help="Number of top centroids to display"),
    shap_metric: Literal["mean-abs-target-class", "mean-abs-all", "corr-target-class", "corr-all"] = typer.Option(
        "mean-abs-target-class", 
        help="SHAP metric: 'mean-abs-target-class', 'mean-abs-all', 'corr-target-class', or 'corr-all'"
    ),
):
    """
    SHAP Feature Importance Analysis for Random Forest BoWav Classifier.
    
    Analyzes Brain class and one non-brain class using SHAP values.
    """
    # Validate non_brain_class
    if non_brain_class not in CLASS_LABELS[1:]:  # Exclude "Brain"
        typer.echo(f"Error: --non-brain-class must be one of: {CLASS_LABELS[1:]}")
        raise typer.Exit(code=1)
    
    typer.echo(f"Starting SHAP Feature Importance Analysis (shap_metric={shap_metric})...")

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
    estimator, background_data, eval_data, eval_labels = load_data(eval_config)
    calibrate_idf_fn = make_calibrate_idf_fn(eval_config)
    
    # Load codebooks
    codebooks = load_codebooks(eval_config)

    # Define cache file path
    cache_file = output_path / "shap_cache.npz"

    # Compute SHAP importance (with caching)
    typer.echo("Computing SHAP values...")
    if cache_file.exists():
        typer.echo(f"Cache found at {cache_file}")
    shap_values, predictions = compute_shap_importance(
        estimator, background_data, eval_data, calibrate_idf_fn, cache_file
    )
    
    # Determine file suffix based on metric
    suffix = f"_{shap_metric}"
    
    # Analyze Brain class
    typer.echo(f"Analyzing Brain class (shap_metric={shap_metric})...")
    brain_df = analyze_class(
        shap_values, predictions, eval_labels, eval_data, 0, 
        num_centroids, shap_metric
    )
    brain_df.to_csv(output_path / f"brain{suffix}_centroid_analysis.csv", index=False)
    plot_centroids(
        brain_df, codebooks, "Brain", 
        output_path / f"brain{suffix}_top_centroids.pdf",
        num_centroids
    )
    
    # Analyze non-brain class
    non_brain_idx = CLASS_LABELS.index(non_brain_class)
    typer.echo(f"Analyzing {non_brain_class} class (shap_metric={shap_metric})...")
    non_brain_df = analyze_class(
        shap_values, predictions, eval_labels, eval_data, non_brain_idx,
        num_centroids, shap_metric
    )
    non_brain_df.to_csv(
        output_path / f"{non_brain_class.lower()}{suffix}_centroid_analysis.csv", 
        index=False
    )
    plot_centroids(
        non_brain_df, codebooks, non_brain_class,
        output_path / f"{non_brain_class.lower()}{suffix}_top_centroids.pdf",
        num_centroids
    )
    
    typer.echo(f"Analysis complete. Results saved to {output_path}")


if __name__ == "__main__":
    app()
