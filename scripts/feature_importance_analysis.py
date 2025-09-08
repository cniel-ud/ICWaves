#!/usr/bin/env python3
"""
SHAP-based Feature Importance Analysis for Random Forest Classifier

This script performs SHAP (SHapley Additive exPlanations) feature importance
analysis on a random forest classifier trained with BoWav (Bag of Wavelets)
features and cross-validated on 5-minute segments using the cue dataset.

The analysis includes:
1. Loading the trained random forest classifier
2. Computing SHAP values for feature importance analysis
3. Mapping feature indices to their corresponding centroids and codebooks
4. Creating comprehensive visualizations including SHAP-specific plots
5. Saving detailed results and summary reports

SHAP provides more interpretable feature importance compared to basic Random Forest
feature importance by considering feature interactions and providing local explanations.

Usage:
    python scripts/feature_importance_analysis.py [--output_dir results/feature_importance]
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import shap

# Set OMP constants to use only 8 CPUs
os.environ["OMP_NUM_THREADS"] = "8"

from icwaves.evaluation.config import EvalConfig
from icwaves.data.loading import load_data_bundles, get_feature_extractor
from icwaves.data_loaders import load_codebooks_wrapper

CLASS_LABELS = [
    "Brain",
    "Muscle",
    "Eye",
    "Heart",
    "Line Noise",
    "Channel Noise",
    "Other",
]


def setup_config(
    eval_dataset: str = "cue",
    feature_extractor: str = "bowav",
    classifier_type: str = "random_forest",
    validation_segment_len: int = 300,
    cmmn_filter: str = None,
    is_classifier_trained_on_normalized_data: bool = False,
    root: Path = None,
) -> EvalConfig:
    """
    Set up the evaluation configuration for feature importance analysis.

    Args:
        eval_dataset: Dataset name ("cue")
        feature_extractor: Feature extractor type ("bowav")
        classifier_type: Classifier type ("random_forest")
        validation_segment_len: Validation segment length in seconds (300 for 5-min)
        cmmn_filter: CMMN filter type (None, "unnormed-barycenter", "subj_to_subj")
        is_classifier_trained_on_normalized_data: Whether classifier was trained on normalized data
        root: Root path to the project

    Returns:
        EvalConfig: Configuration object
    """
    if root is None:
        root = Path().absolute().parent

    # Set CMMN filter resampling based on dataset
    is_cmmn_filter_resampled = True if eval_dataset == "cue" else False

    config = EvalConfig(
        eval_dataset=eval_dataset,
        feature_extractor=feature_extractor,
        classifier_type=classifier_type,
        validation_segment_length=validation_segment_len,
        root=root,
        cmmn_filter=cmmn_filter,
        is_cmmn_filter_resampled=is_cmmn_filter_resampled,
        is_classifier_trained_on_normalized_data=is_classifier_trained_on_normalized_data,
    )

    return config


def load_classifier_and_data(
    config: EvalConfig,
) -> Tuple[RandomForestClassifier, Dict, Dict]:
    """
    Load the trained classifier and associated data.

    Args:
        config: Evaluation configuration

    Returns:
        Tuple of (classifier, best_params, data_bundles)
    """
    print(
        f"Loading classifier from {config.path_to_classifier[config.feature_extractor]}"
    )

    # Load the full results to check if it's a pipeline
    import pickle

    with open(config.path_to_classifier[config.feature_extractor], "rb") as f:
        results = pickle.load(f)

    # Check if the best_estimator is a pipeline
    best_estimator = results["best_estimator"]
    if hasattr(best_estimator, "named_steps"):
        # It's a pipeline - we need the full pipeline for SHAP
        print("Detected pipeline - using full pipeline for SHAP analysis")
        full_pipeline = best_estimator
        clf = best_estimator.named_steps[
            "clf"
        ]  # Extract just the classifier for type checking
    else:
        # It's just a classifier
        full_pipeline = best_estimator
        clf = best_estimator

    if not isinstance(clf, RandomForestClassifier):
        raise ValueError(f"Expected RandomForestClassifier, got {type(clf)}")

    print(f"Loaded Random Forest with {clf.n_estimators} estimators")

    # Get best parameters
    from icwaves.model_selection.hpo_utils import get_best_parameters

    best_params = get_best_parameters(results)

    # Load data bundles
    print("Loading data bundles...")
    data_bundles = load_data_bundles(config)

    return full_pipeline, best_params, data_bundles


def load_training_background_data(
    config: EvalConfig,
    background_size: int = 100,
) -> Tuple[np.ndarray, Dict]:
    """
    Load training data (emotion_study dataset, subjects 8-35) for SHAP background dataset.

    Args:
        config: Evaluation configuration
        background_size: Maximum number of samples for background dataset

    Returns:
        Tuple of (background_features, training_info)
    """
    print(
        "Loading training data (emotion_study dataset, subjects 8-35) for background dataset..."
    )

    # Training subject IDs from train_classifier.txt (subjects 8-35, excluding 22)
    training_subject_ids = list(range(8, 36))
    training_subject_ids.remove(22)

    # Create training arguments namespace that mimics the training script
    from argparse import Namespace
    from icwaves.file_utils import read_args_from_file
    from icwaves.argparser import create_argparser_all_params

    # Create training args by reading from the config file and setting training subject IDs
    training_args = Namespace()
    training_args.feature_extractor = config.feature_extractor
    training_args.classifier_type = config.classifier_type
    training_args.validation_segment_length = config.validation_segment_length
    training_args.cmmn_filter = config.cmmn_filter
    training_args.is_cmmn_filter_resampled = config.is_cmmn_filter_resampled
    training_args.window_length = config.window_length
    training_args.centroid_length = config.centroid_length
    training_args.num_clusters = config.num_clusters
    training_args.minutes_per_ic = config.minutes_per_ic
    training_args.codebook_minutes_per_ic = config.codebook_minutes_per_ic
    training_args.codebook_ics_per_subject = config.codebook_ics_per_subject

    # Set paths for emotion_study dataset with training subjects
    training_args.path_to_raw_data = str(
        config.root / "data/emotion_study/raw_data_and_IC_labels"
    )
    training_args.path_to_preprocessed_data = str(
        config.root / "data/emotion_study/preprocessed_data"
    )
    if config.is_classifier_trained_on_normalized_data:
        cmmn_subfolder = "normed_filtered"
    else:
        cmmn_subfolder = "unfiltered"
    training_args.path_to_centroid_assignments = str(
        config.root / f"data/emotion_study/centroid_assignments/{cmmn_subfolder}"
    )
    training_args.path_to_codebooks = str(
        config.root / f"results/emotion_study/dictionaries_resampled/{cmmn_subfolder}"
    )
    training_args.path_to_results = str(
        config.root / "results/emotion_study/classifier"
    )

    # Set CMMN filter path if needed
    if config.cmmn_filter is not None:
        training_args.path_to_cmmn_filters = str(
            config.root / f"data/emotion_study/cmmn_filters/original"
        )
    else:
        training_args.path_to_cmmn_filters = None

    # Most importantly: set the training subject IDs
    training_args.subj_ids = training_subject_ids

    print(f"Loading training data with subject IDs: {training_subject_ids}")

    # Load training data bundles using the training arguments
    training_data_bundles = load_data_bundles(training_args)
    training_feature_extractor = get_feature_extractor(
        training_args.feature_extractor, training_data_bundles
    )

    # Get training data bundle
    training_data_bundle = training_data_bundles[config.feature_extractor]

    print(
        f"Loaded training data with {len(training_data_bundle.data)} ICs from subjects {np.unique(training_data_bundle.subj_ind)}"
    )

    # Convert segment length for feature extraction
    from icwaves.feature_extractors.utils import convert_segment_length

    converted_segment_len = convert_segment_length(
        [config.validation_segment_length],
        config.feature_extractor,
        training_data_bundle.srate,
        config.window_length,
    )[0]

    segment_len_dict = {
        config.feature_extractor: converted_segment_len[config.feature_extractor]
    }

    # Extract features from training data
    training_X_dict = {config.feature_extractor: training_data_bundle.data}
    training_X_features = training_feature_extractor(training_X_dict, segment_len_dict)

    # Reshape training features to 2D
    if training_X_features.ndim == 3:
        n_ics_train, n_segments_train, n_features_train = training_X_features.shape
        training_X_features = training_X_features.reshape(-1, n_features_train)

    # Create background dataset from training data
    actual_background_size = min(background_size, len(training_X_features) // 10)
    if len(training_X_features) > actual_background_size:
        print(
            f"Sampling {actual_background_size} instances from {len(training_X_features)} total training instances for background dataset"
        )
        background_indices = np.random.choice(
            len(training_X_features), actual_background_size, replace=False
        )
        X_background = training_X_features[background_indices]
    else:
        print(
            f"Using all {len(training_X_features)} training instances for background dataset"
        )
        X_background = training_X_features

    # Create training info dictionary
    training_info = {
        "subject_ids": training_subject_ids,
        "n_ics": len(training_data_bundle.data),
        "n_training_instances": len(training_X_features),
        "background_size": len(X_background),
        "unique_subjects": np.unique(training_data_bundle.subj_ind).tolist(),
    }

    return X_background, training_info


def get_shap_feature_importance_data(
    model: object,  # Can be RandomForestClassifier or Pipeline
    config: EvalConfig,
    data_bundles: Dict,
    background_data: np.ndarray,
    training_info: Dict,
    sample_size: int = 1000,
) -> Tuple[np.ndarray, Dict, shap.TreeExplainer, np.ndarray]:
    """
    Extract SHAP feature importances and create mapping information.

    Args:
        model: Trained model (RandomForestClassifier or Pipeline containing one)
        config: Evaluation configuration
        data_bundles: Data bundles dictionary
        sample_size: Number of samples to use for SHAP analysis

    Returns:
        Tuple of (shap_values_mean, feature_info, explainer, sample_data)
    """
    # Extract the RandomForest classifier from pipeline if needed
    if hasattr(model, "named_steps"):
        rf_clf = model.named_steps["clf"]
        is_pipeline = True
    else:
        rf_clf = model
        is_pipeline = False
    # Get number of centroids and codebooks for BoWav
    n_centroids = data_bundles[config.feature_extractor].n_centroids

    # Load codebooks to get the number of codebooks (ICLabel classes)
    codebooks = load_codebooks_wrapper(config)
    n_codebooks = len(codebooks)

    # Get feature extractor
    feature_extractor = get_feature_extractor(config.feature_extractor, data_bundles)

    # Extract features from evaluation data (cue dataset) for SHAP analysis
    print("Extracting features from evaluation data (cue dataset) for SHAP analysis...")
    data_bundle = data_bundles[config.feature_extractor]

    # Convert segment length for feature extraction
    from icwaves.feature_extractors.utils import convert_segment_length

    converted_segment_len = convert_segment_length(
        [config.validation_segment_length],
        config.feature_extractor,
        data_bundle.srate,
        config.window_length,
    )[0]

    # Extract features using the same method as evaluation
    X_dict = {config.feature_extractor: data_bundle.data}
    segment_len_dict = {
        config.feature_extractor: converted_segment_len[config.feature_extractor]
    }

    # Extract features for evaluation data
    X_features = feature_extractor(X_dict, segment_len_dict)

    # Reshape to 2D for classifier (n_samples, n_features)
    if X_features.ndim == 3:
        # If we have segments, flatten to (n_ics * n_segments, n_features)
        n_ics, n_segments, n_features = X_features.shape
        X_features = X_features.reshape(-1, n_features)

    # Sample evaluation data for SHAP analysis
    if len(X_features) > sample_size:
        print(
            f"Sampling {sample_size} instances from {len(X_features)} total evaluation instances for SHAP analysis"
        )
        sample_indices = np.random.choice(len(X_features), sample_size, replace=False)
        X_sample = X_features[sample_indices]
    else:
        print(f"Using all {len(X_features)} evaluation instances for SHAP analysis")
        X_sample = X_features

    print(f"Feature vector dimensions: {X_sample.shape[1]} features")
    print(
        f"Background dataset size: {len(background_data)} instances (from emotion_study training data, subjects {training_info['subject_ids']})"
    )
    print(
        f"SHAP analysis dataset size: {len(X_sample)} instances (from cue evaluation data)"
    )
    print(
        f"BoWav configuration: {n_centroids} centroids × {n_codebooks} codebooks = {n_centroids * n_codebooks} features"
    )

    # Create SHAP explainer with interventional perturbation using training data as background
    print("Creating SHAP TreeExplainer with training data background dataset...")

    # Transform background data if we have a pipeline
    if is_pipeline:
        print("Transforming background data with pipeline...")
        background_data = model["scaler"].transform(background_data)
        background_data = background_data.toarray()

    explainer = shap.TreeExplainer(
        rf_clf, background_data, feature_perturbation="interventional"
    )
    print("Computing SHAP values...")
    if is_pipeline:
        # If we have a pipeline, we need to transform the data before computing SHAP values
        print("Transforming evaluation data with pipeline...")
        X_sample = model["scaler"].transform(X_sample)
        X_sample = X_sample.toarray()

    shap_values = explainer.shap_values(X_sample)

    # For multi-class classification, shap_values is an array of shape (n_samples, n_features, n_classes)
    # We are interested in brain vs non-brain classification, so we focus on class 0 (brain)
    shap_values_brain = shap_values[:, :, 0]

    # Calculate mean absolute SHAP values as feature importance for brain class
    shap_importance = np.mean(np.abs(shap_values_brain), axis=0)

    # Create feature mapping information
    feature_info = {
        "n_centroids": n_centroids,
        "n_codebooks": n_codebooks,
        "n_features": len(shap_importance),
        "codebook_names": CLASS_LABELS,
        "sample_size": len(X_sample),
        "total_instances": len(X_features),
    }

    return shap_importance, feature_info, explainer, X_sample


def create_feature_importance_dataframe(
    feature_importances: np.ndarray, feature_info: Dict
) -> pd.DataFrame:
    """
    Create a structured DataFrame with feature importance information.

    Args:
        feature_importances: Array of feature importance values
        feature_info: Dictionary with feature mapping information

    Returns:
        DataFrame with feature importance data
    """
    n_centroids = feature_info["n_centroids"]
    n_codebooks = feature_info["n_codebooks"]
    codebook_names = feature_info["codebook_names"]

    # Create DataFrame with feature importance data
    data = []
    for i, importance in enumerate(feature_importances):
        # Map feature index to codebook and centroid
        codebook_idx = i // n_centroids
        centroid_idx = i % n_centroids

        data.append(
            {
                "feature_idx": i,
                "codebook_idx": codebook_idx,
                "codebook_name": (
                    codebook_names[codebook_idx]
                    if codebook_idx < len(codebook_names)
                    else f"Codebook_{codebook_idx}"
                ),
                "centroid_idx": centroid_idx,
                "importance": importance,
            }
        )

    df = pd.DataFrame(data)

    # Add ranking
    df["importance_rank"] = (
        df["importance"].rank(ascending=False, method="dense").astype(int)
    )

    # Sort by importance (descending)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    return df


def analyze_importance_by_codebook(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze feature importance aggregated by codebook (ICLabel class).

    Args:
        df: DataFrame with individual feature importances

    Returns:
        DataFrame with codebook-level analysis
    """
    codebook_analysis = (
        df.groupby(["codebook_idx", "codebook_name"])
        .agg({"importance": ["sum", "mean", "std", "count"], "importance_rank": "mean"})
        .round(6)
    )

    # Flatten column names
    codebook_analysis.columns = [
        "total_importance",
        "mean_importance",
        "std_importance",
        "n_features",
        "mean_rank",
    ]
    codebook_analysis = codebook_analysis.reset_index()

    # Sort by total importance
    codebook_analysis = codebook_analysis.sort_values(
        "total_importance", ascending=False
    ).reset_index(drop=True)

    return codebook_analysis


def visualize_top_centroids_globally(
    df: pd.DataFrame,
    config: EvalConfig,
    output_dir: Path,
    k: int = 10,
):
    """
    Visualize the top k most important centroids globally across all codebooks.

    Args:
        df: DataFrame with individual feature importances
        config: Evaluation configuration
        output_dir: Directory to save plots
        k: Number of top centroids to visualize
    """
    # Load codebooks to get actual centroid data
    codebooks = load_codebooks_wrapper(config)

    # Get top k features
    top_k_features = df.head(k)

    # Create subplots for top k centroids
    n_cols = min(5, k)
    n_rows = (k + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if k == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (_, row) in enumerate(top_k_features.iterrows()):
        ax_row = i // n_cols
        ax_col = i % n_cols
        ax = axes[ax_row, ax_col] if n_rows > 1 else axes[ax_col]

        codebook_idx = int(row["codebook_idx"])
        centroid_idx = int(row["centroid_idx"])

        # Get the centroid data
        if codebook_idx < len(codebooks):
            centroid = codebooks[codebook_idx][centroid_idx]

            # Plot the centroid waveform
            time_axis = np.linspace(
                0, len(centroid) / 250, len(centroid)
            )  # Assuming 250 Hz sampling rate
            ax.plot(time_axis, centroid, "b-", linewidth=1.5)
            ax.set_title(
                f'Rank {i+1}: {row["codebook_name"]}\nCentroid {centroid_idx} (Imp: {row["importance"]:.4f})',
                fontsize=10,
            )
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
        else:
            ax.text(
                0.5,
                0.5,
                f"Codebook {codebook_idx}\nNot Available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(
                f"Rank {i+1}: Codebook {codebook_idx}, Centroid {centroid_idx}"
            )

    # Hide unused subplots
    for i in range(k, n_rows * n_cols):
        ax_row = i // n_cols
        ax_col = i % n_cols
        ax = axes[ax_row, ax_col] if n_rows > 1 else axes[ax_col]
        ax.set_visible(False)

    plt.suptitle(
        f"Top {k} Most Important Centroids Globally\n"
        f"Random Forest - BoWav - {config.eval_dataset.upper()} Dataset",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    plt.savefig(
        output_dir / f"top_{k}_centroids_global.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / f"top_{k}_centroids_global.pdf", bbox_inches="tight")
    plt.close()


def visualize_top_centroids_per_codebook(
    df: pd.DataFrame,
    config: EvalConfig,
    output_dir: Path,
    k: int = 5,
):
    """
    Visualize the top k most important centroids for each codebook.

    Args:
        df: DataFrame with individual feature importances
        config: Evaluation configuration
        output_dir: Directory to save plots
        k: Number of top centroids per codebook to visualize
    """
    # Load codebooks to get actual centroid data
    codebooks = load_codebooks_wrapper(config)

    # Get unique codebooks
    unique_codebooks = df["codebook_idx"].unique()
    n_codebooks = len(unique_codebooks)

    # Create a large figure with subplots for each codebook
    fig, axes = plt.subplots(n_codebooks, k, figsize=(4 * k, 3 * n_codebooks))
    if n_codebooks == 1:
        axes = axes.reshape(1, -1)
    elif k == 1:
        axes = axes.reshape(-1, 1)

    for cb_idx, codebook_idx in enumerate(sorted(unique_codebooks)):
        # Get top k features for this codebook
        codebook_features = df[df["codebook_idx"] == codebook_idx].head(k)
        codebook_name = (
            codebook_features.iloc[0]["codebook_name"]
            if len(codebook_features) > 0
            else f"Codebook {codebook_idx}"
        )

        for feat_idx, (_, row) in enumerate(codebook_features.iterrows()):
            ax = axes[cb_idx, feat_idx] if n_codebooks > 1 else axes[feat_idx]

            centroid_idx = int(row["centroid_idx"])

            # Get the centroid data
            if codebook_idx < len(codebooks):
                centroid = codebooks[codebook_idx][centroid_idx]

                # Plot the centroid waveform
                time_axis = np.linspace(
                    0, len(centroid) / 250, len(centroid)
                )  # Assuming 250 Hz sampling rate
                ax.plot(time_axis, centroid, "b-", linewidth=1.5)
                ax.set_title(
                    f'C{centroid_idx} (Imp: {row["importance"]:.4f})', fontsize=10
                )
                ax.set_xlabel("Time (s)", fontsize=8)
                ax.set_ylabel("Amplitude", fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"Centroid {centroid_idx}\nNot Available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"C{centroid_idx}")

        # Hide unused subplots for this codebook
        for feat_idx in range(len(codebook_features), k):
            ax = axes[cb_idx, feat_idx] if n_codebooks > 1 else axes[feat_idx]
            ax.set_visible(False)

        # Add codebook label on the left
        if n_codebooks > 1:
            axes[cb_idx, 0].set_ylabel(f"{codebook_name}\n\nAmplitude", fontsize=10)

    plt.suptitle(
        f"Top {k} Most Important Centroids per ICLabel Class\n"
        f"Random Forest - BoWav - {config.eval_dataset.upper()} Dataset",
        fontsize=14,
        y=0.98,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, left=0.08)

    plt.savefig(
        output_dir / f"top_{k}_centroids_per_codebook.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / f"top_{k}_centroids_per_codebook.pdf", bbox_inches="tight")
    plt.close()


def create_visualizations(
    df: pd.DataFrame,
    codebook_analysis: pd.DataFrame,
    output_dir: Path,
    config: EvalConfig,
    background_data: np.ndarray = None,
    training_info: Dict = None,
):
    """
    Create and save visualization plots for feature importance analysis.

    Args:
        df: DataFrame with individual feature importances
        codebook_analysis: DataFrame with codebook-level analysis
        output_dir: Directory to save plots
        config: Evaluation configuration
        background_data: Background data for centroid visualization
        training_info: Training information dictionary
    """
    # Set up plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Top N most important features
    fig, ax = plt.subplots(figsize=(12, 8))
    top_n = min(20, len(df))
    top_features = df.head(top_n)

    bars = ax.barh(
        range(top_n),
        top_features["importance"],
        color=plt.cm.viridis(np.linspace(0, 1, top_n)),
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(
        [
            f"C{row['codebook_idx']}_F{row['centroid_idx']}"
            for _, row in top_features.iterrows()
        ]
    )
    ax.set_xlabel("Feature Importance")
    ax.set_title(
        f"Top {top_n} Most Important Features for Brain Classification\n"
        f"Random Forest - BoWav - {config.eval_dataset.upper()} Dataset - 5min segments"
    )
    ax.invert_yaxis()

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(
            width + width * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            ha="left",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(
        output_dir / "top_features_importance.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "top_features_importance.pdf", bbox_inches="tight")
    plt.close()

    # 2. Feature importance by codebook (ICLabel class)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        codebook_analysis["codebook_name"], codebook_analysis["total_importance"]
    )
    ax.set_xlabel("ICLabel Class (Codebook)")
    ax.set_ylabel("Total Feature Importance")
    ax.set_title(
        f"Feature Importance by ICLabel Class for Brain Classification\n"
        f"Random Forest - BoWav - {config.eval_dataset.upper()} Dataset - 5min segments"
    )
    plt.xticks(rotation=45, ha="right")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_dir / "importance_by_codebook.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "importance_by_codebook.pdf", bbox_inches="tight")
    plt.close()

    # 3. Heatmap of feature importance by codebook and centroid
    n_centroids = df["centroid_idx"].max() + 1
    n_codebooks = df["codebook_idx"].max() + 1

    # Create importance matrix
    importance_matrix = np.zeros((n_codebooks, n_centroids))
    for _, row in df.iterrows():
        importance_matrix[row["codebook_idx"], row["centroid_idx"]] = row["importance"]

    fig, ax = plt.subplots(
        figsize=(max(12, n_centroids // 4), max(8, n_codebooks // 2))
    )
    sns.heatmap(
        importance_matrix,
        xticklabels=[f"C{i}" for i in range(n_centroids)],
        yticklabels=codebook_analysis["codebook_name"],
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Feature Importance"},
    )
    ax.set_xlabel("Centroid Index")
    ax.set_ylabel("ICLabel Class (Codebook)")
    ax.set_title(
        f"Feature Importance Heatmap\n"
        f"Random Forest - BoWav - {config.eval_dataset.upper()} Dataset - 5min segments"
    )

    plt.tight_layout()
    plt.savefig(output_dir / "importance_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "importance_heatmap.pdf", bbox_inches="tight")
    plt.close()

    # 4. Distribution of feature importances
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram
    ax1.hist(df["importance"], bins=50, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Feature Importance")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Feature Importances")
    ax1.axvline(
        df["importance"].mean(),
        color="red",
        linestyle="--",
        label=f'Mean: {df["importance"].mean():.6f}',
    )
    ax1.legend()

    # Box plot by codebook
    df.boxplot(column="importance", by="codebook_name", ax=ax2)
    ax2.set_xlabel("ICLabel Class (Codebook)")
    ax2.set_ylabel("Feature Importance")
    ax2.set_title("Feature Importance Distribution by ICLabel Class")
    plt.xticks(rotation=45, ha="right")

    plt.suptitle(
        f"Feature Importance Distributions\n"
        f"Random Forest - BoWav - {config.eval_dataset.upper()} Dataset - 5min segments"
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "importance_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(output_dir / "importance_distributions.pdf", bbox_inches="tight")
    plt.close()

    # 5. Top k centroids globally
    print("Creating top centroids visualizations...")
    visualize_top_centroids_globally(df, config, output_dir, k=10)

    # 6. Top k centroids per codebook
    visualize_top_centroids_per_codebook(df, config, output_dir, k=5)

    print(f"Visualizations saved to {output_dir}")


def save_results(
    df: pd.DataFrame,
    codebook_analysis: pd.DataFrame,
    feature_info: Dict,
    config: EvalConfig,
    output_dir: Path,
):
    """
    Save analysis results to CSV files and summary text.

    Args:
        df: DataFrame with individual feature importances
        codebook_analysis: DataFrame with codebook-level analysis
        feature_info: Dictionary with feature mapping information
        config: Evaluation configuration
        output_dir: Directory to save results
    """
    # Save detailed feature importance data
    df.to_csv(output_dir / "feature_importance_detailed.csv", index=False)

    # Save codebook-level analysis
    codebook_analysis.to_csv(
        output_dir / "feature_importance_by_codebook.csv", index=False
    )

    # Create and save summary report
    summary_file = output_dir / "feature_importance_summary.txt"
    with open(summary_file, "w") as f:
        f.write("SHAP Feature Importance Analysis Summary - Brain Classification\n")
        f.write("=" * 65 + "\n\n")

        f.write("Analysis Focus:\n")
        f.write(
            "  This analysis focuses on feature importance for brain vs non-brain classification.\n"
        )
        f.write(
            "  SHAP values are computed specifically for the brain class (class 0).\n"
        )
        f.write("  Brain class: ICLabel class 0\n")
        f.write(
            "  Non-brain classes: ICLabel classes 1-6 (Muscle, Eye, Heart, Line Noise, Channel Noise, Other)\n\n"
        )

        f.write("Configuration:\n")
        f.write(f"  Dataset: {config.eval_dataset}\n")
        f.write(f"  Feature Extractor: {config.feature_extractor}\n")
        f.write(f"  Classifier: {config.classifier_type}\n")
        f.write(
            f"  Validation Segment Length: {config.validation_segment_length}s (5 minutes)\n"
        )
        f.write(f"  CMMN Filter: {config.cmmn_filter}\n")
        f.write(
            f"  Normalized Data: {config.is_classifier_trained_on_normalized_data}\n\n"
        )

        f.write("Feature Space:\n")
        f.write(f"  Total Features: {feature_info['n_features']}\n")
        f.write(f"  Number of Centroids: {feature_info['n_centroids']}\n")
        f.write(f"  Number of Codebooks: {feature_info['n_codebooks']}\n\n")

        f.write("Top 10 Most Important Features for Brain Classification:\n")
        for i, (_, row) in enumerate(df.head(10).iterrows()):
            f.write(
                f"  {i+1:2d}. Codebook {row['codebook_idx']} ({row['codebook_name']}), "
                f"Centroid {row['centroid_idx']:3d}: {row['importance']:.6f}\n"
            )

        f.write("\nImportance by ICLabel Class (Codebook) for Brain Classification:\n")
        for _, row in codebook_analysis.iterrows():
            f.write(
                f"  {row['codebook_name']:15s}: Total={row['total_importance']:.6f}, "
                f"Mean={row['mean_importance']:.6f}, Std={row['std_importance']:.6f}\n"
            )

        f.write(f"\nOverall Statistics:\n")
        f.write(f"  Mean Importance: {df['importance'].mean():.6f}\n")
        f.write(f"  Std Importance:  {df['importance'].std():.6f}\n")
        f.write(f"  Min Importance:  {df['importance'].min():.6f}\n")
        f.write(f"  Max Importance:  {df['importance'].max():.6f}\n")

    print(f"Results saved to {output_dir}")
    print(f"Summary report: {summary_file}")


def main():
    """Main function to run feature importance analysis."""
    parser = argparse.ArgumentParser(
        description="Feature Importance Analysis for Random Forest BoWav Classifier"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/feature_importance",
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--cmmn_filter",
        type=str,
        default=None,
        choices=[None, "unnormed-barycenter", "subj_to_subj"],
        help="CMMN filter type to use",
    )

    args = parser.parse_args()

    # Set up paths
    root = (
        Path().absolute().parent
        if Path().absolute().name == "scripts"
        else Path().absolute()
    )
    output_dir = Path(args.output_dir)

    print("Starting Feature Importance Analysis")
    print("=" * 50)

    # Set up configuration
    config = setup_config(
        eval_dataset="cue",
        feature_extractor="bowav",
        classifier_type="random_forest",
        validation_segment_len=300,  # 5 minutes
        cmmn_filter=args.cmmn_filter,
        is_classifier_trained_on_normalized_data=False,
        root=root,
    )

    print(f"Configuration:")
    print(f"  Dataset: {config.eval_dataset}")
    print(f"  Feature Extractor: {config.feature_extractor}")
    print(f"  Classifier: {config.classifier_type}")
    print(f"  Validation Segment Length: {config.validation_segment_length}s")
    print(f"  CMMN Filter: {config.cmmn_filter}")
    print(f"  Normalized Data: {config.is_classifier_trained_on_normalized_data}")
    print()

    # Load classifier and data
    rf_clf, best_params, data_bundles = load_classifier_and_data(config)

    # Load training background data
    background_data, training_info = load_training_background_data(config)

    # Extract SHAP feature importance data
    feature_importances, feature_info, explainer, sample_data = (
        get_shap_feature_importance_data(
            rf_clf, config, data_bundles, background_data, training_info
        )
    )

    # Create structured DataFrame
    df = create_feature_importance_dataframe(feature_importances, feature_info)

    # Analyze by codebook
    codebook_analysis = analyze_importance_by_codebook(df)

    # Create visualizations
    create_visualizations(df, codebook_analysis, output_dir, config)

    # Save results
    save_results(df, codebook_analysis, feature_info, config, output_dir)

    print("\nFeature Importance Analysis Complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
