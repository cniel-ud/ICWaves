#!/usr/bin/env python3
"""
SHAP Feature Importance Analysis for BoWav classifiers.

Generalized over classifier (logistic / random_forest) and evaluation dataset
(emotion_study test subjects / cue / epic). For each (classifier, dataset)
combination the Brain class is analyzed with two covariance-based approaches:

Plot 1 - Positive SHAP Centroids:
   - Computes covariance between SHAP values (target class Brain) and y_true == Brain
   - Computes mean SHAP across samples for target class Brain
   - Drops features with mean SHAP < 0
   - Sorts remaining centroids by covariance or max SHAP (descending) and selects top k

Plot 2 - Negative SHAP Centroids:
   - Computes covariance between SHAP values (target class Brain) and y_true == Brain
   - Computes mean SHAP across samples for target class Brain
   - Drops features with mean SHAP >= 0
   - Sorts remaining centroids by covariance or min SHAP (most negative, descending) and selects top k

Classifier-specific SHAP explainer:
   - random_forest -> shap.TreeExplainer (interventional)
   - logistic       -> shap.LinearExplainer
Both return SHAP values shaped (n_samples, n_features, n_classes), so the
downstream analysis/plotting is identical.

Dataset-specific details are resolved through EvalConfig:
   - Sampling rate: cue = 500 Hz (resampled codebooks); emotion_study / epic =
     256 Hz (native codebooks). Used for both the centroid time axis and PSDs.
   - IDF reweighting: applied only when evaluating a target dataset different
     from the source (emotion_study is the source, so it uses the source IDF
     unchanged). Mirrors notebooks/eval_classifier.py.

Usage:
    python scripts/feature_importance.py --classifier random_forest --dataset cue
    python scripts/feature_importance.py --classifier logistic --dataset epic --num-centroids 10
    python scripts/feature_importance.py --classifier all --dataset all --sort-by magnitude
    python scripts/feature_importance.py --classifier all --dataset all --validation-segment 50
"""

import os
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import typer

from icwaves.evaluation.evaluation import load_estimator
from icwaves.evaluation.config import EvalConfig
from icwaves.evaluation.utils import make_calibrate_idf_fn, get_base_results_filename
from icwaves.file_utils import (
    parse_config_file_args,
    _build_centroid_assignments_file,
    _build_preprocessed_data_file,
)
from icwaves.sikmeans.shift_kmeans import _asignment_step

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

SUPPORTED_CLASSIFIERS = ["logistic", "random_forest"]
# Validation segment length (minutes) the classifier was cross-validated with:
# 5 (valSegLen300, count pooling) or 50 (valSegLenNone, majority vote).
SUPPORTED_VALIDATION_SEGMENTS = [5, 50]


def config_stem(classifier, validation_segment):
    """Config-file stem (config/_private/<stem>.txt) of a trained classifier."""
    return f"{classifier}_{validation_segment}min_cmmn-None_idf"


def combo_dirname(classifier, dataset, validation_segment):
    """Output folder name; the 5-min classifiers keep the original folder names."""
    suffix = "" if validation_segment == 5 else f"_{validation_segment}min"
    return f"{classifier}_{dataset}{suffix}"

SUPPORTED_DATASETS = ["emotion_study", "cue", "epic"]

# The IDF reweighting adapts the source-trained scaler to a *target* dataset.
# emotion_study is the source (subjects 8-35 train), so its test subjects use
# the source IDF unchanged -- matching notebooks/eval_classifier.py.
SOURCE_DATASET = "emotion_study"


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
        filtered_values = covariances
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


class ExampleSource:
    """Lazy access to the windows of the evaluation ICs and their centroid
    assignments, to draw example windows matched to a centroid.

    Nothing is recomputed: the centroid assignments (n_ics, n_codebooks,
    n_windows) are the ones used to build the BoWav features, and the windowed ICs
    (n_ics, n_windows, window_length) are the preprocessed data of the evaluation
    dataset. Both are memory-mapped, so only the sampled windows are read.
    """

    def __init__(self, eval_config, seed=0):
        self.seed = seed
        path = eval_config.path_to_centroid_assignments
        self.assignments = np.load(
            path / _build_centroid_assignments_file(eval_config), mmap_mode="r"
        )
        npz_path = eval_config.path_to_preprocessed_data / _build_preprocessed_data_file(
            eval_config
        )
        with np.load(npz_path) as data:  # members are read lazily
            self.labels = data["labels"]
            self.expert_mask = data["expert_label_mask"].astype(bool)
            self.subj_ind = data["subj_ind"]
            self.zero_window_mask = (
                data["zero_window_mask"] if "zero_window_mask" in data.files else None
            )
        self.windows = _memmap_npz_member(npz_path, "windowed_ics")

        # One random test subject, shared by all the figures of the dataset. Only
        # subjects with both a brain and a non-brain annotated IC are eligible.
        # For EPIC, a "subject" is a 10-min EEG segment with its own ICA, containing
        # both brain and artifact ICs. If no subject is eligible, the examples are
        # drawn from all the annotated ICs of the dataset.
        brain = self.expert_mask & (self.labels == 0)
        nonbrain = self.expert_mask & (self.labels != 0)
        eligible = sorted(
            set(self.subj_ind[brain]).intersection(self.subj_ind[nonbrain])
        )
        # Same seed -> same subject for all classifiers of a given dataset
        rng = np.random.default_rng(seed)
        self.subject = int(rng.choice(eligible)) if eligible else None
        self.pool = (
            np.ones(len(self.labels), dtype=bool)
            if self.subject is None
            else self.subj_ind == self.subject
        )

    def sample(self, centroid_index, brain, codebooks, n_centroids=128):
        """Random window of a (non-)brain annotated IC assigned to the centroid.

        Returns a dict (ic, window, label, shift, samples) or None if no window of
        the selected subject was assigned to the centroid.
        """
        j, c = divmod(centroid_index, n_centroids)
        ic_mask = self.pool & self.expert_mask
        ic_mask &= (self.labels == 0) if brain else (self.labels != 0)
        ics = np.where(ic_mask)[0]
        if len(ics) == 0:
            return None
        matched = np.argwhere(np.asarray(self.assignments[ics, j, :]) == c)
        rng = np.random.default_rng([self.seed, centroid_index, int(brain)])
        for k in rng.permutation(len(matched)):
            ic, w = ics[matched[k, 0]], int(matched[k, 1])
            if self.zero_window_mask is not None and self.zero_window_mask[ic, w]:
                continue
            samples = np.array(self.windows[ic, w])
            if not np.any(samples):
                continue
            # Position of the best-matching centroid-length segment in the window
            _, shift, _ = _asignment_step(samples[None, :], codebooks[j], "cosine", None)
            return dict(
                ic=int(ic),
                window=w,
                label=int(self.labels[ic]),
                shift=int(np.ravel(shift)[0]),
                samples=samples,
            )
        return None


def _memmap_npz_member(npz_path, name):
    """Memory-map an array stored (uncompressed) in an .npz file."""
    with zipfile.ZipFile(npz_path) as z:
        info = z.getinfo(f"{name}.npy")
        assert info.compress_type == zipfile.ZIP_STORED, "npz must be uncompressed"
    with open(npz_path, "rb") as f:
        f.seek(info.header_offset + 26)
        name_len, extra_len = np.frombuffer(f.read(4), dtype="<u2")
        f.seek(info.header_offset + 30 + int(name_len) + int(extra_len))
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        offset = f.tell()
    return np.memmap(
        npz_path,
        dtype=dtype,
        mode="r",
        offset=offset,
        shape=shape,
        order="F" if fortran else "C",
    )


def sample_examples(df, source, codebooks):
    """Draw a brain and a non-brain example window for each centroid of df."""
    examples = {}
    for idx in df["centroid_index"].astype(int):
        examples[idx] = {
            "brain": source.sample(idx, True, codebooks),
            "nonbrain": source.sample(idx, False, codebooks),
        }
    return examples


def examples_to_frame(examples, source):
    rows = []
    for idx, ex in examples.items():
        for kind, e in ex.items():
            rows.append(
                {
                    "centroid_index": idx,
                    "kind": kind,
                    "subject": source.subject,
                    "ic": None if e is None else e["ic"],
                    "window": None if e is None else e["window"],
                    "label": None if e is None else CLASS_LABELS[e["label"]],
                }
            )
    return pd.DataFrame(rows)


def plot_centroids(
    df,
    codebooks,
    class_name,
    output_path,
    num_centroids,
    sampling_rate,
    psd_method="mtm",
    examples=None,
    subject=None,
):
    """Plot top centroids with time-domain waveforms and PSDs.

    Args:
        examples: Optional {centroid_index: {"brain": ex, "nonbrain": ex}} (see
            ExampleSource.sample). Adds two columns with an example window of a
            brain and of a non-brain annotated IC matched to the centroid.
        subject: Test subject the examples were drawn from (shown in the titles).
        sampling_rate: Sampling rate (Hz) of the centroids (dataset-dependent).
        psd_method: Method for PSD computation, either "mtm" (multitaper) or "welch"
    """
    from mne.time_frequency import psd_array_multitaper
    from scipy import signal

    # Grid: k rows (one per centroid), 2 columns (time domain + PSD)
    n_rows = num_centroids
    n_cols = 4 if examples is not None else 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, row in df.iterrows():
        idx = int(row["centroid_index"])
        codebook_idx = idx // 128
        centroid_idx = idx % 128
        waveform = codebooks[codebook_idx][centroid_idx]

        # Time domain plot (left column)
        time_axis = np.arange(len(waveform)) / sampling_rate
        axes[i, 0].plot(time_axis, waveform)
        axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_title(
            f"#{idx} {CLASS_LABELS[codebook_idx]} | "
            f"cov {row['covariance']:.1e} | "
            rf"$\bar{{\phi}}$ {row['mean_shap']:+.4f}",
            fontsize=10,
        )

        # Occurrence rate ratio (Brain / Not Brain)
        rate_ratio = row["rate_in_brain"] / max(row["rate_not_brain"], 1e-12)
        axes[i, 0].text(
            0.02,
            0.98,
            f"Rate ratio (brain / non-brain): {rate_ratio:.2f}",
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
        axes[i, 1].set_title(f"PSD (Peak: {max_freq:.1f} Hz)")
        axes[i, 1].set_xlim(0, 100)
        visible = psd_db[freqs <= 100]
        visible = visible[np.isfinite(visible)]
        if visible.size:
            margin = 0.05 * (visible.max() - visible.min())
            axes[i, 1].set_ylim(visible.min() - margin, visible.max() + margin)

        if examples is not None:
            for col, (kind, title) in zip(
                (2, 3), (("brain", "brain"), ("nonbrain", "non-brain"))
            ):
                ax = axes[i, col]
                ex = examples[idx][kind]
                subj_str = "" if subject is None else f", subject {subject}"
                if ex is None:
                    ax.text(
                        0.5,
                        0.5,
                        "no matched window",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"Example in {title} IC{subj_str}")
                    ax.set_axis_off()
                    continue
                t = np.arange(len(ex["samples"])) / sampling_rate
                ax.plot(t, ex["samples"], color="tab:gray", lw=0.8)
                # Shaded: segment of the window matched to the centroid
                seg_start = ex["shift"] / sampling_rate
                seg_end = (ex["shift"] + len(waveform)) / sampling_rate
                ax.axvspan(seg_start, seg_end, color="tab:blue", alpha=0.2)
                # Centroid overlaid on the matched segment, rescaled by the
                # least-squares gain (the cosine assignment ignores amplitude)
                segment = ex["samples"][ex["shift"] : ex["shift"] + len(waveform)]
                gain = np.dot(segment, waveform) / np.dot(waveform, waveform)
                cos_sim = np.dot(segment, waveform) / (
                    np.linalg.norm(segment) * np.linalg.norm(waveform)
                )
                ax.plot(
                    t[ex["shift"] : ex["shift"] + len(waveform)],
                    gain * waveform,
                    color="tab:red",
                    lw=1.0,
                    label=f"centroid (scaled), cos = {cos_sim:.2f}",
                )
                ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
                ax.set_xlabel("Time (s)")
                ic_class = CLASS_LABELS[ex["label"]]
                ax.set_title(
                    f"Example in {title} IC ({ic_class}){subj_str}, "
                    f"IC {ex['ic']}, window {ex['window']}",
                    fontsize=9,
                )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def load_data(eval_config, background_size=500, sample_size=1000):
    """Load classifier, background data, and evaluation data.

    The training (background) data always comes from the source dataset
    (emotion_study), while the evaluation data comes from eval_config's dataset.

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

    # Base filename is classifier/valseg/cmmn specific (source-side stem).
    output_base_filename = get_base_results_filename(eval_config)

    # Load background data (source training data: emotion_study subjects 8-35)
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

    # Load evaluation data (target dataset). emotion_study test subjects live in
    # the "full" split, same as cue/epic.
    eval_file = (
        eval_config.root
        / f"data/{eval_config.eval_dataset}/bowav/full"
        / f"{output_base_filename}.npz"
    )
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


def _make_explainer(classifier, clf, background_data):
    """SHAP explainer appropriate for the classifier type.

    random_forest -> TreeExplainer (interventional); logistic -> LinearExplainer.
    Both produce SHAP values of shape (n_samples, n_features, n_classes).

    TODO: shap wraps the background array in maskers.Independent, whose default
    max_samples=100 randomly subsamples it, so the explainers use only 100 of the
    background rows (LinearExplainer: only the mean of those rows matters). To use
    all rows, pass shap.maskers.Independent(background_data, max_samples=len(
    background_data)) instead of the array, and recompute the SHAP caches.
    """
    if classifier == "random_forest":
        return shap.TreeExplainer(
            clf, background_data, feature_perturbation="interventional"
        )
    elif classifier == "logistic":
        return shap.LinearExplainer(clf, background_data)
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")


def compute_shap_importance(
    classifier,
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
        classifier: "random_forest" or "logistic" (selects the SHAP explainer).
        calibrate_idf_fn: IDF reweighting closure, or None to use the source IDF
            unchanged (source dataset).
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

    # Calibrate IDF for all eval data (use slice(None) to select all subjects).
    # For the source dataset there is no reweighting -> use the estimator as-is.
    if calibrate_idf_fn is not None:
        calibrated_estimator = calibrate_idf_fn(estimator, slice(None))
    else:
        calibrated_estimator = estimator

    # Transform background data with original scaler (from training dataset)
    background_data = estimator["scaler"].transform(background_data).toarray()

    # Transform eval data with calibrated scaler (for cross-dataset generalization)
    eval_data_transformed = (
        calibrated_estimator["scaler"].transform(eval_data).toarray()
    )

    # Create SHAP explainer and compute values
    explainer = _make_explainer(classifier, clf, background_data)
    if classifier == "random_forest":
        # Additivity check disabled: it flags small numerical deviations between
        # the sum of SHAP values and the model output (below 7e-3 in the settings
        # we checked). LinearExplainer has no such kwarg.
        shap_values = explainer.shap_values(
            eval_data_transformed, check_additivity=False
        )
    else:
        shap_values = explainer.shap_values(eval_data_transformed)
    shap_values = np.asarray(shap_values)

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


def run_analysis(
    classifier,
    dataset,
    output_dir,
    num_centroids,
    sort_by,
    sample_size,
    psd_method,
    root,
    examples=True,
    seed=0,
    validation_segment=5,
):
    """Run the full SHAP feature-importance analysis for one (classifier,
    dataset) combination and write plots + CSVs under output_dir/<clf>_<ds>
    (with a _50min suffix for the 50-min validated classifiers)."""
    typer.echo(
        f"\n=== {classifier} / {dataset} / l_val={validation_segment} min "
        f"(sort_by={sort_by}, sample_size={sample_size}, psd_method={psd_method}) ==="
    )

    # Load the train_config from the classifier's config file
    config_file = (
        root / "config/_private" / f"{config_stem(classifier, validation_segment)}.txt"
    )
    train_config = parse_config_file_args(config_file, feature_extractor="bowav")

    # epic ICs are 10-min; cue/emotion_study use the default (50 min).
    minutes_per_ic = 10 if dataset == "epic" else None

    # Create EvalConfig object
    eval_config = EvalConfig(
        eval_dataset=dataset,
        train_config=train_config,
        root=root,
        cmmn_filter=None,
        minutes_per_ic=minutes_per_ic,
    )

    # Per-combination output directory
    output_path = Path(output_dir) / combo_dirname(
        classifier, dataset, validation_segment
    )
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

    # IDF reweighting only for target datasets; the source uses its own IDF.
    if dataset == SOURCE_DATASET:
        calibrate_idf_fn = None
    else:
        calibrate_idf_fn = make_calibrate_idf_fn(eval_config)

    # Load codebooks (dataset-appropriate sampling rate) and its rate
    codebooks = load_codebooks(eval_config)
    sampling_rate = 500 if dataset == "cue" else 256

    # Example windows: one random test subject per dataset, shared by the positive
    # and negative figures (and by all classifiers).
    source = ExampleSource(eval_config, seed) if examples else None
    if source is not None:
        typer.echo(f"Example windows drawn from test subject {source.subject}")

    # Define cache file path
    cache_file = output_path / "shap_cache.npz"

    # Compute SHAP importance (with caching)
    typer.echo("Computing SHAP values...")
    if cache_file.exists():
        typer.echo(f"Cache found at {cache_file}")
    shap_values, predictions, eval_data, eval_labels = compute_shap_importance(
        classifier,
        estimator,
        background_data,
        eval_data,
        eval_labels,
        calibrate_idf_fn,
        cache_file,
    )

    # Plot 1: Positive mean SHAP centroids
    typer.echo(
        f"Analyzing Brain class - Positive mean SHAP centroids "
        f"(sort_by={sort_by}, psd_method={psd_method})..."
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
    pos_examples = (
        sample_examples(positive_df, source, codebooks) if source is not None else None
    )
    if pos_examples is not None:
        examples_to_frame(pos_examples, source).to_csv(
            output_path / f"brain_positive_shap_{sort_by}_{psd_method}_examples.csv",
            index=False,
        )
    plot_centroids(
        positive_df,
        codebooks,
        "Brain (Positive SHAP)",
        output_path / f"brain_positive_shap_{sort_by}_{psd_method}_top_centroids.pdf",
        num_centroids,
        sampling_rate,
        psd_method,
        pos_examples,
        None if source is None else source.subject,
    )

    # Plot 2: Negative mean SHAP centroids
    typer.echo(
        f"Analyzing Brain class - Negative mean SHAP centroids "
        f"(sort_by={sort_by}, psd_method={psd_method})..."
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
    neg_examples = (
        sample_examples(negative_df, source, codebooks) if source is not None else None
    )
    if neg_examples is not None:
        examples_to_frame(neg_examples, source).to_csv(
            output_path / f"brain_negative_shap_{sort_by}_{psd_method}_examples.csv",
            index=False,
        )
    plot_centroids(
        negative_df,
        codebooks,
        "Brain (Negative SHAP)",
        output_path / f"brain_negative_shap_{sort_by}_{psd_method}_top_centroids.pdf",
        num_centroids,
        sampling_rate,
        psd_method,
        neg_examples,
        None if source is None else source.subject,
    )

    typer.echo(f"Analysis complete. Results saved to {output_path}")


@app.command()
def main(
    classifier: str = typer.Option(
        "random_forest",
        help="Classifier: 'logistic', 'random_forest', or 'all'",
    ),
    dataset: str = typer.Option(
        "cue",
        help="Eval dataset: 'emotion_study', 'cue', 'epic', or 'all'",
    ),
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
    examples: bool = typer.Option(
        True,
        help="Add two columns with example windows of a brain and a non-brain "
        "annotated IC matched to each centroid (random test subject).",
    ),
    seed: int = typer.Option(
        0, help="Seed for the random choice of the test subject and example windows"
    ),
    validation_segment: int = typer.Option(
        5,
        help="Validation segment length (minutes) of the classifier to explain: "
        "5 (count pooling) or 50 (majority vote)",
    ),
):
    """
    SHAP Feature Importance Analysis for BoWav classifiers.

    Generates two Brain-class plots (positive / negative mean SHAP) for each
    selected (classifier, dataset) combination:
    - Plot 1: Positive mean SHAP centroids sorted by covariance or max SHAP
    - Plot 2: Negative mean SHAP centroids sorted by covariance or min SHAP (most negative)
    """
    # Validate arguments
    if sort_by not in ["covariance", "magnitude"]:
        typer.echo("Error: --sort-by must be 'covariance' or 'magnitude'")
        raise typer.Exit(code=1)

    if psd_method not in ["mtm", "welch"]:
        typer.echo("Error: --psd-method must be 'mtm' or 'welch'")
        raise typer.Exit(code=1)

    if validation_segment not in SUPPORTED_VALIDATION_SEGMENTS:
        typer.echo(
            f"Error: --validation-segment must be one of {SUPPORTED_VALIDATION_SEGMENTS}"
        )
        raise typer.Exit(code=1)

    classifiers = SUPPORTED_CLASSIFIERS if classifier == "all" else [classifier]
    datasets = SUPPORTED_DATASETS if dataset == "all" else [dataset]

    for c in classifiers:
        if c not in SUPPORTED_CLASSIFIERS:
            typer.echo(
                f"Error: --classifier must be one of {SUPPORTED_CLASSIFIERS} or 'all'"
            )
            raise typer.Exit(code=1)
    for d in datasets:
        if d not in SUPPORTED_DATASETS:
            typer.echo(f"Error: --dataset must be one of {SUPPORTED_DATASETS} or 'all'")
            raise typer.Exit(code=1)

    typer.echo(
        f"Starting SHAP Feature Importance Analysis "
        f"(classifiers={classifiers}, datasets={datasets})..."
    )

    # Setup root path
    root = Path(__file__).absolute().parents[1]

    for c in classifiers:
        for d in datasets:
            run_analysis(
                c,
                d,
                output_dir,
                num_centroids,
                sort_by,
                sample_size,
                psd_method,
                root,
                examples,
                seed,
                validation_segment,
            )

    typer.echo("All analyses complete.")


if __name__ == "__main__":
    app()
