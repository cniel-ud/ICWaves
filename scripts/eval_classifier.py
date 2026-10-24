#!/usr/bin/env python3
"""
Evaluate the trained (emotion_study) classifiers on the evaluation datasets.

For each selected (eval_dataset, eval CMMN filter, feature_extractor,
classifier_type, validation_segment_length) combination, the per-subject Brain
F1 score is computed over a sweep of prediction windows, together with the
ICLabel reference scores. Results are collected in a flat summary CSV, and a
comparison plot (feature extractors vs ICLabel) is created per dataset and
classifier type.

Usage:
    python scripts/eval_classifier.py
    python scripts/eval_classifier.py --eval-dataset cue --eval-dataset epic
    python scripts/eval_classifier.py --feature-extractor bowav --classifier-type logistic
"""

import os

# Use only 8 CPUs. Must be set before numpy/scipy/sklearn are imported.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

from argparse import Namespace
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer
from matplotlib import pyplot as plt

from icwaves.data.loading import get_feature_extractor, load_data_bundles
from icwaves.data.types import DataBundle
from icwaves.evaluation.config import EvalConfig
from icwaves.evaluation.evaluation import (
    eval_classifier_per_subject_brain_F1,
    get_results_filepath,
    load_estimator,
)
from icwaves.evaluation.iclabel import compute_iclabel_scores_for_dataset
from icwaves.evaluation.utils import (
    get_eval_cmmn_filter_options,
    make_calibrate_idf_fn,
    sl2min,
)
from icwaves.file_utils import get_cmmn_suffix, parse_config_file_args
from icwaves.viz import create_comparison_plot

app = typer.Typer()


class EvalDataset(str, Enum):
    emotion_study = "emotion_study"
    cue = "cue"
    epic = "epic"


class FeatureExtractor(str, Enum):
    bowav = "bowav"
    psd_autocorr = "psd_autocorr"


class ClassifierType(str, Enum):
    random_forest = "random_forest"
    logistic = "logistic"


RESULTS_COLUMNS = [
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

# Prediction windows (seconds) over which the classifiers are evaluated
VALIDATION_TIMES = np.r_[
    [9.0, 19.5, 30.0, 39.0, 49.5, 60],
    np.arange(2 * 60, 5 * 60 + 1, 60),
    np.arange(5 * 60, 5 * 60 + 1, 60),
    np.arange(10 * 60, 50 * 60 + 1, 5 * 60),
].astype(float)

# EPIC ICs are 10 minutes long: evaluate them on a single 10-minute window
EPIC_MINUTES_PER_IC = 10
EPIC_PREDICTION_WINDOW = 10 * 60

# Validation segment lengths used to train the classifiers (-1: whole IC)
VALIDATION_SEGMENT_LENS = [300, -1]

# CMMN filter applied to the training data (None or "normed-barycenter")
TRAIN_CMMN_FILTER = None
TRAIN_USE_IDF = True


def get_prediction_windows(eval_dataset: str) -> np.ndarray:
    """Prediction windows (in seconds) used to evaluate eval_dataset."""
    if eval_dataset == EvalDataset.epic:
        return VALIDATION_TIMES[VALIDATION_TIMES == EPIC_PREDICTION_WINDOW]
    return VALIDATION_TIMES


def get_minutes_per_ic(eval_dataset: str) -> Optional[int]:
    """Minutes of each IC used for evaluation. None: same value as in training."""
    return EPIC_MINUTES_PER_IC if eval_dataset == EvalDataset.epic else None


def load_train_config(
    config_folder: Path,
    classifier_type: str,
    feature_extractor: str,
    validation_segment_len: int,
) -> Namespace:
    """Parse the config file with the args used to train a classifier."""
    cmmn_str = get_cmmn_suffix(TRAIN_CMMN_FILTER)
    idf_str = "_idf" if TRAIN_USE_IDF else ""
    config_file_name = (
        f"{classifier_type}_{sl2min(validation_segment_len)}{cmmn_str}{idf_str}.txt"
    )
    typer.echo(f"Reading config file: {config_file_name}")
    return parse_config_file_args(config_folder / config_file_name, feature_extractor)


def evaluate_config(
    config: EvalConfig,
    data_bundles: dict[str, DataBundle],
    validation_times: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate one trained classifier on config.eval_dataset.

    Args:
        config: Evaluation configuration
        data_bundles: Data bundles of config.eval_dataset for config.feature_extractor
        validation_times: Array of validation times (prediction windows) in seconds

    Returns:
        DataFrame with RESULTS_COLUMNS: one row per prediction window
    """
    feature_extractor_str = config.feature_extractor
    train_config = config.train_config
    feature_extractor_dict = {
        feature_extractor_str: get_feature_extractor(
            feature_extractor_str, data_bundles
        )
    }

    clf, best_params = load_estimator(config.path_to_classifier[feature_extractor_str])
    clf_dict = {feature_extractor_str: clf}
    agg_method = {
        feature_extractor_str: best_params["input_or_output_aggregation_method"]
    }

    # Handle conversion for PSD autocorr segment length, as the sampling rate of
    # cue and emotion_study are 500 Hz and 256 Hz, respectively
    # TODO: improve this manual fix
    if (
        config.eval_dataset == EvalDataset.cue
        and "psd_autocorr" in best_params["training_segment_length"]
    ):
        best_params["training_segment_length"]["psd_autocorr"] = int(
            best_params["training_segment_length"]["psd_autocorr"] / 256 * 500
        )

    if (
        config.eval_dataset != EvalDataset.emotion_study
        and "bowav" in feature_extractor_str
    ):
        calibrate_idf_fn = make_calibrate_idf_fn(config)
    else:
        calibrate_idf_fn = None

    typer.echo("Computing F1 score...")
    results = eval_classifier_per_subject_brain_F1(
        config,
        clf_dict,
        feature_extractor_dict,
        validation_times,
        data_bundles,
        agg_method,
        best_params["training_segment_length"],
        get_results_filepath(config),
        calibrate_idf_fn,
    )

    cmmn_filter = str(config.cmmn_filter)
    return pd.DataFrame(
        {
            "eval_dataset": config.eval_dataset,
            "cmmn_filter": cmmn_filter,
            "feature_extractor": feature_extractor_str,
            "classifier_type": config.classifier_type,
            "is_normalized": train_config.cmmn_filter == "normed-barycenter",
            "validation_segment_len": config.validation_segment_length,
            "prediction_window": results["Prediction window [minutes]"],
            "mean_f1": results[
                f"Brain F1 score - {feature_extractor_str} - cmmn-{cmmn_filter}"
            ],
            "std_f1": results[f"StdDev - {feature_extractor_str} - cmmn-{cmmn_filter}"],
        },
        columns=RESULTS_COLUMNS,
    )


def evaluate_dataset(
    eval_dataset: str,
    feature_extractors: list[str],
    classifier_types: list[str],
    root: Path,
) -> pd.DataFrame:
    """Evaluate all the selected classifiers, and ICLabel, on eval_dataset."""
    validation_times = get_prediction_windows(eval_dataset)
    minutes_per_ic = get_minutes_per_ic(eval_dataset)
    config_folder = root / "config/_private"
    train_configs = {
        (feature_extractor, classifier_type, sl): load_train_config(
            config_folder, classifier_type, feature_extractor, sl
        )
        for feature_extractor in feature_extractors
        for classifier_type in classifier_types
        for sl in VALIDATION_SEGMENT_LENS
    }

    dataset_results = []
    subj_ids = None
    for eval_cmmn_filter in get_eval_cmmn_filter_options(
        eval_dataset, TRAIN_CMMN_FILTER
    ):
        for feature_extractor in feature_extractors:
            # The data bundles only depend on the dataset, the eval CMMN filter,
            # and the feature extractor: load them once for all classifiers
            data_bundles = None
            for classifier_type in classifier_types:
                for sl in VALIDATION_SEGMENT_LENS:
                    train_config = train_configs[
                        (feature_extractor, classifier_type, sl)
                    ]
                    typer.echo(
                        f"\nRunning evaluation for: dataset={eval_dataset}, "
                        f"filter={eval_cmmn_filter}, feature={feature_extractor}, "
                        f"classifier={classifier_type}, "
                        f"validation_segment_len={train_config.validation_segment_length}"
                    )
                    config = EvalConfig(
                        eval_dataset=eval_dataset,
                        train_config=train_config,
                        root=root,
                        cmmn_filter=eval_cmmn_filter,
                        minutes_per_ic=minutes_per_ic,
                    )
                    subj_ids = config.subj_ids
                    if data_bundles is None:
                        data_bundles = load_data_bundles(config)
                    dataset_results.append(
                        evaluate_config(config, data_bundles, validation_times)
                    )

    typer.echo(f"Computing ICLabel scores for {eval_dataset}...")
    mean_std_f1_iclabel, per_subject_f1_iclabel = compute_iclabel_scores_for_dataset(
        eval_dataset, subj_ids, validation_times, root
    )
    dataset_results.append(mean_std_f1_iclabel)
    iclabel_dir = root / "results" / eval_dataset / "evaluation"
    iclabel_dir.mkdir(parents=True, exist_ok=True)
    per_subject_f1_iclabel.to_csv(iclabel_dir / "ICLabel.csv", index=False)

    return pd.concat(dataset_results, ignore_index=True)


def plot_feature_comparison(
    all_results: pd.DataFrame,
    eval_dataset: str,
    classifier_type: str,
    validation_segment_len: float,
    plot_dir: Path,
    train_data_str: str,
) -> None:
    """Plot mean Brain F1 vs prediction window for each feature extractor and ICLabel."""
    validation_segment_len_str = sl2min(validation_segment_len)
    save_path = (
        plot_dir / f"feature_comparison_{eval_dataset}_{validation_segment_len_str}"
        f"_{classifier_type}_{train_data_str}.pdf"
    )
    fig, _ = create_comparison_plot(
        all_results,
        fixed_params={
            "eval_dataset": eval_dataset,
            "validation_segment_len": validation_segment_len,
            "classifier_type": classifier_type,
        },
        vary_by="feature_extractor",
        save_path=save_path,
        add_title=False,
    )
    plt.close(fig)
    typer.echo(f"Saved plot to {save_path}")


@app.command()
def main(
    eval_dataset: Annotated[
        Optional[list[EvalDataset]],
        typer.Option(help="Dataset(s) to evaluate on. Repeat to select several."),
    ] = None,
    feature_extractor: Annotated[
        Optional[list[FeatureExtractor]],
        typer.Option(
            help="Feature extractor(s) to evaluate. Repeat to select several."
        ),
    ] = None,
    classifier_type: Annotated[
        Optional[list[ClassifierType]],
        typer.Option(help="Classifier type(s) to evaluate. Repeat to select several."),
    ] = None,
):
    """Evaluate the trained classifiers and ICLabel on the selected datasets.

    Options left unset default to all their values.
    """
    eval_datasets = [d.value for d in (eval_dataset or list(EvalDataset))]
    feature_extractors = [
        f.value for f in (feature_extractor or list(FeatureExtractor))
    ]
    classifier_types = [c.value for c in (classifier_type or list(ClassifierType))]

    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    train_data_str = "normalized" if TRAIN_CMMN_FILTER == "normed-barycenter" else "raw"

    all_results = pd.concat(
        [
            evaluate_dataset(d, feature_extractors, classifier_types, root)
            for d in eval_datasets
        ],
        ignore_index=True,
    )

    results_file = results_dir / f"mean_std_brain_f1_{train_data_str}.csv"
    all_results.to_csv(results_file, index=False)
    typer.echo(f"Saved results to {results_file}")

    # One plot per evaluated (dataset, classifier type, validation segment length).
    # ICLabel rows have classifier_type "iclabel", so they are excluded here and
    # added to every plot by create_comparison_plot. EPIC is not plotted: it is
    # evaluated on a single prediction window, so there is no curve to compare.
    evaluated = all_results[
        all_results["classifier_type"].isin(classifier_types)
        & (all_results["eval_dataset"] != EvalDataset.epic.value)
    ]
    plot_keys = evaluated[
        ["eval_dataset", "classifier_type", "validation_segment_len"]
    ].drop_duplicates()
    for d, c, sl in plot_keys.itertuples(index=False):
        plot_feature_comparison(all_results, d, c, sl, plot_dir, train_data_str)


if __name__ == "__main__":
    app()
