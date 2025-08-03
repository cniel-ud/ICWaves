# %%
# Set OMP constants to use only 6 CPUs
import os

os.environ["OMP_NUM_THREADS"] = "8"

# Imports and setup
from pathlib import Path
import numpy as np
from icwaves.evaluation.evaluation import (
    load_classifier,
    eval_classifier_per_subject_brain_F1,
)
from icwaves.evaluation.config import EvalConfig
from icwaves.data.loading import get_feature_extractor, load_data_bundles
from icwaves.viz import plot_line_with_error_area
from icwaves.evaluation.iclabel import calculate_iclabel_f1_scores


# %%
def get_cmmn_filter_options(eval_dataset, is_classifier_trained_on_normalized_data):
    if eval_dataset == "emotion_study":
        if is_classifier_trained_on_normalized_data:
            cmmn_filter_options = ["original"]
        else:
            cmmn_filter_options = [None]
    else:
        if is_classifier_trained_on_normalized_data:
            cmmn_filter_options = ["original"]
        else:
            cmmn_filter_options = [None, "original", "subj_to_subj"]
    return cmmn_filter_options


validation_times = np.r_[
    [9.0, 19.5, 30.0, 39.0, 49.5, 60],
    np.arange(2 * 60, 5 * 60 + 1, 60),
    np.arange(5 * 60, 5 * 60 + 1, 60),
    np.arange(10 * 60, 50 * 60 + 1, 5 * 60),
].astype(float)
root = Path().absolute().parent
mean_std_df = {}
is_classifier_trained_on_normalized_data = False
for eval_dataset in ["cue", "emotion_study"]:
    is_cmmn_filter_resampled = True if eval_dataset == "cue" else False
    mean_std_df[eval_dataset] = {}
    cmmn_filter_options = get_cmmn_filter_options(
        eval_dataset, is_classifier_trained_on_normalized_data
    )
    for cmmn_filter in cmmn_filter_options:
        cmmn_filter_str = str(cmmn_filter)
        mean_std_df[eval_dataset][cmmn_filter_str] = {}
        for feature_extractor_str in [
            "bowav",
            "psd_autocorr",
        ]:  # , "bowav_psd_autocorr"]:
            mean_std_df[eval_dataset][cmmn_filter_str][feature_extractor_str] = {}
            # Configuration
            config = EvalConfig(
                eval_dataset=eval_dataset,
                feature_extractor=feature_extractor_str,
                classifier_type="random_forest",
                validation_segment_length=300,
                root=root,
                cmmn_filter=cmmn_filter,
                is_cmmn_filter_resampled=is_cmmn_filter_resampled,
                is_classifier_trained_on_normalized_data=is_classifier_trained_on_normalized_data,
            )
            agg_method = {}
            # Load data
            print(
                f"Getting data from {eval_dataset}, using CMMN filter {cmmn_filter_str}, and building feature extractor for {feature_extractor_str}..."
            )
            data_bundles = load_data_bundles(config)
            feature_extractor = get_feature_extractor(
                feature_extractor_str, data_bundles
            )
            feature_extractor = {feature_extractor_str: feature_extractor}
            for classifier_type in ["random_forest"]:  # ["random_forest", "logistic"]:
                config.classifier_type = classifier_type
                mean_std_df[eval_dataset][cmmn_filter_str][feature_extractor_str][
                    classifier_type
                ] = {}
                for validation_segment_len in [300, -1]:
                    config.validation_segment_length = validation_segment_len
                    print(
                        f"Validation segment length: {validation_segment_len} | classifier type: {classifier_type}"
                    )
                    # Load classifier
                    clf, best_params = load_classifier(
                        config.path_to_classifier[feature_extractor_str]
                    )
                    clf = {feature_extractor_str: clf}

                    # TODO: make this cleaner
                    # for 'psd_autocorr', the training segment length is in samples. Since the sampling rate
                    # for 'emotion' and 'cue' are 256 Hz and 500 Hz, we need to convert that length.
                    if (
                        eval_dataset == "cue"
                        and "psd_autocorr" in best_params["training_segment_length"]
                    ):
                        best_params["training_segment_length"]["psd_autocorr"] = int(
                            best_params["training_segment_length"]["psd_autocorr"]
                            / 256
                            * 500
                        )

                    # Evaluate
                    print("Computing F1 score...")
                    agg_method[feature_extractor_str] = best_params[
                        "input_or_output_aggregation_method"
                    ]
                    mean_std_df[eval_dataset][cmmn_filter_str][feature_extractor_str][
                        classifier_type
                    ][validation_segment_len] = eval_classifier_per_subject_brain_F1(
                        config,
                        clf,
                        feature_extractor,
                        validation_times,
                        data_bundles,
                        agg_method,
                        best_params["training_segment_length"],
                    )
# %%
import matplotlib.pyplot as plt

global_x_ticks = np.array([0.15, 0.5, 1, 2, 3, 5, 10, 30, 50])
# colors = {"bowav/logistic": "blue", "psd_autocorr/logistic": "green", "bowav/random_forest": "orange", "psd_autocorr/random_forest": "purple"}
# colors = {"random_forest/None": "green", "random_forest/subj_to_subj": "blue", "random_forest/original": "orange"}
colors_options = {
    1: {"bowav": "orange", "psd_autocorr": "blue"},
    2: {
        "bowav/None": "orange",
        "bowav/sub_to_subj": "green",
        "psd_autocorr/None": "purple",
        "psd_autocorr/subj_to_subj": "magenta",
    },
}

val_seg_len_map = {-1: "All", 300: "5-minutes"}
# The *.mat files with the ICLabel labels were manually created for the following prediction times
#     9.0,    19.5,    30.0,    39.0,    49.5
#    60.0,   120.0,   180.0,   240.0,   300.0
#   300.0,   600.0,   900.0,  1200.0,  1500.0
#  1800.0,  2100.0,  2400.0,  2700.0,  3000.0
validation_times_ = 1.5 * (validation_times // 1.5)

for eval_dataset in ["cue", "emotion_study"]:
    iclabel_data_dir = root.joinpath(f"data/{eval_dataset}/ICLabels")
    subj_ids = (
        list(range(1, 8)) if eval_dataset == "emotion_study" else list(range(1, 13))
    )
    iclabel_df = calculate_iclabel_f1_scores(
        iclabel_data_dir, subj_ids, validation_times_
    )
    iclabel_df = iclabel_df.rename(
        columns={
            "Brain F1 score - iclabel": "iclabel",
        }
    )

    cmmn_filter_options = get_cmmn_filter_options(
        eval_dataset, is_classifier_trained_on_normalized_data
    )
    n_filters = len(cmmn_filter_options)
    colors = colors_options[n_filters]

    for validation_segment_len in [-1, 300]:
        for classifier_type in ["random_forest"]:  # ["random_forest", "logistic"]:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax = plot_line_with_error_area(
                ax,
                iclabel_df,
                "Prediction window [minutes]",
                "iclabel",
                "StdDev - iclabel",
                color="red",
            )
            for feature_extractor_str in [
                "bowav",
                "psd_autocorr",
            ]:  # , "bowav_psd_autocorr"]:
                for cmmn_filter in cmmn_filter_options:
                    cmmn_filter = str(cmmn_filter)
                    df = mean_std_df[eval_dataset][cmmn_filter][feature_extractor_str][
                        classifier_type
                    ][validation_segment_len]
                    # rename columns to also include the classifier type (e.g., "Brain F1 score - bowav" -> "Brain F1 score - bowav-logistic")
                    df = df.rename(
                        columns={
                            f"Brain F1 score - {feature_extractor_str} - cmmn-{cmmn_filter}": f"{feature_extractor_str}/{cmmn_filter}",
                            f"StdDev - {feature_extractor_str} - cmmn-{cmmn_filter}": f"StdDev - {feature_extractor_str}/{cmmn_filter}",
                        }
                    )
                    colors_key_pattern = (
                        f"{feature_extractor_str}/{cmmn_filter}"
                        if n_filters == 2
                        else f"{feature_extractor_str}"
                    )
                    ax = plot_line_with_error_area(
                        ax,
                        df,
                        "Prediction window [minutes]",
                        f"{feature_extractor_str}//{cmmn_filter}",
                        f"StdDev - {feature_extractor_str}/{cmmn_filter}",
                        color=colors[colors_key_pattern],
                    )

            ax.set_xscale("log")
            ax.set_xticks(global_x_ticks, labels=global_x_ticks)
            ax.set_xlim(global_x_ticks[0], 50)
            ax.set_xlabel("Prediction window [minutes]")
            ax.set_ylabel("Mean Brain F1 score")
            ax.set_title(
                f"Mean brain F1 score across subjects | dataset:{eval_dataset} | validation segment length:{val_seg_len_map[validation_segment_len]}"
            )
            ax.legend()
            ax.grid(True)

# %%
colors

# %%
colors_key_pattern

# %%
