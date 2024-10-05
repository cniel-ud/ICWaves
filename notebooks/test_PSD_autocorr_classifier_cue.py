# %%
"""
A classifier trained using the PSD+autocorr features, computed in the same way as in ICLabel.
The classifier is trained on the emotion study data.
"""
import copy
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import default_rng

from scipy.io import loadmat
from sklearn.metrics import f1_score

from icwaves.feature_extractors.iclabel_features import get_iclabel_features_per_segment
from icwaves.preprocessing import load_or_build_ics_and_labels
from icwaves.viz import plot_line_with_error_area
from icwaves.utils import compute_brain_F1_score_per_subject

plt.rcParams.update({"text.usetex": True, "font.size": 12})
# %%
"""
Get trained classifier and dictionaries
"""
root = Path(__file__).parent.parent
print(f"Root folder is {root}")

train_subj_ids = list(range(8, 36))
train_subj_ids.remove(22)  # subject 22 is not present
cue_subj_ids = list(range(1, 13))


# TODO: generalize these scripts to work with any dataset/feature/classifier
class Args:
    path_to_raw_data = root.joinpath("data/cue/raw_data_and_IC_labels")
    path_to_results = root.joinpath("results/emotion_study/classifier")
    path_to_preprocessed_data = root.joinpath("data/cue/preprocessed_data")
    window_length = 1.5
    minutes_per_ic = 50.0
    num_clusters = 128
    centroid_length = 1.0
    penalty = "elasticnet"
    regularization_factor = [0.01, 0.1, 1.0]
    expert_weight = [float(x) for x in [1, 16, 32, 64]]
    l1_ratio = [float(x) for x in [0.1, 0.5, 1.0]]
    subj_ids = train_subj_ids
    training_segment_length = [float(x) for x in [10, 30, 60, 120, 300]]
    validation_segment_length = -1  # NOTE: not being used rn


args = Args()
rng = default_rng(13)

results_folder = Path(args.path_to_results)
results_folder.mkdir(exist_ok=True, parents=True)
# TODO: generate this file name dynamically
results_file = "PSD_autocorr_valseg_none.pkl"
results_file = results_folder.joinpath(results_file)
clf_path = args.path_to_results.joinpath(results_file)
with clf_path.open("rb") as f:
    results = pickle.load(f)
clf = results["best_estimator"]

best_index = results["rank_test_scores"].argmin()
best_score = results[f"mean_test_scores"][best_index]
best_params = copy.deepcopy(results["params"][best_index])
input_or_output_aggregation_method = best_params["input_or_output_aggregation_method"]
training_segment_length = best_params["n_training_windows_per_segment"]
# %% Load or build preprocessed data
args.subj_ids = cue_subj_ids
ics, labels, srate, expert_label_mask, subj_ind, noisy_labels = (
    load_or_build_ics_and_labels(args)
)
# %%
# Predict on increments of 10 seconds, up to the first 50 minutes
# first minute: 10 seconds
# 1-5 min: 1 minutes
# 5-end: 5 minutes
validation_segment_len_seconds_arr = np.r_[
    np.arange(60, 5 * 60 + 1, 60),
    np.arange(10 * 60, 50 * 60 + 1, 5 * 60),
]
validation_segment_len_samples_arr = validation_segment_len_seconds_arr * srate
validation_segment_len_samples_arr = validation_segment_len_samples_arr.astype(int)

# %%
# average F1 per subject
# Jack knife: when computing mean F1 for each subject, compute mean F1 in the
# hold out to put error bars on the mean F1 across all ICs
# get variance across subjects
columns = [
    "Prediction window [minutes]",
    "Subject ID",
    "Brain F1 score",
    "Number of ICs",
]
f1_scores_df = pd.DataFrame(columns=columns)
# `get_iclabel_features_per_segment` has args (signal, sfreq, use_autocorr, segment_len)
# `feature_extractor` has args (time_series, segment_length)
feature_extractor = (
    lambda time_series, segment_length: get_iclabel_features_per_segment(
        signal=time_series,
        sfreq=srate,
        use_autocorr=True,
        segment_len=segment_length,
    )
)
for validation_segment_length in validation_segment_len_samples_arr:
    print(f"Computing F1 score after aggregating {validation_segment_length} samples")
    for subj_id in cue_subj_ids:
        print(f"Subject {subj_id}")
        subj_mask = subj_ind == subj_id

        brain_f1_score = compute_brain_F1_score_per_subject(
            clf,
            ics,
            labels,
            expert_label_mask,
            input_or_output_aggregation_method,
            feature_extractor,
            validation_segment_length,
            training_segment_length,
            subj_mask,
        )

        f1_scores_df.loc[len(f1_scores_df)] = {
            "Prediction window [minutes]": validation_segment_length / srate / 60,
            "Subject ID": subj_id,
            "Brain F1 score": brain_f1_score,
            "Number of ICs": subj_mask.sum(),
        }
# %%
f1_across_subj_psd_autocorr = f1_scores_df.groupby("Prediction window [minutes]")[
    "Brain F1 score"
].std()
f1_across_subj_psd_autocorr = f1_across_subj_psd_autocorr.rename("StdDev").reset_index()
mean_f1_df = (
    f1_scores_df.groupby("Prediction window [minutes]")["Brain F1 score"]
    .mean()
    .rename("Brain F1 score")
    .reset_index()
)
f1_across_subj_psd_autocorr = pd.merge(
    f1_across_subj_psd_autocorr, mean_f1_df, on="Prediction window [minutes]"
)

f1_across_subj_psd_autocorr = f1_across_subj_psd_autocorr.rename(
    columns={
        "StdDev": "StdDev - PSD_autocorr",
        "Brain F1 score": "Brain F1 score - PSD_autocorr",
    }
)
fig, ax = plt.subplots(figsize=(9, 5))
ax = plot_line_with_error_area(
    ax,
    f1_across_subj_psd_autocorr,
    "Prediction window [minutes]",
    "Brain F1 score - PSD_autocorr",
    "StdDev - PSD_autocorr",
)
ax.set_xscale("log")
ax.set_xticks([1, 2, 3, 5, 10, 30, 50], labels=[1, 2, 3, 5, 10, 30, 50])
ax.set_xlim(1, 50)
ax.set_xlabel("Prediction window [minutes]")
ax.set_ylabel("Mean Brain F1 score")
ax.set_title("Mean brain F1 score across subjects")
ax.legend()
ax.grid(True)
# %%
columns = [
    "Prediction window [minutes]",
    "Subject ID",
    "Brain F1 score",
    "Number of ICs",
]
f1_scores_ICLabel_df = pd.DataFrame(columns=columns)
cue_dir = Path("../data/cue/")
fnames = [f"subj-{i:02}.mat" for i in args.subj_ids]
for test_segment_len in validation_segment_len_seconds_arr:
    subdir = cue_dir.joinpath(f"IC_labels_at_{test_segment_len:.1f}_seconds")
    for subj_id in args.subj_ids:
        file = subdir.joinpath(f"subj-{subj_id:02}.mat")
        with file.open("rb") as f:
            data = loadmat(f)
            expert_label_mask = data["expert_label_mask"].flatten().astype(bool)
            labels = data["labels"] - 1
            noisy_labels = data["noisy_labels"]
        ICLabel_labels = np.argmax(noisy_labels, axis=1)
        y_expert = labels[expert_label_mask]
        y_pred_expert = ICLabel_labels[expert_label_mask]
        brain_f1_score = f1_score(y_expert, y_pred_expert, labels=[0], average=None)
        f1_scores_ICLabel_df.loc[len(f1_scores_ICLabel_df)] = {
            "Prediction window [minutes]": test_segment_len / 60,
            "Subject ID": subj_id,
            "Brain F1 score": brain_f1_score[0],
            "Number of ICs": expert_label_mask.sum(),
        }
# %%
f1_across_subj_iclabel = f1_scores_ICLabel_df.groupby("Prediction window [minutes]")[
    "Brain F1 score"
].std()
f1_across_subj_iclabel = f1_across_subj_iclabel.rename("StdDev - ICLabel").reset_index()
mean_f1_df = (
    f1_scores_ICLabel_df.groupby("Prediction window [minutes]")["Brain F1 score"]
    .mean()
    .rename("Brain F1 score - ICLabel")
    .reset_index()
)
f1_across_subj_iclabel = pd.merge(
    f1_across_subj_iclabel, mean_f1_df, on="Prediction window [minutes]"
)

fig, ax = plt.subplots(figsize=(9, 5))
ax = plot_line_with_error_area(
    ax,
    f1_across_subj_psd_autocorr,
    "Prediction window [minutes]",
    "Brain F1 score - PSD_autocorr",
    "StdDev - PSD_autocorr",
)
ax = plot_line_with_error_area(
    ax,
    f1_across_subj_iclabel,
    "Prediction window [minutes]",
    "Brain F1 score - ICLabel",
    "StdDev - ICLabel",
    color="red",
)
# TODO: save these f1_* dataframes and load them to skip computation
# ax = plot_line_with_error_area(
#     ax,
#     f1_across_subj_bowav_sub,
#     "Prediction window [minutes]",
#     "Brain F1 score - BoWav",
#     "StdDev - BoWav",
#     color="green",
# )
ax.set_xscale("log")
ax.set_xticks([1, 2, 3, 5, 10, 30, 50], labels=[1, 2, 3, 5, 10, 30, 50])
ax.set_xlim(1, 50)
ax.set_xlabel("Prediction window [minutes]")
ax.set_ylabel("Mean Brain F1 score")
ax.set_title("Mean brain F1 score across subjects")
ax.legend()
ax.grid(True)

# %%
