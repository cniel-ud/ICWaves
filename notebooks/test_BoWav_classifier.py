# %%
"""
The "emotion_study" dataset has EEG data from 34 subjects. Subjects 8 to 35
(subject 22 is missing) were used to train a multiclass classifier, and subject
1 to 7 were used to test it. This script computes the test results on those 7
subjects, using the trained classifier.
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

from icwaves.feature_extractors.bowav import (
    build_bowav_from_centroid_assignments,
    build_or_load_centroid_assignments_and_labels,
)
from icwaves.viz import plot_line_with_error_area
from icwaves.file_utils import build_results_file
from icwaves.utils import compute_brain_F1_score_per_subject

plt.rcParams.update({"text.usetex": True, "font.size": 12})
# %%
root = Path(__file__).parent.parent
print(f"Root folder is {root}")

train_subj_ids = list(range(8, 36))
train_subj_ids.remove(22)  # subject 22 is not present
test_subj_ids = list(range(1, 8))


class Args:
    path_to_raw_data = root.joinpath("data/emotion_study/raw_data_and_IC_labels")
    path_to_results = root.joinpath("results/emotion_study/classifier")
    path_to_preprocessed_data = root.joinpath("data/emotion_study/preprocessed_data")
    path_to_codebooks = root.joinpath("results/emotion_study/dictionaries")
    path_to_centroid_assignments = root.joinpath(
        "data/emotion_study/centroid_assignments"
    )
    window_length = 1.5
    minutes_per_ic = 50.0
    num_clusters = 128
    centroid_length = 1.0
    penalty = "elasticnet"
    regularization_factor = [0.01, 0.1, 1.0]
    expert_weight = [float(x) for x in [1, 16, 32, 64]]
    l1_ratio = [float(x) for x in [0.1, 0.5, 1.0]]
    codebook_minutes_per_ic = 50.0
    codebook_ics_per_subject = 2
    tf_idf_norm = ["none", "l1", "l2"]
    subj_ids = train_subj_ids
    training_segment_length = [float(x) for x in [10, 30, 60, 120, 300]]
    validation_segment_length = -1


args = Args()
rng = default_rng(13)

img_dir = root.joinpath("img")

TF_IDF_NORM_MAP = {
    "none": None,
    "l_1": 1,
    "l_2": 2,
    "l_inf": np.inf,
}

results_folder = Path(args.path_to_results)
results_folder.mkdir(exist_ok=True, parents=True)
results_file = build_results_file(args=args)
results_file = results_folder.joinpath(results_file)
clf_path = args.path_to_results.joinpath(results_file)
with clf_path.open("rb") as f:
    results = pickle.load(f)
clf = results["best_estimator"]["clf"]
# %%
best_index = results["rank_test_scores"].argmin()
best_score = results[f"mean_test_scores"][best_index]
best_params = copy.deepcopy(results["params"][best_index])

# %% Load or build preprocessed data
args.subj_ids = test_subj_ids
centroid_assignments, labels, expert_label_mask, subj_ind, noisy_labels, n_centroids = (
    build_or_load_centroid_assignments_and_labels(args)
)
# %%
# Predict on increments of 10 seconds, up to the first 50 minutes
# first minute: 10 seconds
# 1-5 min: 1 minutes
# 5-end: 5 minutes
validation_segment_length_arr = np.r_[
    np.arange(10, 60 + 1, 10),
    np.arange(2 * 60, 5 * 60 + 1, 60),
    np.arange(10 * 60, 50 * 60 + 1, 5 * 60),
]
n_validation_windows_per_segment_arr = (
    validation_segment_length_arr / args.window_length
)
n_validation_windows_per_segment_arr = n_validation_windows_per_segment_arr.astype(int)
input_or_output_aggregation_method = best_params["input_or_output_aggregation_method"]
n_training_windows_per_segment = best_params["n_training_windows_per_segment"]

# %%
# average F1 per subject
# Jack knife: when computing mean F1 for each subject, compute mean F1 in the
# hold out to put error bars on the mean F1 across all ICs
# get variance across subjects
# TODO: do this for ICLabel and PSD+autocorr
columns = [
    "Prediction window [minutes]",
    "Subject ID",
    "Brain F1 score",
    "Number of ICs",
]
f1_scores_df = pd.DataFrame(columns=columns)
# build_bowav_from_centroid_assignments has args (centroid_assignments, n_centroids, n_windows_per_segment)
# `feature_extractor` has args (time_series, segment_len)
feature_extractor = (
    lambda time_series, segment_len: build_bowav_from_centroid_assignments(
        time_series, n_centroids, segment_len
    )
)
for n_validation_windows_per_segment in n_validation_windows_per_segment_arr:
    print(
        f"Computing F1 score after aggregating {n_validation_windows_per_segment * args.window_length} seconds"
    )
    for subj_id in test_subj_ids:
        print(f"Subject {subj_id}")
        subj_mask = subj_ind == subj_id

        brain_f1_score = compute_brain_F1_score_per_subject(
            clf,
            centroid_assignments,
            labels,
            expert_label_mask,
            input_or_output_aggregation_method,
            feature_extractor,
            n_validation_windows_per_segment,
            n_training_windows_per_segment,
            subj_mask,
        )

        f1_scores_df.loc[len(f1_scores_df)] = {
            "Prediction window [minutes]": n_validation_windows_per_segment
            * args.window_length
            / 60,
            "Subject ID": subj_id,
            "Brain F1 score": brain_f1_score,
            "Number of ICs": subj_mask.sum(),
        }
# %%
f1_across_subj_bowav = f1_scores_df.groupby("Prediction window [minutes]")[
    "Brain F1 score"
].std()
f1_across_subj_bowav = f1_across_subj_bowav.rename("StdDev").reset_index()
mean_f1_df = (
    f1_scores_df.groupby("Prediction window [minutes]")["Brain F1 score"]
    .mean()
    .rename("Brain F1 score")
    .reset_index()
)
f1_across_subj_bowav = pd.merge(
    f1_across_subj_bowav, mean_f1_df, on="Prediction window [minutes]"
)

f1_across_subj_bowav = f1_across_subj_bowav.rename(
    columns={"StdDev": "StdDev - BoWav", "Brain F1 score": "Brain F1 score - BoWav"}
)
fig, ax = plt.subplots(figsize=(9, 5))
ax = plot_line_with_error_area(
    ax,
    f1_across_subj_bowav,
    "Prediction window [minutes]",
    "Brain F1 score - BoWav",
    "StdDev - BoWav",
)
ax.set_xscale("log")
ax.set_xticks(
    [0.15, 0.5, 1, 2, 3, 5, 10, 30, 50], labels=[0.15, 0.5, 1, 2, 3, 5, 10, 30, 50]
)
ax.set_xlim(0.15, 50)
ax.set_xlabel("Prediction window [minutes]")
ax.set_ylabel("Mean Brain F1 score")
ax.set_title("Mean brain F1 score across subjects")
ax.legend()
ax.grid(True)
# %%
validation_segment_length_arr = (
    n_validation_windows_per_segment_arr * args.window_length
)
columns = [
    "Prediction window [minutes]",
    "Subject ID",
    "Brain F1 score",
    "Number of ICs",
]
f1_scores_ICLabel_df = pd.DataFrame(columns=columns)
emotion_study_dir = Path("../data/emotion_study/")
fnames = [f"subj-{i:02}.mat" for i in args.subj_ids]
for test_segment_len in validation_segment_length_arr:
    subdir = emotion_study_dir.joinpath(f"IC_labels_at_{test_segment_len}_seconds")
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
    f1_across_subj_bowav,
    "Prediction window [minutes]",
    "Brain F1 score - BoWav",
    "StdDev - BoWav",
)
ax = plot_line_with_error_area(
    ax,
    f1_across_subj_iclabel,
    "Prediction window [minutes]",
    "Brain F1 score - ICLabel",
    "StdDev - ICLabel",
    color="red",
)
ax.set_xscale("log")
ax.set_xticks(
    [0.15, 0.5, 1, 2, 3, 5, 10, 30, 50], labels=[0.15, 0.5, 1, 2, 3, 5, 10, 30, 50]
)
ax.set_xlim(0.15, 50)
ax.set_xlabel("Prediction window [minutes]")
ax.set_ylabel("Mean Brain F1 score")
ax.set_title("Mean brain F1 score across subjects")
ax.legend()
ax.grid(True)

# %%
# # %% BoWav confusion matrix on expert data
# disp_labels_x = [
#     "brain",
#     "muscle",
#     "eye",
#     "Heart",
#     "Line Noise",
#     "Channel Noise",
#     "Other",
# ]
# disp_labels_y = ["brain", "muscle", "eye"]

# cm = confusion_matrix(y_expert, y_pred_expert, labels=np.arange(7), normalize="true")
# cm = cm[:3]
# plot_confusion_matrix(
#     cm,
#     cmap="viridis",
#     display_labels=[disp_labels_x, disp_labels_y],
#     xticks_rotation="vertical",
# )
# fpath = img_dir.joinpath("IC_confmat_BoWav_expert.svg")
# plt.savefig(fpath, bbox_inches="tight", pad_inches=0)
# # %% BoWav confusion matrix on ICLabel-ed data
# y_noisy = y[~expert_label_mask]
# y_pred_noisy = y_pred[~expert_label_mask]
# cm = confusion_matrix(y_noisy, y_pred_noisy, labels=np.arange(7), normalize="true")
# plot_confusion_matrix(
#     cm,
#     cmap="viridis",
#     display_labels=[disp_labels_x, disp_labels_x],
#     xticks_rotation="vertical",
#     ylabel="ICLabel",
# )
# fpath = img_dir.joinpath("IC_confmat_BoWav_non-expert.svg")
# plt.savefig(fpath, bbox_inches="tight", pad_inches=0)

# # %% ICLabel confusion matrix on expert data
# noisy_labels_vec = np.argmax(noisy_labels, axis=1)
# ICLabel_expert = noisy_labels_vec[expert_label_mask]
# cm = confusion_matrix(y_expert, ICLabel_expert, labels=np.arange(7), normalize="true")
# cm = cm[:3]
# plot_confusion_matrix(
#     cm,
#     cmap="viridis",
#     display_labels=[disp_labels_x, disp_labels_y],
#     xticks_rotation="vertical",
# )
# fpath = img_dir.joinpath("IC_confmat_ICLabel_expert.svg")
# plt.savefig(fpath, bbox_inches="tight", pad_inches=0)

# # %%
# clf_report_bowav = classification_report(
#     y_expert,
#     y_pred_expert,
#     labels=np.arange(0, 3),
#     target_names=["brain", "muscle", "eye"],
#     output_dict=True,
# )
# df_bowav = pd.DataFrame(clf_report_bowav).transpose()
# df_bowav
# # %%
# clf_report_ICLabel = classification_report(
#     y_expert,
#     ICLabel_expert,
#     labels=np.arange(0, 3),
#     target_names=["brain", "muscle", "eye"],
#     output_dict=True,
# )
# df_ICLabel = pd.DataFrame(clf_report_ICLabel).transpose()
# df_ICLabel
# %%
# result = (
#     f1_scores_df.groupby("Prediction window [minutes]")
#     .apply(jackknife_stddev)
#     .reset_index()
# )
# result = result.rename(
#     columns={"Jackknife StdDev": "StdDev", "Jackknife Mean": "Brain F1 score"}
# )
# result
# # %%
# fig, ax = plt.subplots(figsize=(9, 5))
# ax = plot_line_with_error_area(
#     ax, result, "Prediction window [minutes]", "Brain F1 score", "StdDev"
# )
# ax.plot(
#     f1_across_subj_bowav["Prediction window [minutes]"],
#     f1_across_subj_bowav["Brain F1 score"],
#     "k--",
#     label="Across subject mean",
# )
# ax.set_xscale("log")
# ax.set_xticks(
#     [0.15, 0.5, 1, 2, 3, 5, 10, 30, 50], labels=[0.15, 0.5, 1, 2, 3, 5, 10, 30, 50]
# )
# ax.set_xlabel("Prediction window [minutes]")
# ax.set_ylabel("Mean brain F1 score")
# ax.set_xlim(0.15, 50)
# ax.set_title("Jackknife estimates of mean and standard deviation of Brain F1 score")
# ax.legend()
# ax.grid(True)
