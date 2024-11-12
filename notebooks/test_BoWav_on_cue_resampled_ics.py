# %%
"""
The "emotion_study" dataset has EEG data from 34 subjects. Subjects 8 to 35
(subject 22 is missing) were used to train a multiclass classifier. The 'cue'
dataset has 12 subjects. This script computes the test results on the 12 'cue'
subjects, using the classifier on the 27 'emotion_study' subjects. Since the
sampling rates in 'emotion_study' and cue are 256 Hz and 500 Hz, respectively,
here we use 'cue' ICs that were resampled from 500 Hz to 256 Hz.

With the cue ICs resampled, we compute the BoWav feature using the
'emotion_study' dictionaries and the 'cue' EEG independent components, to then make
the prediction using the classifier trained on the 'emotion_study' dataset, and finally
compare with the 'cue' ground truth labels.
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
"""
Get trained classifier and dictionaries
"""
root = Path(__file__).parent.parent
print(f"Root folder is {root}")

emotion_train_subj_ids = list(range(8, 36))
emotion_train_subj_ids.remove(22)  # subject 22 is not present
cue_subj_ids = list(range(1, 8))


class Args:
    path_to_raw_data = root.joinpath("data/cue/resampled_raw_data_and_IC_labels")
    path_to_results = root.joinpath("results/emotion_study/classifier")
    path_to_preprocessed_data = root.joinpath("data/cue/resampled_preprocessed_data")
    path_to_codebooks = root.joinpath("results/emotion_study/dictionaries")
    path_to_centroid_assignments = root.joinpath(
        "data/cue/resampled_centroid_assignments"
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
    subj_ids = emotion_train_subj_ids
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

best_index = results["rank_test_scores"].argmin()
best_score = results[f"mean_test_scores"][best_index]
best_params = copy.deepcopy(results["params"][best_index])
input_or_output_aggregation_method = best_params["input_or_output_aggregation_method"]
training_segment_length = best_params["training_segment_length"]
# %% Load or build preprocessed data
args.subj_ids = cue_subj_ids
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
validation_segment_length_arr = validation_segment_length_arr / args.window_length
validation_segment_length_arr = validation_segment_length_arr.astype(int)

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
for validation_segment_length in validation_segment_length_arr:
    print(
        f"Computing F1 score after aggregating {validation_segment_length * args.window_length} seconds"
    )
    for subj_id in cue_subj_ids:
        print(f"Subject {subj_id}")
        subj_mask = subj_ind == subj_id

        brain_f1_score = compute_brain_F1_score_per_subject(
            clf,
            centroid_assignments,
            labels,
            expert_label_mask,
            input_or_output_aggregation_method,
            feature_extractor,
            validation_segment_length,
            training_segment_length,
            subj_mask,
        )

        f1_scores_df.loc[len(f1_scores_df)] = {
            "Prediction window [minutes]": validation_segment_length
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
validation_segment_length_arr = validation_segment_length_arr * args.window_length
columns = [
    "Prediction window [minutes]",
    "Subject ID",
    "Brain F1 score",
    "Number of ICs",
]
f1_scores_ICLabel_df = pd.DataFrame(columns=columns)
cue_dir = Path("../data/cue/")
fnames = [f"subj-{i:02}.mat" for i in args.subj_ids]
for test_segment_len in validation_segment_length_arr:
    subdir = cue_dir.joinpath(f"IC_labels_at_{test_segment_len}_seconds")
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
