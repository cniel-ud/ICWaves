# %%
import copy
import pickle
from pathlib import Path

import numpy as np
import scipy
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import default_rng
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from icwaves.data_loaders import load_codebooks_wrapper
from icwaves.feature_extractors.bowav import (
    build_bowav_from_centroid_assignments,
    build_or_load_centroid_assignments,
)
from icwaves.preprocessing import load_or_build_preprocessed_data
from icwaves.viz import plot_confusion_matrix
from icwaves.utils import build_results_file

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
    regularization_factor = [0.1, 1.0]
    expert_weight = [32.0]
    l1_ratio = [1.0]
    codebook_minutes_per_ic = 50.0
    codebook_ics_per_subject = 2
    bowav_norm = ["none"]
    subj_ids = train_subj_ids
    training_segment_length = [30.0, 300.0]
    validation_segment_length = 300


args = Args()
rng = default_rng(13)

img_dir = root.joinpath("img")

BOWAV_NORM_MAP = {
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
(
    windowed_ics,
    labels,
    srate,
    expert_label_mask,
    subj_ind,
) = load_or_build_preprocessed_data(args)
# %%
# Load codebooks
codebooks = load_codebooks_wrapper(args, srate)
n_centroids = codebooks[0].shape[0]

# Load or build centroid assignments
centroid_assignments = build_or_load_centroid_assignments(args, windowed_ics, codebooks)
# %%
input_or_output_aggregation_method = best_params["input_or_output_aggregation_method"]
n_validation_windows_per_segment = best_params["n_validation_windows_per_segment"]
n_training_windows_per_segment = best_params["n_training_windows_per_segment"]
bowav_norm = best_params["bowav_norm"]
if input_or_output_aggregation_method == "count_pooling":
    bowav_test = build_bowav_from_centroid_assignments(
        centroid_assignments, n_centroids, n_validation_windows_per_segment, bowav_norm
    )
else:
    bowav_test = build_bowav_from_centroid_assignments(
        centroid_assignments, n_centroids, n_training_windows_per_segment, bowav_norm
    )
# %%
n_segments_per_time_series = bowav_test.shape[1]
# vertically concatenate test BoWav vectors: (m, n, p) -> (m*n, p)
bowav_test = np.vstack(bowav_test)
y_pred = clf.predict(bowav_test)

# Maybe aggregate output
if input_or_output_aggregation_method == "majority_vote":
    if n_validation_windows_per_segment is None:
        n_windows_per_time_series = centroid_assignments.shape[2]
        n_validation_windows_per_segment = n_windows_per_time_series

    n_train_segments_per_validation_segment = (
        n_validation_windows_per_segment // n_training_windows_per_segment
    )
    y_pred = y_pred.reshape(-1, n_train_segments_per_validation_segment)
    y_pred = scipy.stats.mode(y_pred, axis=1)[0]
    n_segments_per_time_series = (
        n_segments_per_time_series // n_train_segments_per_validation_segment
    )

# expand labels and expert mask to match test BoWav vectors
y = np.repeat(labels, n_segments_per_time_series)
expert_label_mask = np.repeat(expert_label_mask, n_segments_per_time_series)

# %%
y_expert = y[expert_label_mask]
y_pred_expert = y_pred[expert_label_mask]

# %% BoWav confusion matrix on expert data
disp_labels_x = [
    "brain",
    "muscle",
    "eye",
    "Heart",
    "Line Noise",
    "Channel Noise",
    "Other",
]
disp_labels_y = ["brain", "muscle", "eye"]

cm = confusion_matrix(y_expert, y_pred_expert, labels=np.arange(7), normalize="true")
cm = cm[:3]
plot_confusion_matrix(
    cm,
    cmap="viridis",
    display_labels=[disp_labels_x, disp_labels_y],
    xticks_rotation="vertical",
)
fpath = img_dir.joinpath("IC_confmat_BoWav_expert.svg")
plt.savefig(fpath, bbox_inches="tight", pad_inches=0)
# %% BoWav confusion matrix on ICLabel-ed data
y_noisy = y[~expert_label_mask]
y_pred_noisy = y_pred[~expert_label_mask]
cm = confusion_matrix(y_noisy, y_pred_noisy, labels=np.arange(7), normalize="true")
plot_confusion_matrix(
    cm,
    cmap="viridis",
    display_labels=[disp_labels_x, disp_labels_x],
    xticks_rotation="vertical",
    ylabel="ICLabel",
)
fpath = img_dir.joinpath("IC_confmat_BoWav_non-expert.svg")
plt.savefig(fpath, bbox_inches="tight", pad_inches=0)

# %% ICLabel confusion matrix on expert data
noisy_labels_vec = np.argmax(noisy_labels, axis=1)
ICLabel_expert = noisy_labels_vec[expert_label_mask]
cm = confusion_matrix(y_expert, ICLabel_expert, labels=np.arange(7), normalize="true")
cm = cm[:3]
plot_confusion_matrix(
    cm,
    cmap="viridis",
    display_labels=[disp_labels_x, disp_labels_y],
    xticks_rotation="vertical",
)
fpath = img_dir.joinpath("IC_confmat_ICLabel_expert.svg")
plt.savefig(fpath, bbox_inches="tight", pad_inches=0)

# %%
clf_report_bowav = classification_report(
    y_expert,
    y_pred_expert,
    labels=np.arange(0, 3),
    target_names=["brain", "muscle", "eye"],
    output_dict=True,
)
df_bowav = pd.DataFrame(clf_report_bowav).transpose()
df_bowav
# %%
clf_report_ICLabel = classification_report(
    y_expert,
    ICLabel_expert,
    labels=np.arange(0, 3),
    target_names=["brain", "muscle", "eye"],
    output_dict=True,
)
df_ICLabel = pd.DataFrame(clf_report_ICLabel).transpose()
df_ICLabel
# %%
score_BoWav_expert = balanced_accuracy_score(y_expert, y_pred_expert)
score_ICLabel_expert = balanced_accuracy_score(y_expert, ICLabel_expert)
print(score_BoWav_expert, score_ICLabel_expert)

# %% TODO: Find the code used to create the codebook plots in the dissertation. If it is already in
# plot_codebooks.ipynb, then delete the code below:
# def show_codebook(codebook):

#     k, P = codebook.shape
#     n_rows = np.ceil(np.sqrt(k)).astype(int)
#     n_cols = k // n_rows
#     fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
#     ax = ax.flatten()
#     for i in range(n_rows*n_cols):
#         if i < k:
#             ax[i].plot(codebook[i])
#         ax[i].axis('off')

#     return fig, ax
# # %%
# for i, cb in enumerate(codebooks):
#     fig, ax = show_codebook(cb)
#     fig.suptitle(IC_classes[i])
#     plt.tight_layout()
# # %%
# args = codebook_args
# dict_dir = Path(args.root, 'results/dictionaries')
# pat = (
#     f'sikmeans_P-{args.centroid_len}_k-{args.num_clusters}'
#     f'_class-*_minutesPerIC-{args.minutes_per_ic}'
#     f'_icsPerSubj-{args.ics_per_subject}.npz'
# )
# file_list = list(dict_dir.glob(pat))
# expr = r'.+_class-(?P<label>\d).*npz'
# p = re.compile(expr)
# label_sort = [int(p.search(str(file))['label']) for file in file_list]
# %%
