#%%
import copy
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.random import default_rng
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix)

from icwaves.data_loaders import load_codebooks, load_raw_set
from icwaves.feature_extractors.bowav import bag_of_waves
from icwaves.viz import plot_confusion_matrix

#%%
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 12
})
img_dir = Path('/home/cmendoza/MEGA/Research/dissertation/LaTeX/img/')
#%%

class Args:
    root = '..'
    window_len = 384
    minutes_per_ic = 50
    num_clusters = 128
    centroid_len = 256
    srate = 256
    penalty = 'elasticnet'
    regularization_factor = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    expert_weight = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    l1_ratio = [0, 0.2, 0.4, 0.6, 0.8, 1]
    codebook_minutes_per_ic = 50.0
    codebook_ics_per_subject = 2


args = Args()
rng = default_rng(13)
#%%
C_str = '_'.join([str(i) for i in args.regularization_factor])
ew_str = '_'.join([str(i) for i in args.expert_weight])
l1_ratio_str = '_'.join([str(i) for i in args.l1_ratio])
fname = (
    f'clf-lr_penalty-{args.penalty}_solver-saga_C-{C_str}'
    f'_l1_ratio-{l1_ratio_str}'
    f'_expert_weight-{ew_str}'
    f'_cbookMinPerIC-{args.codebook_minutes_per_ic}'
    f'_cbookICsPerSubj-{args.codebook_ics_per_subject}.pickle'
)
clf_path = Path(args.root, 'results/classifier', fname)
with clf_path.open('rb') as f:
    results = pickle.load(f)
clf = results['best_estimator']['clf']
# %%
best_index = results["rank_test_scores"].argmin()
best_score = results[f"mean_test_scores"][best_index]
best_params = copy.deepcopy(results["params"][best_index])

# %% Load/generate data
fname = (
    f'test_data_k-{args.num_clusters}_P-{args.centroid_len}'
    f'_winlen-{args.window_len}_minPerIC-{args.minutes_per_ic}'
    f'_cbookMinPerIc-{args.codebook_minutes_per_ic}'
    f'_cbookICsPerSubj-{args.codebook_ics_per_subject}.npz'
)
data_file = Path(args.root, 'data/ds003004/BoWav', fname)
if data_file.is_file():
    with np.load(data_file) as data:
        X = data['X']
        y = data['y']
        noisy_labels = data['noisy_labels']
        expert_label_mask = data['expert_label_mask']
        subj_ind = data['subj_ind']
else:
    raw_ics, y, expert_label_mask, \
        subj_ind, noisy_labels = load_raw_set(args, rng, train=False)
    codebook_args = copy.deepcopy(args)
    codebook_args.minutes_per_ic = args.codebook_minutes_per_ic
    codebook_args.ics_per_subject = args.codebook_ics_per_subject
    codebooks = load_codebooks(codebook_args)
    X = bag_of_waves(raw_ics, codebooks)
    with data_file.open('wb') as f:
        np.savez(
            f, raw_ics=raw_ics, X=X, y=y, noisy_labels=noisy_labels,
            expert_label_mask=expert_label_mask, subj_ind=subj_ind)

# %%
y_pred = clf.predict(X)

#%%
y_expert = y[expert_label_mask]
y_pred_expert = y_pred[expert_label_mask]

#%% BoWav confusion matrix on expert data
disp_labels_x = ['brain', 'muscle', 'eye', 'Heart',
                 'Line Noise', 'Channel Noise', 'Other']
disp_labels_y = ['brain', 'muscle', 'eye']

cm = confusion_matrix(
    y_expert,
    y_pred_expert,
    labels=np.arange(7),
    normalize='true'
)
cm = cm[:3]
plot_confusion_matrix(
    cm,
    cmap='viridis',
    display_labels=[disp_labels_x, disp_labels_y],
    xticks_rotation='vertical'
)
fpath = img_dir.joinpath('IC_confmat_BoWav_expert.svg')
plt.savefig(fpath, bbox_inches='tight', pad_inches=0)
#%% BoWav confusion matrix on ICLabel-ed data
y_noisy = y[~expert_label_mask]
y_pred_noisy = y_pred[~expert_label_mask]
cm = confusion_matrix(
    y_noisy,
    y_pred_noisy,
    labels=np.arange(7),
    normalize='true'
)
plot_confusion_matrix(
    cm,
    cmap='viridis',
    display_labels=[disp_labels_x, disp_labels_x],
    xticks_rotation='vertical',
    ylabel='ICLabel'
)
fpath = img_dir.joinpath('IC_confmat_BoWav_non-expert.svg')
plt.savefig(fpath, bbox_inches='tight', pad_inches=0)

#%% ICLabel confusion matrix on expert data
noisy_labels_vec = np.argmax(noisy_labels, axis=1)
ICLabel_expert = noisy_labels_vec[expert_label_mask]
cm = confusion_matrix(
    y_expert,
    ICLabel_expert,
    labels=np.arange(7),
    normalize='true'
)
cm = cm[:3]
plot_confusion_matrix(
    cm,
    cmap='viridis',
    display_labels=[disp_labels_x, disp_labels_y],
    xticks_rotation='vertical'
)
fpath = img_dir.joinpath('IC_confmat_ICLabel_expert.svg')
plt.savefig(fpath, bbox_inches='tight', pad_inches=0)

#%%
clf_report_bowav = classification_report(y_expert, y_pred_expert, labels=np.arange(0,3), target_names=['brain', 'muscle', 'eye'], output_dict=True)
df_bowav = pd.DataFrame(clf_report_bowav).transpose()
df_bowav
#%%
clf_report_ICLabel = classification_report(y_expert, ICLabel_expert, labels=np.arange(0,3), target_names=['brain', 'muscle', 'eye'], output_dict=True)
df_ICLabel = pd.DataFrame(clf_report_ICLabel).transpose()
df_ICLabel
#%%
score_BoWav_expert = balanced_accuracy_score(y_expert, y_pred_expert)
score_ICLabel_expert = balanced_accuracy_score(y_expert, ICLabel_expert)
print(score_BoWav_expert, score_ICLabel_expert)

#%%
recall = np.zeros((3, 2))
for i in range(3):
    true_ind = y_expert == i
    n_true = np.count_nonzero(true_ind)
    recall[i, 0] = np.count_nonzero(y_pred_expert[true_ind]==i)/n_true
    recall[i, 1] = np.count_nonzero(noisy_labels_vec[true_ind] == i)/n_true
df = pd.DataFrame(recall, columns=['BoWav', 'ICLabel'], index=['brain', 'muscle', 'eye'])
df
#%%
precision = np.zeros((3, 2))
for i in range(3):
    true_ind = y_expert == i
    tp = np.count_nonzero(y_pred_expert[true_ind] == i)
    fp = np.count_nonzero(y_pred_expert[~true_ind] == i)
    precision[i, 0] = tp/(tp+fp)
    tp = np.count_nonzero(noisy_labels_vec[true_ind] == i)
    fp = np.count_nonzero(noisy_labels_vec[~true_ind] == i)
    precision[i, 1] = tp/(tp+fp)
df = pd.DataFrame(precision, columns=['BoWav', 'ICLabel'], index=[
                  'brain', 'muscle', 'eye'])
df

# %%
def show_codebook(codebook):

    k, P = codebook.shape
    n_rows = np.ceil(np.sqrt(k)).astype(int)
    n_cols = k // n_rows
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    ax = ax.flatten()
    for i in range(n_rows*n_cols):
        if i < k:
            ax[i].plot(codebook[i])
        ax[i].axis('off')

    return fig, ax
# %%
for i, cb in enumerate(codebooks):
    fig, ax = show_codebook(cb)
    fig.suptitle(IC_classes[i])
    plt.tight_layout()
# %%
args = codebook_args
dict_dir = Path(args.root, 'results/dictionaries')
pat = (
    f'sikmeans_P-{args.centroid_len}_k-{args.num_clusters}'
    f'_class-*_minutesPerIC-{args.minutes_per_ic}'
    f'_icsPerSubj-{args.ics_per_subject}.npz'
)
file_list = list(dict_dir.glob(pat))
expr = r'.+_class-(?P<label>\d).*npz'
p = re.compile(expr)
label_sort = [int(p.search(str(file))['label']) for file in file_list]
# %%
