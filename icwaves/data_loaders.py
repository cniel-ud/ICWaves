import re
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def load_raw_train_set(args, rng):

    data_dir = Path(args.root, 'data/ds003004/icact_iclabel')
    file_list = list(data_dir.glob(f'train_subj-*.mat'))

    n_ics_per_subj = []
    for file in file_list:
        with file.open('rb') as f:
            matdict = loadmat(f, variable_names='expert_labels')
            expert_labels = matdict['expert_labels']
            n_ics_per_subj.append(expert_labels.shape[0])

    n_ics = np.sum(n_ics_per_subj)
    minutes_per_window = (args.window_len/args.srate/60)
    n_win_per_ic = np.ceil(args.minutes_per_ic / minutes_per_window).astype(int)

    # NOTE: float32. ICs were saved in matlab as single.
    X = np.zeros((n_ics, n_win_per_ic, args.window_len), dtype=np.float32)
    y = -1 * np.ones(n_ics, dtype=int)

    cum_ic_ind = 0
    expert_label_mask = np.full(n_ics, False)
    subj_ind_ar = np.zeros(n_ics, dtype=int)
    p = re.compile(r'.+train_subj-(?P<subjID>\d{2}).mat')
    for file in file_list:
        with file.open('rb') as f:
            matdict = loadmat(f)
            expert_labels = matdict['expert_labels']
            icaact = matdict['icaact']
            noisy_labels = matdict['noisy_labels']

        m = p.search(str(file))
        subjID = int(m.group('subjID'))

        ics_with_expert_label = (expert_labels > 0).nonzero()[0]
        for ic_ind, ic in enumerate(icaact):
            time_idx = np.arange(0, ic.size-args.window_len+1, args.window_len)
            time_idx = rng.choice(time_idx, size=n_win_per_ic, replace=False)
            time_idx = time_idx[:, None] + np.arange(args.window_len)[None, :]
            X[cum_ic_ind] = ic[time_idx]

            subj_ind_ar[cum_ic_ind] = subjID

            if ic_ind in ics_with_expert_label:
                # -1: Let class labels start at 0 in python
                y[cum_ic_ind] = expert_labels[ic_ind] - 1
                expert_label_mask[cum_ic_ind] = True
            else:
                noisy_label = np.argmax(noisy_labels[ic_ind])
                y[cum_ic_ind] = noisy_label
            cum_ic_ind += 1

    return X, y, expert_label_mask, subj_ind_ar


def load_codebooks(args):
    pat = f'sikmeans_P-{args.centroid_len}_k-{args.num_clusters}*.npz'
    dict_dir = Path(args.root, 'results/dictionaries')
    file_list = list(dict_dir.glob(pat))

    n_codebooks = len(file_list)
    codebooks = np.zeros((n_codebooks, args.num_clusters,
                        args.centroid_len), dtype=np.float32)
    for i_codebook, file in enumerate(file_list):
        with np.load(file) as data:
            codebooks[i_codebook] = data['centroids']

    return codebooks