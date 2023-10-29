import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

EXPERT_ANNOTATED_CLASSES = [1, 2, 3]  # brain, muscle, eye (Matlab indexing)

CLASS_LABELS = ['Brain', 'Muscle', 'Eye', 'Heart',
                'Line Noise', 'Channel Noise', 'Other']

def load_raw_set(args, rng):

    data_dir = Path(args.root, 'data/emotion_study/raw_data_and_IC_labels')
    fnames = [f"subj-{i}.mat" for i in args.subj_ids]
    file_list = [data_dir.joinpath(f) for f in fnames]

    n_ics_per_subj = []
    for file in file_list:
        with file.open('rb') as f:
            matdict = loadmat(f, variable_names=['labels', 'srate'])
            labels = matdict['labels']
            srate = matdict['srate'] # assumes all subjects have the same sampling rate
            srate = srate.item(0) # `srate.shape=(1,1)`. This extracts the number.
            n_ics_per_subj.append(labels.shape[0])

    n_ics = np.sum(n_ics_per_subj)
    minutes_per_window = (args.window_len/srate/60)
    n_win_per_ic = np.ceil(args.minutes_per_ic / minutes_per_window).astype(int)
    n_win_per_segment = args.n_windows_per_segment
    if n_win_per_segment > 0 and n_win_per_segment < n_win_per_ic:
        n_segments = n_win_per_ic // n_win_per_segment
    else:
        n_segments = 1
        n_win_per_segment = n_win_per_ic
    segment_len = n_win_per_segment * args.window_len
    window_len = args.window_len

    # NOTE: float32. ICs were saved in matlab as single.
    X = np.zeros((n_ics, n_segments, n_win_per_segment, window_len), dtype=np.float32)
    y = -1 * np.ones((n_ics, n_segments), dtype=int)

    cum_ic_ind = 0
    expert_label_mask_ar = np.full((n_ics, n_segments), False)
    subj_ind = np.zeros((n_ics, n_segments), dtype=int)
    # 7 ICLabel classes
    noisy_labels_ar = np.zeros((n_ics, 7), dtype=np.float32)
    for file, subjID in tqdm(zip(file_list, args.subj_ids)):
        with file.open('rb') as f:
            matdict = loadmat(f)
            data = matdict['data']
            icaweights = matdict['icaweights']
            icasphere = matdict['icasphere']
            noisy_labels = matdict['noisy_labels']
            expert_label_mask = matdict['expert_label_mask']
            # -1: Let class labels start at 0 in python
            labels = matdict['labels'] - 1

        expert_label_mask = expert_label_mask.astype(bool)
        icaact = icaweights @ icasphere @ data

        expert_label_mask = expert_label_mask.astype(bool)
        for ic_ind, ic in enumerate(icaact):
            if n_segments == 1:
                time_idx = np.arange(0, ic.size-window_len+1, window_len)
                time_idx = rng.choice(time_idx, size=n_win_per_ic, replace=False)
                time_idx = time_idx[:, None] + np.arange(window_len)[None, :]
            else:
                time_idx = np.arange(0, ic.size-segment_len+1, segment_len)
                time_idx = rng.choice(time_idx, size=n_segments, replace=False)
                time_idx = time_idx[:, None] + np.arange(segment_len)[None, :]
            segmented_ic = ic[time_idx] # a 2D array
            X[cum_ic_ind] = segmented_ic.reshape((n_segments, n_win_per_segment, window_len))
            y[cum_ic_ind] = labels[ic_ind]
            noisy_labels_ar[cum_ic_ind] = noisy_labels[ic_ind]
            expert_label_mask_ar[cum_ic_ind] = expert_label_mask[ic_ind]
            subj_ind[cum_ic_ind] = subjID
            cum_ic_ind += 1

    X = X.reshape((n_ics * n_segments, n_win_per_segment, window_len))
    y = y.reshape(n_ics * n_segments)
    expert_label_mask_ar = expert_label_mask_ar.reshape(n_ics * n_segments)
    subj_ind = subj_ind.reshape(n_ics * n_segments)

    return X, y, expert_label_mask_ar, subj_ind, noisy_labels_ar


def load_raw_train_set_per_class(args, rng):

    data_dir = Path(args.root, 'data/emotion_study/icact_iclabel')
    file_list = list(data_dir.glob(f'train_subj-*.mat'))

    ic_ind_per_subj = []
    subj_with_no_ic = []
    for i_subj, file in enumerate(file_list):
        with file.open('rb') as f:
            matdict = loadmat(f, variable_names=[
                              'expert_labels', 'noisy_labels'])
            expert_labels = matdict['expert_labels']
            noisy_labels = matdict['noisy_labels']

        if args.class_label in EXPERT_ANNOTATED_CLASSES:
            ic_ind = (expert_labels == args.class_label).nonzero()[0]
        else:
            winner_class = np.argmax(noisy_labels, axis=1)
            winner_class = winner_class + 1  # python to matlab indexing base
            ic_ind = (winner_class == args.class_label).nonzero()[0]

        if ic_ind.size > 0: # subject has IC class
            ic_ind_per_subj.append(ic_ind)
        else:
            subj_with_no_ic.append(i_subj)

    file_list = [file for i, file in \
        enumerate(file_list) if i not in subj_with_no_ic]

    n_subj = len(ic_ind_per_subj)
    n_ics_per_subj = np.array(list(map(lambda x: x.size, ic_ind_per_subj)))
    subj_with_ic_excess = (n_ics_per_subj > args.ics_per_subject).nonzero()[0]
    n_ics_per_subj[subj_with_ic_excess] = args.ics_per_subject
    n_ics = np.sum(n_ics_per_subj)

    for i_subj in subj_with_ic_excess:
        ic_ind_per_subj[i_subj] = rng.choice(
            ic_ind_per_subj[i_subj], size=args.ics_per_subject, replace=False)

    icaact_list = [None] * n_subj
    for i_subj, file in tqdm(enumerate(file_list)):
        with file.open('rb') as f:
            matdict = loadmat(f, variable_names='icaact')
            icaact = matdict['icaact']

        icaact_list[i_subj] = icaact[ic_ind_per_subj[i_subj]]

    # ICs from different subjects have different lenths, so we don't
    # concatenate into a single array
    ic_lenght_per_subj = np.array(list(map(lambda x: x.shape[1], icaact_list)))
    max_n_win_per_subj = ic_lenght_per_subj // args.window_len
    max_minutes_per_ic_per_subj = (max_n_win_per_subj *
                                   args.window_len) / args.srate / 60
    min_max_minutes_per_ic = np.min(max_minutes_per_ic_per_subj)

    take_all = True
    if (
        args.minutes_per_ic is not None and
        args.minutes_per_ic < min_max_minutes_per_ic
    ):
        minutes_per_window = (args.window_len/args.srate/60)
        n_win_per_ic = np.ceil(args.minutes_per_ic /
                            minutes_per_window).astype(int)
        take_all = False
    else:
        n_win_per_ic = ic_lenght_per_subj // args.window_len

    tot_win = (n_win_per_ic * n_ics_per_subj).sum()
    tot_hrs = tot_win * args.window_len / args.srate / 3600

    print(f"Training ICs for '{CLASS_LABELS[args.class_label-1]}': {n_ics}")
    print(f"Number of training hours: {tot_hrs:.2f}")

    X = np.zeros((tot_win, args.window_len), dtype=icaact_list[0].dtype)
    win_start = 0
    for i_subj, ics in tqdm(enumerate(icaact_list)):
        n_win = n_win_per_ic[i_subj] if take_all else n_win_per_ic
        for ic in ics:
            time_idx = np.arange(0, ic.size-args.window_len+1, args.window_len)
            time_idx = rng.choice(time_idx, size=n_win, replace=False)
            time_idx = time_idx[:, None] + np.arange(args.window_len)[None, :]
            X[win_start:win_start+n_win] = ic[time_idx]
            win_start += n_win

    return X


def load_codebooks(args):

    dict_dir = Path(args.root, 'results/dictionaries')

    n_codebooks = 7
    codebooks = np.zeros((n_codebooks, args.num_clusters,
                        args.centroid_len), dtype=np.float32)

    for i_class in range(n_codebooks):
        fname = (
            f'sikmeans_P-{args.centroid_len}_k-{args.num_clusters}'
            f'_class-{i_class+1}_minutesPerIC-{args.minutes_per_ic}'
            f'_icsPerSubj-{args.ics_per_subject}.npz'
        )
        fpath = dict_dir.joinpath(fname)
        with np.load(fpath) as data:
            codebooks[i_class] = data['centroids']

    return codebooks