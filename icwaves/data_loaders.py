import copy
from pathlib import Path
import logging

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

EXPERT_ANNOTATED_CLASSES = [1, 2, 3]  # brain, muscle, eye (Matlab indexing)

CLASS_LABELS = ['Brain', 'Muscle', 'Eye', 'Heart',
                'Line Noise', 'Channel Noise', 'Other']


def load_raw_train_set_per_class(args, rng):

    data_dir = args.path_to_centroid_assignments
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

    dict_dir = args.path_to_codebooks

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


def load_codebooks_wrapper(args):
    codebook_args = copy.deepcopy(args)
    codebook_args.minutes_per_ic = args.codebook_minutes_per_ic
    codebook_args.ics_per_subject = args.codebook_ics_per_subject
    codebooks = load_codebooks(codebook_args)
    return codebooks