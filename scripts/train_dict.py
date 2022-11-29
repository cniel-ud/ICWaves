import os
from argparse import ArgumentParser
from time import perf_counter
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from icwaves.sikmeans.shift_kmeans import shift_invariant_k_means
from numpy.random import default_rng

parser = ArgumentParser()
parser.add_argument("--root", help="Path to root folder", default=os.getcwd())
parser.add_argument("--class-label", type=int, default=1,
                    choices=[1, 2, 3, 4, 5, 6, 7], help="ICLabel index")
# Same as srate (1 second window):
parser.add_argument("--centroid-len", type=int, default=256,
                    help="Centroid length")
# 1.5 * centroid_len:
parser.add_argument("--window-len", type=int, default=384,
                    help="Length of non-overlapping window length")
parser.add_argument('--num-clusters', type=int,
                    default=16, help='Number of clusters')
parser.add_argument('--n-runs', type=int,
                    default=3, help='Number of runs')
parser.add_argument('--n-jobs', type=int,
                    default=1, help='Value for n_jobs (sklearn)')
parser.add_argument('--train-ic-hours', type=float,
                    default=7, help='Number of total training hours')


EXPERT_ANNOTATED_CLASSES = [1, 2, 3] # brain, muscle, eye

if __name__ == '__main__':

    args = parser.parse_args()

    srate, win_len = 256, args.window_len

    data_dir = Path(args.root, 'data/ds003004/icact_iclabel')
    file_list = data_dir.glob(f'train_subj-*.mat')

    icaact_list = []
    for file in file_list:
        with file.open('rb') as f:
            matdict = loadmat(f)
            icaact = matdict['icaact']
            noisy_labels = matdict['noisy_labels']
            expert_labels = matdict['expert_labels']

        if args.class_label in EXPERT_ANNOTATED_CLASSES:
            ic_ind = (expert_labels == args.class_label).nonzero()[0]
        else:
            winner_class = np.argmax(noisy_labels, axis=1)
            winner_class = winner_class + 1 # python to matlab indexing base
            ic_ind = (winner_class == args.class_label).nonzero()[0]

        icaact_list.append(icaact[ic_ind])

    # ICs from different subjects have different lenths, so we don't
    # concatenate into a single array
    n_ics_per_subj = np.array(list(map(lambda x: x.shape[0], icaact_list)))
    n_ics = np.sum(n_ics_per_subj)
    rng = default_rng(13)

    min_time_pnts = np.min(
        np.array(list(map(lambda x: x.shape[1], icaact_list))))
    tot_win = args.train_ic_hours / (win_len/srate/3600)
    n_win_per_ic = np.ceil(tot_win/n_ics).astype(int)
    tot_win = n_win_per_ic * n_ics
    assert n_win_per_ic * win_len < min_time_pnts - win_len + 1

    X = np.zeros((tot_win, win_len), dtype=icaact_list[0].dtype)
    win_start = 0
    for ics in icaact_list:
        for ic in ics:
            time_idx = np.arange(0, ic.size-win_len+1, win_len)
            time_idx = rng.choice(time_idx, size=n_win_per_ic, replace=False)
            time_idx = time_idx[:, None] + np.arange(win_len)[None, :]
            X[win_start:win_start+n_win_per_ic] = ic[time_idx]
            win_start += n_win_per_ic
    del icaact_list

    metric, init = 'cosine', 'random'
    start = perf_counter()
    centroids, labels, shifts, distances, inertia, _ = shift_invariant_k_means(
        X, args.num_clusters, args.centroid_len, metric=metric, init=init, n_init=args.n_runs, rng=rng,  verbose=True, n_jobs=args.n_jobs)
    stop = perf_counter()
    print(f'Time running sikmeans: {stop-start:.3f} seconds')

    dict_dir = Path(args.root, 'results/dictionaries')
    out_file = dict_dir.joinpath(
        f'sikmeans_P-{args.centroid_len}_k-{args.num_clusters}'
        f'_class-{args.class_label}.npz'
    )
    with out_file.open('wb') as f:
        np.savez(out_file, centroids=centroids, labels=labels,
                shifts=shifts, distances=distances, inertia=inertia)
