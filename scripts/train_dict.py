import os
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.random import default_rng

from icwaves.data_loaders import load_raw_train_set_per_class
from icwaves.sikmeans.shift_kmeans import shift_invariant_k_means

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
    rng = default_rng(13)

    X = load_raw_train_set_per_class(args)

    metric, init = 'cosine', 'random'
    t_start = perf_counter()
    centroids, labels, shifts, distances, inertia, _ = shift_invariant_k_means(
        X, args.num_clusters, args.centroid_len, metric=metric, init=init, n_init=args.n_runs, rng=rng,  verbose=True, n_jobs=args.n_jobs)
    t_stop = perf_counter()
    print(f'Time running sikmeans: {t_stop-t_start:.3f} seconds')

    dict_dir = Path(args.root, 'results/dictionaries')
    out_file = dict_dir.joinpath(
        f'sikmeans_P-{args.centroid_len}_k-{args.num_clusters}'
        f'_class-{args.class_label}.npz'
    )
    with out_file.open('wb') as f:
        np.savez(out_file, centroids=centroids, labels=labels,
                shifts=shifts, distances=distances, inertia=inertia)
