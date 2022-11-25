import os
from argparse import ArgumentParser
from time import perf_counter
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearnex import patch_sklearn
patch_sklearn()
from icwaves.sikmeans.shift_kmeans import shift_invariant_k_means
from numpy.random import default_rng

parser = ArgumentParser()
parser.add_argument("--root", help="Path to root folder", default=os.getcwd())
parser.add_argument("--class-label", type=int, default=1,
                    choices=[1, 2, 3, 4, 5, 6, 7], help="ICLabel index")
parser.add_argument("--centroid-len", type=int, default=512,
                    help="Centroid length")
parser.add_argument("--window-len", type=int, default=768,
                    help="Length of non-overlapping window length")
parser.add_argument('--num-clusters', type=int,
                    default=16, help='Number of clusters')
parser.add_argument('--n-runs', type=int,
                    default=3, help='Number of runs')
parser.add_argument('--test-size', type=float,
                    default=0.2, help='Test set size as fraction of total data')



args = parser.parse_args()

data_dir = Path(args.root, 'data/ds003004/icact_iclabel')
file_list = data_dir.glob(f'icact_subj-*_iclabel-{args.class_label}.mat')

icact = []
for file in file_list:
    with file.open('rb') as f:
        matdict = loadmat(f)
        icact.append(matdict['components'])    

n_ics = len(icact)
n_test_ics = np.round(n_ics * args.test_size).astype(int)
n_train_ics = n_ics - n_test_ics
rng = default_rng(13)
train_idx = rng.choice(n_ics, size=n_train_ics, replace=False)
train_icact = [icact[i] for i in train_idx]

win_len = 768
tot_win = np.sum([ic.shape[0] * (ic.shape[1]//win_len) for ic in train_icact])
X = np.zeros((tot_win, win_len), dtype=icact[0].dtype) #XXX: use float64???
start = 0
for i, ics in enumerate(train_icact):
    for j, ic in enumerate(ics):
        idx = np.arange(0, ic.size-win_len+1, win_len)
        idx = idx[:, None] + np.arange(win_len)[None, :]
        n_win = idx.shape[0]
        X[start:start+n_win] = ic[idx]
        start += n_win    
del train_icact

P, k = args.centroid_len, args.num_clusters
metric, init = 'cosine', 'random'
start = perf_counter()
centroids, labels, shifts, distances, inertia, _ = shift_invariant_k_means(
    X, args.num_clusters, args.centroid_len, metric=metric, init=init, n_init=args.n_runs, rng=rng,  verbose=True)
stop = perf_counter()
print(f'Time running sikmeans: {stop-start:.3f} seconds')

out_file = data_dir.joinpath(
    f'sikmeans_P-{args.centroid_len}_k-{args.num_clusters}'
    f'_class-{args.class_label}.npz'
)
with out_file.open('wb') as f:
    np.savez(out_file, centroids=centroids, labels=labels,
             shifts=shifts, distances=distances, inertia=inertia)
