# %%
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from icwaves.data_loaders import load_codebooks, load_raw_set
from icwaves.feature_extractors.bowav import bag_of_waves
from icwaves.model_selection.search import grid_search_cv
from icwaves.model_selection.split import LeaveOneSubjectOutExpertOnly

parser = ArgumentParser()
parser.add_argument(
    "--root", help="Path to root folder of the project", required=True)
parser.add_argument('--srate', type=float,
                    default=256, help='Sampling rate')
# Same as srate (1 second window):
parser.add_argument("--centroid-len", type=int, default=256,
                    help="Centroid length")
# 1.5 * centroid_len:
parser.add_argument("--window-len", type=int, default=384,
                    help="Length of non-overlapping window length")
parser.add_argument('--num-clusters', type=int,
                    default=16, help='Number of clusters')
parser.add_argument('--n-jobs', type=int,
                    default=1, help='Value for n_jobs (sklearn)')
parser.add_argument('--minutes-per-ic', type=float,
                    default=15, help='Number of minutes per IC to extract BagOfWaves features')
parser.add_argument('--regularization-factor', type=float, nargs='+',
                    default=[0.1, 1, 10], help='Regularization factor used by the classifier. In LogisticRegression, it is the value of C.')
parser.add_argument('--expert-weight', type=float, nargs='+',
                    default=[1, 2, 4], help='Sample weight given to ICs with expert labels.')
parser.add_argument('--l1-ratio', type=float, nargs='+',
                    default=[0, 0.2, 0.4, 0.6, 0.8, 1])
parser.add_argument('--max-iter', type=int,
                    default=1000, help='Maximum number of iterations')
parser.add_argument('--penalty', default='elasticnet',
                    options=['l1', 'l2', 'elasticnet', 'none'])


if __name__ == '__main__':

    args = parser.parse_args()

    new_rng = default_rng(13)
    # scikit-learn doesn't support the new numpy Generator:
    old_rng = np.random.RandomState(13)

    # Load/generate data
    fname = (
        f'data_k-{args.num_clusters}_P-{args.centroid_len}'
        f'winlen-{args.window_len}_minPerIC-{args.minutes_per_ic}.npz'
    )
    data_file = Path(args.root, 'data/ds003004/BoWav', fname)
    if data_file.is_file():
        with np.load(data_file) as data:
            X = data['X']
            y = data['y']
            expert_label_mask = data['expert_label_mask']
            subj_ind = data['subj_ind']
    else:
        raw_ics, y, expert_label_mask, subj_ind, _ = \
            load_raw_set(args, new_rng)
        codebooks = load_codebooks(args)
        X = bag_of_waves(raw_ics, codebooks)
        with data_file.open('wb') as f:
            np.savez(
                f, raw_ics=raw_ics, X=X, y=y,
                expert_label_mask=expert_label_mask, subj_ind=subj_ind)

    cv = LeaveOneSubjectOutExpertOnly(expert_label_mask)

    pipe = Pipeline([('scaler', TfidfTransformer()),
                    ('clf', LogisticRegression())])

    clf_params = dict(
        clf__class_weight='balanced',
        clf__solver='saga',
        clf__penalty=args.penalty,
        clf__random_state=old_rng,
        clf__multi_class='multinomial',
        clf__warm_start=True,
        clf__max_iter=args.max_iter,
    )
    pipe.set_params(**clf_params)

    candidate_params = dict(
        clf__C=args.regularization_factor,
        clf__l1_ratio=args.l1_ratio,
        expert_weight=args.expert_weight
    )
    results = grid_search_cv(
        pipe,
        candidate_params,
        X,
        y,
        subj_ind,
        expert_label_mask,
        cv,
        args.n_jobs
    )

    C_str = '_'.join([str(i) for i in candidate_params['clf__C']])
    l1_ratio_str = '_'.join([str(i) for i in candidate_params['clf__l1_ratio']])
    ew_str = '_'.join([str(i) for i in candidate_params['expert_weight']])
    fname = (
        f'clf-lr_penalty-{args.penalty}_solver-saga_C-{C_str}'
        f'_l1_ratio-{l1_ratio_str}'
        f'_expert_weight-{ew_str}.pickle'
    )
    fpath = Path(args.root, 'results/classifier', fname)
    with fpath.open('wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
