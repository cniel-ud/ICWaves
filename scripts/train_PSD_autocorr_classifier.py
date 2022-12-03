# %%
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from scipy.io import loadmat
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from icwaves.model_selection.search import grid_search_cv
from icwaves.model_selection.split import LeaveOneSubjectOutExpertOnly

parser = ArgumentParser()
parser.add_argument(
    "--root", help="Path to root folder of the project", required=True)
parser.add_argument('--srate', type=float,
                    default=256, help='Sampling rate')
parser.add_argument('--n-jobs', type=int,
                    default=1, help='Value for n_jobs (sklearn)')
parser.add_argument('--regularization-factor', type=float, nargs='+',
                    default=[0.1, 1, 10], help='Regularization factor used by the classifier. In LogisticRegression, it is the value of C.')
parser.add_argument('--expert-weight', type=float, nargs='+',
                    default=[1, 2, 4], help='Sample weight given to ICs with expert labels.')
parser.add_argument('--l1-ratio', type=float, nargs='+',
                    default=[0, 0.2, 0.4, 0.6, 0.8, 1])
parser.add_argument('--max-iter', type=int,
                    default=1000, help='Maximum number of iterations')
parser.add_argument('--penalty', default='elasticnet',
                    choices=['l1', 'l2', 'elasticnet', 'none'])


if __name__ == '__main__':

    args = parser.parse_args()

    new_rng = default_rng(13)
    # scikit-learn doesn't support the new numpy Generator:
    old_rng = np.random.RandomState(13)

    # Load data
    data_file = Path(args.root, 'data/ds003004/spectral_features',
                    'train_data.mat')
    with data_file.open('rb') as f:
        matdict = loadmat(f)
        X = matdict['X_train']
        y = matdict['y_train']
        expert_label_mask = matdict['expert_label_mask_train']
        subj_ind = matdict['subj_ind_ar_train']

    # We expect a 1D array. Matlab always add a singleton dimension that we need to
    # remove here.
    y = y.squeeze()
    expert_label_mask = expert_label_mask.squeeze()
    subj_ind = subj_ind.squeeze()

    # Make sure expert_label_mask is boolean. Matlab R2020b converts to double
    # when concatenating booleans! Might be removed once we generate the data
    # from Matlab again with the right type.
    expert_label_mask = expert_label_mask.astype(bool)

    cv = LeaveOneSubjectOutExpertOnly(expert_label_mask)

    # ICLabel max-scale PSDs and autocorrelation functions
    # Do we still need to scale the features here?
    # My intuition tells me not...I checked and although the features are not
    # zero mean, their mean is not widely different, and the variances are
    # rather small.
    clf = LogisticRegression()

    clf_params = dict(
        class_weight='balanced',
        solver='saga',
        penalty=args.penalty,
        random_state=old_rng,
        multi_class='multinomial',
        warm_start=True,
        max_iter=args.max_iter,
    )
    clf.set_params(**clf_params)

    candidate_params = dict(
        C=args.regularization_factor,
        l1_ratio=args.l1_ratio,
        expert_weight=args.expert_weight
    )
    results = grid_search_cv(
        clf,
        candidate_params,
        X,
        y,
        subj_ind,
        expert_label_mask,
        cv,
        args.n_jobs
    )

    C_str = '_'.join([str(i) for i in candidate_params['C']])
    l1_ratio_str = '_'.join([str(i)
                            for i in candidate_params['l1_ratio']])
    ew_str = '_'.join([str(i) for i in candidate_params['expert_weight']])
    fname = (
        f'clf-lr_penalty-{args.penalty}_solver-saga_C-{C_str}'
        f'_l1_ratio-{l1_ratio_str}'
        f'_expert_weight-{ew_str}'
        '_PSD-autocorr.pickle'
    )
    fpath = Path(args.root, 'results/classifier', fname)
    with fpath.open('wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
