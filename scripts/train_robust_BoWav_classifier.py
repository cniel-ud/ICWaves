# %%
import copy
import logging
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from icwaves.data_loaders import load_codebooks, load_codebooks_wrapper, load_raw_set
from icwaves.feature_extractors.bowav import bag_of_waves, build_or_load_centroid_assignments
from icwaves.model_selection.search import grid_search_cv
from icwaves.model_selection.split import LeaveOneSubjectOutExpertOnly
from icwaves.preprocessing import load_or_build_preprocessed_data
from icwaves.utils import _build_results_file
import sklearn
import scipy

parser = ArgumentParser()
parser.add_argument(
    "--path-to-raw-data", help="Path to raw data", required=True)
parser.add_argument(
    "--path-to-preprocessed-data", help="Path to preprocessed data", required=True)
parser.add_argument(
    "--path-to-centroid-assignments-folder", help="Path to centroid assignments", required=True)
parser.add_argument(
    "--path-to-results", help="Path to results", required=True)
parser.add_argument(
    "--path-to-codebooks", help="Path to codebooks", required=True)
parser.add_argument('--subj-ids', help='A list with the subject ids to be used during training.', nargs='+', required=True)
# parser.add_argument('--srate', type=float,
#                     default=256, help='Sampling rate')
# Same as srate (1 second window):
parser.add_argument("--centroid-len", type=float, default=1.,
                    help="Centroid length in seconds")
# 1.5 * centroid_len:
parser.add_argument("--window-length", type=float, default=1.5,
                    help="Length of window assigned to centroid, in seconds")
parser.add_argument("--training-segment-length", type=float, nargs='+', default=[10, 30, 90, 180, 300],
                    help="Length in seconds of segment used during training.")
parser.add_argument("--n-seconds-for-validation", type=int, default=5,
                    help="Number of seconds used for validation. Use -1 if you want to use all the validation data")
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
                    choices=['l1', 'l2', 'elasticnet', 'none'])
parser.add_argument('--codebook-minutes-per-ic', type=float,
                    default=None, help='Number of minutes per IC to train the class-specific codebook')
parser.add_argument('--codebook-ics-per-subject', type=int,
                    default=2, help='Maximum number of ICs per subject to train the class-specific codebook')
parser.add_argument('--bowav-norm', help='Instance-wise normalization in BoWav', nargs='+', default=['none', 'l_1', 'l_2', 'l_inf'])


BOWAV_NORM_MAP = {
    'none': None,
    'l_1': 1,
    'l_2': 2,
    'l_inf': np.inf,
}

if __name__ == '__main__':

    logging.basicConfig(format='%(levelname)s:%(filename)s:%(message)s', level=logging.DEBUG)
    logging.info('Started')

    args = parser.parse_args()

    new_rng = default_rng(13)
    # scikit-learn doesn't support the new numpy Generator:
    old_rng = np.random.RandomState(13)

    # Load or build preprocessed data
    windowed_ics, labels, srate, expert_label_mask, subj_ind = load_or_build_preprocessed_data(args)

    # Load codebooks
    codebooks = load_codebooks_wrapper(args)
    n_centroids = codebooks[0].shape[0]

    # Load or build centroid assignments
    centroid_assignments = build_or_load_centroid_assignments(args, windowed_ics, codebooks)

    input_or_output_aggregation_method = ['count_pooling', 'majority_vote']

    training_segment_length = args.training_segment_length
    # convert segment length (in seconds) to n_training_windows_per_segment
    n_training_windows_per_segment = [int(s * srate / args.window_length) for s in training_segment_length]
    logging.info(f"n_training_windows_per_segment: {n_training_windows_per_segment}")

    n_seconds_for_validation = args.n_seconds_for_validation
    # convert n_seconds_for_validation to n_windows_for_validation
    if n_seconds_for_validation == -1:
        n_windows_for_validation = None
    else:
        n_windows_for_validation = int(n_seconds_for_validation * srate / args.window_length)
    logging.info(f"n_windows_for_validation: {n_windows_for_validation}")

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
        expert_weight=args.expert_weight,
        bowav_norm=args.bowav_norm,
        input_or_output_aggregation_method=input_or_output_aggregation_method,
        n_training_windows_per_segment=n_training_windows_per_segment,
        n_windows_for_validation=n_windows_for_validation,
        n_centroids=n_centroids,
    )
    results = grid_search_cv(
        pipe,
        candidate_params,
        centroid_assignments,
        labels,
        subj_ind,
        expert_label_mask,
        cv,
        args.n_jobs
    )


    results_file = _build_results_file(
        args,
        candidate_params['clf__C'],
        candidate_params['clf__l1_ratio'],
        candidate_params['expert_weight'],
        candidate_params['bowav_norm'],
        training_segment_length,
        n_seconds_for_validation,
    )

    # Add to results the version of scikit-learn, numpy, and
    # scipy to improve reproducibility
    results['sklearn_version'] = sklearn.__version__
    results['numpy_version'] = np.__version__
    results['scipy_version'] = scipy.__version__

    with results_file.open('wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    logging.info('Finished')
