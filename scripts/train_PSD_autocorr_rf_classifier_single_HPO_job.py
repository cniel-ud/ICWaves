# %%
from itertools import product
import logging
import pickle
from pathlib import Path

import numpy as np
from numpy.random import default_rng
import scipy
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

from icwaves.feature_extractors.iclabel_features import get_iclabel_features_per_segment
from icwaves.file_utils import read_args_from_file
from icwaves.model_selection.split import LeaveOneSubjectOutExpertOnly
from icwaves.argparser import (
    create_argparser_one_parameter_one_split,
    create_argparser_all_params,
)
from icwaves.model_selection.validation import _fit_and_score
from icwaves.preprocessing import _get_ics_and_labels, load_or_build_ics_and_labels


if __name__ == "__main__":

    logging.basicConfig(
        format="%(levelname)s:%(filename)s:%(message)s", level=logging.DEBUG
    )
    logging.info("Started")

    parser = create_argparser_one_parameter_one_split()
    one_run_args = parser.parse_args()

    args_list = read_args_from_file(one_run_args.path_to_config_file)
    job_id = one_run_args.job_id

    all_params_parser = create_argparser_all_params()
    args = all_params_parser.parse_args(args_list)

    new_rng = default_rng(13)
    # scikit-learn doesn't support the new numpy Generator:
    old_rng = np.random.RandomState(13)

    # Load data
    ics, labels, srate, expert_label_mask, subj_ind, noisy_labels = (
        load_or_build_ics_and_labels(args)
    )

    input_or_output_aggregation_method = ["count_pooling", "majority_vote"]
    training_segment_length = [int(l * srate) for l in args.training_segment_length]
    validation_segment_length = args.validation_segment_length
    if validation_segment_length == -1:
        validation_segment_length = None
    else:
        validation_segment_length = int(validation_segment_length * srate)
    validation_segment_length = [validation_segment_length]

    cv = LeaveOneSubjectOutExpertOnly(expert_label_mask)

    # ICLabel max-scale PSDs and autocorrelation functions
    # Do we still need to scale the features here?
    # My intuition tells me not...I checked and although the features are not
    # zero mean, their mean is not widely different, and the variances are
    # rather small.
    clf = RandomForestClassifier()

    clf_params = dict(
        n_estimators=300,
        random_state=old_rng,
        class_weight="balanced",
    )
    clf.set_params(**clf_params)

    candidate_params = dict(
        min_samples_split=args.min_samples_split,
        expert_weight=args.expert_weight,
        input_or_output_aggregation_method=input_or_output_aggregation_method,
        training_segment_length=training_segment_length,
        validation_segment_length=validation_segment_length,
    )

    candidate_params = list(ParameterGrid(candidate_params))
    n_candidates = len(candidate_params)
    list_of_candidate_params_and_splits = list(
        product(
            enumerate(candidate_params),
            enumerate(cv.split(ics, labels, subj_ind)),
        )
    )
    (cand_idx, parameters), (split_idx, (train, test)) = (
        list_of_candidate_params_and_splits[job_id]
    )

    n_splits = cv.get_n_splits(ics, labels, groups=subj_ind)

    # '0' is the 'brain' class. We want to compute the F1-score for this class only.
    parameters["scorer_kwargs"] = {"labels": [0], "average": None}

    # `get_iclabel_features_per_segment` has args (signal, sfreq, use_autocorr, segment_len)
    # `feature_extractor` has args (time_series, segment_length)
    feature_extractor = (
        lambda time_series, segment_length: get_iclabel_features_per_segment(
            signal=time_series,
            sfreq=srate,
            use_autocorr=True,
            segment_len=segment_length,
        )
    )

    result = _fit_and_score(
        clf,
        ics,
        labels,
        expert_label_mask,
        train=train,
        test=test,
        parameters=parameters,
        scorer=f1_score,
        feature_extractor=feature_extractor,
        split_progress=(split_idx, n_splits),
        candidate_progress=(cand_idx, n_candidates),
    )

    results_folder = Path(args.path_to_results, "tmp_random_forest_PSD_autocorr")
    results_folder.mkdir(exist_ok=True, parents=True)
    results_file = f"candidate_{cand_idx}_split_{split_idx}.pkl"
    results_file = results_folder.joinpath(results_file)

    # Add to results the version of scikit-learn, numpy, and
    # scipy to improve reproducibility
    result["sklearn_version"] = sklearn.__version__
    result["numpy_version"] = np.__version__
    result["scipy_version"] = scipy.__version__

    with results_file.open("wb") as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    logging.info("Finished")
