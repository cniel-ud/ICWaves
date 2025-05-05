import copy
import logging
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from icwaves.preprocessing import _get_metadata_for_windowed_ics

EXPERT_ANNOTATED_CLASSES = [1, 2, 3]  # brain, muscle, eye (Matlab indexing)

CLASS_LABELS = [
    "Brain",
    "Muscle",
    "Eye",
    "Heart",
    "Line Noise",
    "Channel Noise",
    "Other",
]


# The train set here were subjects 8 to 35 (subject 22 is missing) from the
# 'emotion_study' dataset.
def load_raw_train_set_per_class(args, rng):
    file_list, _, n_win_per_ic, srate = _get_metadata_for_windowed_ics(args)

    ic_ind_per_subj_dict = {}
    file_dict = {}
    n_ics_per_subj_dict = {}
    for i_subj, file in zip(args.subj_ids, file_list):
        with file.open("rb") as f:
            matdict = loadmat(f, variable_names=["labels"])
            labels = matdict["labels"]

        ic_ind = (labels == args.class_label).nonzero()[0]

        if ic_ind.size > 0:  # subject has IC class
            ic_ind_per_subj_dict[i_subj] = ic_ind
            file_dict[i_subj] = file
            n_ics_per_subj_dict[i_subj] = ic_ind.size
            if ic_ind.size > args.ics_per_subject:
                n_ics_per_subj_dict[i_subj] = args.ics_per_subject
                ic_ind_per_subj_dict[i_subj] = rng.choice(
                    ic_ind_per_subj_dict[i_subj],
                    size=args.ics_per_subject,
                    replace=False,
                )

    n_ics = sum(n_ics_per_subj_dict.values())
    tot_win = n_ics * n_win_per_ic
    tot_hrs = tot_win * args.window_length / 3600
    print(f"Training ICs for '{CLASS_LABELS[args.class_label-1]}': {n_ics}")
    print(f"Number of training hours: {tot_hrs:.2f}")

    window_length = int(args.window_length * srate)
    ic_windows = np.zeros((tot_win, window_length), dtype=np.float32)
    win_start = 0
    for i_subj, file in tqdm(file_dict.items()):
        with file.open("rb") as f:
            matdict = loadmat(f)
            data = matdict["data"]
            icaweights = matdict["icaweights"]
            icasphere = matdict["icasphere"]

        icaact = icaweights @ icasphere @ data
        icaact = icaact[ic_ind_per_subj_dict[i_subj]]

        if args.path_to_cmmn_filters is not None:
            cmmn_path = Path(args.path_to_cmmn_filters)
            fname = f"subj-{i_subj:02}.npz"
            fpath = cmmn_path.joinpath(fname)
            if not fpath.exists():
                raise FileNotFoundError(f"File {fpath} does not exist.")
            with np.load(fpath) as cmmn_map:
                cmmn_filter = cmmn_map["arr_0"]

        for ic_ind, ic in tqdm(enumerate(icaact)):
            time_idx = np.arange(0, ic.size - window_length + 1, window_length)
            time_idx = time_idx[:n_win_per_ic]
            time_idx = time_idx[:, None] + np.arange(window_length)[None, :]
            if args.path_to_cmmn_filters is not None:
                ic = np.convolve(ic, cmmn_filter, mode="full")[: len(ic)]
            ic_windows[win_start : win_start + n_win_per_ic] = ic[time_idx]
            win_start += n_win_per_ic

    return ic_windows, srate


def load_codebooks(args):
    dict_dir = Path(args.path_to_codebooks)
    if not dict_dir.is_dir():
        raise ValueError(f"Directory {dict_dir} does not exist")

    # TODO: avoid hard coding this
    n_codebooks = 7

    # TODO: move to a function in charge of building this name
    fname = (
        f"sikmeans_P-{args.centroid_length}_k-{args.num_clusters}"
        f"_class-{1}_minutesPerIC-{args.minutes_per_ic}"
        f"_icsPerSubj-{args.ics_per_subject}.npz"
    )
    fpath = dict_dir.joinpath(fname)
    with np.load(fpath) as data:
        codebook_class_1 = data["centroids"]
    n_centroids, centroid_length = codebook_class_1.shape

    # TODO: args.num_clusters might not be longer needed?
    codebooks = np.zeros((n_codebooks, n_centroids, centroid_length), dtype=np.float32)
    codebooks[0] = codebook_class_1

    for i_class in range(1, n_codebooks):
        fname = (
            # TODO: The P value is now in seconds and not in number of samples, to avoid
            # the need of knowing the sampling rate. Refactor code that saves the file,
            # and manually rename files of codebooks that were already learned.
            f"sikmeans_P-{args.centroid_length}_k-{args.num_clusters}"
            f"_class-{i_class+1}_minutesPerIC-{args.minutes_per_ic}"
            f"_icsPerSubj-{args.ics_per_subject}.npz"
        )
        fpath = dict_dir.joinpath(fname)
        with np.load(fpath) as data:
            codebooks[i_class] = data["centroids"]

    return codebooks


def load_codebooks_wrapper(args):
    codebook_args = copy.deepcopy(args)
    codebook_args.minutes_per_ic = args.codebook_minutes_per_ic
    codebook_args.ics_per_subject = args.codebook_ics_per_subject
    codebooks = load_codebooks(codebook_args)
    return codebooks
