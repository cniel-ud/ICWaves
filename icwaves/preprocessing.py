import logging
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import tqdm

from icwaves.utils import _build_preprocessed_data_file


def _get_metadata(args):
    data_dir = args.path_to_raw_data
    fnames = [f"subj-{i}.mat" for i in args.subj_ids]
    file_list = [data_dir.joinpath(f) for f in fnames]

    logging.info("Getting number of time series and sampling rate...")
    logging.info("NOTE: Assumes all subjects have the same sampling rate")
    logging.info(f"Data path: {data_dir}")
    n_ics_per_subj = []
    for file in file_list:
        with file.open("rb") as f:
            matdict = loadmat(f, variable_names=["labels", "srate"])
            labels = matdict["labels"]
            srate = matdict["srate"]
            srate = srate.item(0)  # `srate.shape=(1,1)`. This extracts the number.
            n_ics_per_subj.append(labels.shape[0])

    n_ics = np.sum(n_ics_per_subj)
    minutes_per_window = args.window_len / srate / 60
    n_win_per_ic = np.ceil(args.minutes_per_ic / minutes_per_window).astype(int)

    logging.info(f"Sampling rate = {srate} Hz")
    logging.info(f"Total number of ICs = {n_ics}")
    logging.info(f"Windown length = {minutes_per_window * 60} seconds")
    logging.info(f"Number of windows per IC = {n_win_per_ic}")

    return file_list, n_ics, n_win_per_ic, srate


def _get_windowed_ics_and_labels(args):
    file_list, n_ics, n_win_per_ic, srate = _get_metadata(args)
    window_len = args.window_len

    logging.info("Building data matrix, labels, and other metadata...")
    # NOTE: float32. ICs were saved in matlab as single.
    windowed_ics = np.zeros((n_ics, n_win_per_ic, window_len), dtype=np.float32)
    labels = -1 * np.ones(n_ics, dtype=int)

    cum_ic_ind = 0
    expert_label_mask = np.full(n_ics, False)
    subj_ind = np.zeros(n_ics, dtype=int)
    # 7 ICLabel classes
    noisy_labels = np.zeros((n_ics, 7), dtype=np.float32)
    for file, subjID in tqdm(zip(file_list, args.subj_ids)):
        logging.info(f"Loading data from {file} and subject {subjID}")
        with file.open("rb") as f:
            matdict = loadmat(f)
            data = matdict["data"]
            icaweights = matdict["icaweights"]
            icasphere = matdict["icasphere"]
            noisy_labels_per_subject = matdict["noisy_labels"]
            expert_label_mask_per_subject = matdict["expert_label_mask"]
            # -1: Let class labels start at 0 in python
            labels_per_subject = matdict["labels"] - 1

        expert_label_mask_per_subject = expert_label_mask_per_subject.astype(bool)
        ica_activations = icaweights @ icasphere @ data

        for ic_ind, ic in enumerate(ica_activations):
            time_idx = np.arange(0, ic.size - window_len + 1, window_len)
            time_idx = time_idx[:n_win_per_ic]
            time_idx = time_idx[:, None] + np.arange(window_len)[None, :]
            windowed_ics[cum_ic_ind] = ic[time_idx]
            labels[cum_ic_ind] = labels_per_subject[ic_ind]
            noisy_labels[cum_ic_ind] = noisy_labels_per_subject[ic_ind]
            expert_label_mask[cum_ic_ind] = expert_label_mask_per_subject[ic_ind]
            subj_ind[cum_ic_ind] = subjID
            cum_ic_ind += 1

        logging.info("Done building data matrix, labels, and other metadata")

    return windowed_ics, labels, srate, expert_label_mask, subj_ind, noisy_labels


def load_or_build_preprocessed_data(args):
    preprocessed_data_file = _build_preprocessed_data_file(args)
    if preprocessed_data_file.is_file():
        with np.load(preprocessed_data_file) as data:
            windowed_ics = data["windowed_ics"]
            labels = data["labels"]
            srate = data["srate"]
            expert_label_mask = data["expert_label_mask"]
            subj_ind = data["subj_ind"]
    else:
        (
            windowed_ics,
            labels,
            srate,
            expert_label_mask,
            subj_ind,
            _,
        ) = _get_windowed_ics_and_labels(args)
        with preprocessed_data_file.open("wb") as f:
            np.savez(
                f,
                windowed_ics=windowed_ics,
                labels=labels,
                srate=srate,
                expert_label_mask=expert_label_mask,
                subj_ind=subj_ind,
            )

    return windowed_ics, labels, srate, expert_label_mask, subj_ind
