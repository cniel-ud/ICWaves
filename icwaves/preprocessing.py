import logging
from pathlib import Path
from argparse import Namespace

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from icwaves.file_utils import _build_ics_and_labels_file, _build_preprocessed_data_file
from icwaves.data.types import DataBundle


def _get_base_metadata(args):
    data_dir = Path(args.path_to_raw_data)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")

    fnames = [f"subj-{i:02}.mat" for i in args.subj_ids]
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
    n_points = int(args.minutes_per_ic * 60 * srate)

    logging.info(f"Sampling rate = {srate} Hz")
    logging.info(f"Total number of ICs = {n_ics}")
    logging.info(f"Number of time points per IC = {n_points}")

    return file_list, n_ics, n_points, srate


def _get_metadata_for_windowed_ics(args):

    file_list, n_ics, n_points, srate = _get_base_metadata(args)

    n_win_per_ic = int(np.ceil(n_points / (args.window_length * srate)).item())
    logging.info(f"Number of windows per IC = {n_win_per_ic}")

    return file_list, n_ics, n_win_per_ic, srate


def _get_windowed_ics_and_labels(args):
    file_list, n_ics, n_win_per_ic, srate = _get_metadata_for_windowed_ics(args)
    window_length = int(args.window_length * srate)
    logging.info(f"Window length = {window_length} samples")

    logging.info("Building data matrix, labels, and other metadata...")
    # NOTE: float32. ICs were saved in matlab as single.
    windowed_ics = np.zeros((n_ics, n_win_per_ic, window_length), dtype=np.float32)
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
            time_idx = np.arange(0, ic.size - window_length + 1, window_length)
            time_idx = time_idx[:n_win_per_ic]
            time_idx = time_idx[:, None] + np.arange(window_length)[None, :]
            windowed_ics[cum_ic_ind] = ic[time_idx]
            labels[cum_ic_ind] = labels_per_subject[ic_ind]
            noisy_labels[cum_ic_ind] = noisy_labels_per_subject[ic_ind]
            expert_label_mask[cum_ic_ind] = expert_label_mask_per_subject[ic_ind]
            subj_ind[cum_ic_ind] = subjID
            cum_ic_ind += 1

        logging.info("Done building data matrix, labels, and other metadata")

    return DataBundle(
        data=windowed_ics,
        labels=labels,
        expert_label_mask=expert_label_mask,
        subj_ind=subj_ind,
        noisy_labels=noisy_labels,
        srate=srate,
    )


def _get_ics_and_labels(args):
    file_list, n_ics, n_points, srate = _get_base_metadata(args)

    logging.info("Building data matrix, labels, and other metadata...")
    # NOTE: float32. ICs were saved in matlab as single.
    ics = np.zeros((n_ics, n_points), dtype=np.float32)
    labels = -1 * np.ones(n_ics, dtype=int)

    ic_start = 0
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

        ic_end = ic_start + ica_activations.shape[0]
        ics[ic_start:ic_end] = ica_activations[:, :n_points]
        labels[ic_start:ic_end] = labels_per_subject.squeeze()
        noisy_labels[ic_start:ic_end] = noisy_labels_per_subject
        expert_label_mask[ic_start:ic_end] = expert_label_mask_per_subject.squeeze()
        subj_ind[ic_start:ic_end] = subjID
        ic_start = ic_end

    logging.info("Done building data matrix, labels, and other metadata")

    return DataBundle(
        data=ics,
        labels=labels,
        expert_label_mask=expert_label_mask,
        subj_ind=subj_ind,
        noisy_labels=noisy_labels,
        srate=srate,
    )


def load_or_build_ics_and_labels(args: Namespace) -> DataBundle:
    data_folder = Path(args.path_to_preprocessed_data)
    data_folder.mkdir(exist_ok=True, parents=True)
    preprocessed_data_file = _build_ics_and_labels_file(args)
    preprocessed_data_file = data_folder.joinpath(preprocessed_data_file)
    if preprocessed_data_file.is_file():
        with np.load(preprocessed_data_file) as data:
            ics = data["ics"]
            labels = data["labels"]
            srate = data["srate"]
            expert_label_mask = data["expert_label_mask"]
            subj_ind = data["subj_ind"]
            noisy_labels = data["noisy_labels"]
    else:
        ics, labels, srate, expert_label_mask, subj_ind, noisy_labels = (
            _get_ics_and_labels(args)
        )
        with preprocessed_data_file.open("wb") as f:
            np.savez(
                f,
                ics=ics,
                labels=labels,
                srate=srate,
                expert_label_mask=expert_label_mask,
                subj_ind=subj_ind,
                noisy_labels=noisy_labels,
            )
    return DataBundle(
        data=ics,
        labels=labels,
        expert_label_mask=expert_label_mask,
        subj_ind=subj_ind,
        srate=srate,
        noisy_labels=noisy_labels,
    )


def load_or_build_preprocessed_data(args):
    data_folder = Path(args.path_to_preprocessed_data)
    data_folder.mkdir(exist_ok=True, parents=True)
    preprocessed_data_file = _build_preprocessed_data_file(args)
    preprocessed_data_file = data_folder.joinpath(preprocessed_data_file)
    if preprocessed_data_file.is_file():
        with np.load(preprocessed_data_file) as data:
            db = DataBundle(
                data=data["windowed_ics"],
                labels=data["labels"],
                expert_label_mask=data["expert_label_mask"],
                subj_ind=data["subj_ind"],
                noisy_labels=data["noisy_labels"],
                srate=data["srate"],
            )
    else:
        db = _get_windowed_ics_and_labels(args)
        with preprocessed_data_file.open("wb") as f:
            np.savez(
                f,
                windowed_ics=db.data,
                labels=db.labels,
                srate=db.srate,
                expert_label_mask=db.expert_label_mask,
                subj_ind=db.subj_ind,
                noisy_labels=db.noisy_labels,
            )

    return db
