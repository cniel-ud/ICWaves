# %%
"""
This script resamples the ICs in the cue dataset (500 Hz) to the
sampling rate of the 'emotion_study' dataset (256 Hz).
"""

from scipy.signal import resample_poly
from pathlib import Path
from scipy.io import loadmat, savemat
import numpy as np


emotion_sampling_rate, up = 256, 64
cue_sampling_rate, down = 500, 125  # original sampling rate
# this is the definition of `up` and `down` in resampled_poly
assert emotion_sampling_rate == int((up / down) * cue_sampling_rate)

root = Path(__file__).parent.parent

ICs_path = root.joinpath("data/cue/raw_data_and_IC_labels")
resampled_ICs_path = root.joinpath("data/cue/resampled_raw_data_and_IC_labels")
resampled_ICs_path.mkdir(parents=True, exist_ok=True)
# %%
subj_ids = range(1, 13)
# %%
for subj_id in subj_ids:
    fname = f"subj-{subj_id:02}.mat"
    file = ICs_path.joinpath(fname)
    print(f"Loading data from {file} and subject {subj_id}")
    with file.open("rb") as f:
        matdict = loadmat(f)
        data = matdict["data"]
        icaweights = matdict["icaweights"]
        icasphere = matdict["icasphere"]
        noisy_labels = matdict["noisy_labels"]
        expert_label_mask = matdict["expert_label_mask"]
        # -1: Let class labels start at 0 in python
        labels = matdict["labels"]

    print("Resampling data")
    data = resample_poly(data, up, down, padtype="smooth", axis=1)
    print("Done")

    # save all variables in the resampled folder
    resampled_file = resampled_ICs_path.joinpath(fname)
    print(f"Saving resampled data to {resampled_file}")
    matdict = {
        "data": data,
        "icaweights": icaweights,
        "icasphere": icasphere,
        "noisy_labels": noisy_labels,
        "expert_label_mask": expert_label_mask,
        "labels": labels,
        "srate": emotion_sampling_rate,
    }
    with resampled_file.open("wb") as f:
        savemat(f, matdict)
        print("Done")
# %%
