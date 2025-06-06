# %%
"""
This script resamples CMMN filters learned at 256 Hz
to the sampling rate of the 'cue' dataset (500 Hz).
"""

import numpy as np
from scipy.signal import resample_poly
from pathlib import Path
import matplotlib.pyplot as plt

emotion_sampling_rate, down = 256, 64  # original sampling rate
cue_sampling_rate, up = 500, 125
# this is the definition of `up` and `down` in resampled_poly
assert cue_sampling_rate == int((up / down) * emotion_sampling_rate)

root = Path(__file__).parent.parent
cmmn_filter = "subj_to_subj"
resampled_filters_path = root.joinpath("data/cue/cmmn_filters_resampled")
resampled_filters_path = resampled_filters_path.joinpath(cmmn_filter)
resampled_filters_path.mkdir(parents=True, exist_ok=True)


path_to_cmmn_filters = root.joinpath("data/cue/cmmn_filters", cmmn_filter)
cmmn_path = Path(path_to_cmmn_filters)
subj_ids = list(range(1, 13))
cmmn_filters = {}
cmmn_filters_resampled = {}
for subj_id in subj_ids:
    fname = f"subj-{subj_id:02}.npz"
    fpath = cmmn_path.joinpath(fname)
    if not fpath.exists():
        raise FileNotFoundError(f"File {fpath} does not exist.")
    with np.load(fpath) as cmmn_map:
        cmmn_filters[subj_id] = cmmn_map["arr_0"]
    cmmn_filters_resampled[subj_id] = resample_poly(
        cmmn_filters[subj_id], up, down, padtype="smooth"
    )
    fname = f"subj-{subj_id:02}.npz"
    fpath = resampled_filters_path.joinpath(fname)
    np.savez(fpath, cmmn_filters_resampled[subj_id])

# %%
# plot a few examples of CMMN filters to compare the original with the resampled
for subj_id in subj_ids:
    plt.figure()

    # Create time axes with the same start and end points
    original_len = cmmn_filters[subj_id].shape[0]
    resampled_len = cmmn_filters_resampled[subj_id].shape[0]

    # Calculate the time duration (assuming both cover the same time period)
    time_duration = original_len / emotion_sampling_rate  # in seconds

    # Create time axes
    t_original = np.linspace(0, time_duration, original_len)
    t_resampled = np.linspace(0, time_duration, resampled_len)

    # Plot with proper time axes
    plt.plot(t_original, cmmn_filters[subj_id], label="original")
    plt.plot(t_resampled, cmmn_filters_resampled[subj_id], label="resampled")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.title(f"Subject {subj_id}")
    plt.show()

# %%
