# %%
"""
This script resamples dictionaries learned on the 'emotion_study' dataset (256 Hz)
to the sampling rate of the 'cue' dataset (500 Hz).
"""

import numpy as np
from scipy.signal import resample, resample_poly
from pathlib import Path
import matplotlib.pyplot as plt

from icwaves.data_loaders import load_codebooks_wrapper

emotion_sampling_rate, down = 256, 64  # original sampling rate
cue_sampling_rate, up = 500, 125
# this is the definition of `up` and `down` in resampled_poly
assert cue_sampling_rate == int((up / down) * emotion_sampling_rate)

root = Path(__file__).parent.parent
resampled_dictionaries_path = root.joinpath(
    "results/emotion_study/dictionaries_resampled"
)
resampled_dictionaries_path.mkdir(parents=True, exist_ok=True)


class Args:
    path_to_codebooks = root.joinpath("results/emotion_study/dictionaries")
    window_length = 1.5
    num_clusters = 128
    centroid_length = 1.0  # in seconds
    codebook_minutes_per_ic = 50.0
    codebook_ics_per_subject = 2


args = Args()
codebooks = load_codebooks_wrapper(args, srate=emotion_sampling_rate)
n_classes, n_centroids, centroid_length = codebooks.shape
# %%
new_centroid_length = int(cue_sampling_rate * args.centroid_length)
resampled_codebooks = np.zeros(
    (n_classes, n_centroids, new_centroid_length), dtype=codebooks.dtype
)
for i_class in range(n_classes):
    for i_centroid in range(n_centroids):
        resampled_codebooks[i_class, i_centroid, :] = resample_poly(
            codebooks[i_class, i_centroid, :], up, down, padtype="smooth"
        )

# %%
# plot a few examples of codebooks to compare the original with the resampled
x_o = np.linspace(0, args.centroid_length, codebooks.shape[-1])
x = np.linspace(0, args.centroid_length, resampled_codebooks.shape[-1])
for i_class in range(3):
    plt.figure()
    plt.plot(x_o, codebooks[i_class, 0, :], "o", label="original")
    plt.plot(x, resampled_codebooks[i_class, 0, :], label="resampled")
    plt.legend()
# %%
for i_class in range(n_classes):
    fname = (
        f"sikmeans_P-{args.centroid_length}_k-{args.num_clusters}"
        f"_class-{i_class+1}_minutesPerIC-{args.codebook_minutes_per_ic}"
        f"_icsPerSubj-{args.codebook_ics_per_subject}.npz"
    )
    fpath = resampled_dictionaries_path.joinpath(fname)
    np.savez(fpath, centroids=resampled_codebooks[i_class, :, :])
# %%
