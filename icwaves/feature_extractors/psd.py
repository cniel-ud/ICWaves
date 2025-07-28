"""
Much of this code was taken/adapted from MNE-ICALabel

BSD 3-Clause License

Copyright (c) 2022, MNE
All rights reserved.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def eeg_psd(
    signal: NDArray[np.float64],
    sfreq: float,
    logscale: bool = True,
    normalize: bool = True,
) -> NDArray[np.float32]:
    """PSD feature.

    `signal.shape = (n_chan, n_points)`, with `n_chan` and
    `n_points` being the number of channels and time points of
    the signal.
    """

    n_chan, n_points = signal.shape
    constants = _psd_constants(n_points, sfreq, n_chan)
    psd = _psd_compute_psdmed(signal, sfreq, *constants, logscale=logscale)
    psd = _psd_format(psd, logscale=logscale, normalize=normalize)
    return psd


def _psd_constants(
    n_points: int,
    sfreq: float,
    n_chan: int,
) -> Tuple[
    int, int, int, int, NDArray[np.int32], NDArray[np.float64], NDArray[np.int32]
]:
    """Compute the constants before ``randperm`` is used to compute the subset."""
    # in MATLAB, 'pct_data' variable is never provided and is always initialized
    # to 100. 'pct_data' is only used in a division by 100.. and thus has no
    # impact and is omitted here.
    # in MATLAB, 'nfreqs' variable is always provided as 100 to this function,
    # thus it is either equal to 100 or to the nyquist frequency depending on
    # the nyquist frequency.

    nyquist = np.floor(sfreq / 2).astype(int)
    n_freqs = nyquist if nyquist < 100 else 100  # XXX: parametrize max freq.?

    n_points_ = min(n_points, int(sfreq))
    window = np.hamming(n_points_)
    cutoff = np.floor(n_points / n_points_) * n_points_

    # python is 0-index while matlab is 1-index, thus (1:n_points) becomes
    # np.arange(0, n_points) since 'index' is used to select from arrays.
    range_ = np.ceil(np.arange(0, cutoff - n_points_ + n_points_ / 2, n_points_ / 2))
    index = np.tile(range_, (n_points_, 1)).T + np.arange(0, n_points_)
    index = index.T.astype(int)

    n_seg = index.shape[1]

    # in MATLAB: 'subset = randperm(n_seg, ceil(n_seg * pct_data / 100));'
    # which is basically: 'subset = randperm(n_seg, n_seg);'
    # np.random.seed() can be used to fix the seed to the same value as MATLAB,
    # but since the 'randperm' equivalent in numpy does not exist, it is not
    # possible to reproduce the output in python.
    # 'subset' is used to select from arrays and is 0-index in Python while its
    # 1-index in MATLAB.
    subset = np.random.permutation(range(n_seg))  # 0-index

    return n_chan, n_freqs, n_points_, nyquist, index, window, subset


def _psd_compute_psdmed(
    signal: NDArray[np.float64],
    sfreq: float,
    n_chan: int,
    n_freqs: int,
    n_points: int,
    nyquist: int,
    index: NDArray[np.int32],
    window: NDArray[np.float64],
    subset: NDArray[np.int32],
    logscale: bool = True,
) -> NDArray[np.float64]:
    """Compute the variable 'psdmed', annotated as windowed spectrums."""
    denominator = sfreq * np.sum(np.power(window, 2))
    psdmed = np.zeros((n_chan, n_freqs))
    for it in range(n_chan):
        # Compared to MATLAB, shapes differ as the component dimension (size 1)
        # was squeezed.

        temp = np.hstack([signal[it, index[:, k]] for k in range(index.shape[-1])])
        temp = temp.reshape(*index.shape, order="F")
        # equivalent to:
        # np.vstack([icaact[it, index[:, k]] for k in range(index.shape[-1])]).T
        temp = (temp[:, subset].T * window).T
        temp = np.fft.fft(temp, n_points, axis=0)
        temp = temp * np.conjugate(temp)
        temp = temp[1 : n_freqs + 1, :] * 2 / denominator
        if n_freqs == nyquist:
            temp[-1, :] = temp[-1, :] / 2
        if logscale:
            psdmed[it, :] = 20 * np.real(np.log10(np.median(temp, axis=1)))
        else:  # linear scale
            psdmed[it, :] = np.real(np.median(temp, axis=1))

    return psdmed


def _psd_format(
    psd: NDArray[np.float64],
    logscale: bool = True,
    normalize: bool = True,
) -> NDArray[np.float32]:
    """Apply the formatting steps after 'eeg_rpsd.m'."""
    # extrapolate or prune as needed
    nfreq = psd.shape[1]
    if nfreq < 100:  # XXX: parametrize?
        psd = np.concatenate([psd, np.tile(psd[:, -1:], (1, 100 - nfreq))], axis=1)

    # undo notch filter
    for linenoise_ind in (50, 60):
        # 'linenoise_ind' is used for array selection in psd, which is
        # 0-index in Python and 1-index in MATLAB.
        linenoise_ind -= 1
        linenoise_around = np.array([linenoise_ind - 1, linenoise_ind + 1])
        # 'linenoise_around' is used for array selection in psd, which is
        # 0-index in Python and 1-index in MATLAB.
        difference = (psd[:, linenoise_around].T - psd[:, linenoise_ind]).T
        if logscale:
            notch_ind = np.all(5 < difference, axis=1)
        else:  # linear scale
            notch_ind = np.all(10 ** (5 / 20) < difference, axis=1)
        if any(notch_ind):
            # Numpy doesn't like the selection '[notch_ind, linenoise_ind]' with
            # 'notch_ind' as a bool mask. 'notch_ind' is first converted to int.
            # Numpy doesn't like the selection '[notch_ind, linenoise_around]'
            # with both defined as multi-values 1D arrays (or list). To get
            # around, the syntax [notch_ind[:, None], linenoise_around] is used.
            # That syntax works only with arrays (not list).
            notch_ind = np.where(notch_ind)[0]
            psd[notch_ind, linenoise_ind] = np.mean(
                psd[notch_ind[:, None], linenoise_around], axis=-1
            )

    if normalize:
        # min-max scaling normalization
        psd_min = np.min(psd, axis=-1, keepdims=True)
        psd_max = np.max(psd, axis=-1, keepdims=True)
        psd = 2 * (psd - psd_min) / (psd_max - psd_min) - 1  # range is -1 to 1

        # cast
        return 0.99 * psd.astype(np.float32)
