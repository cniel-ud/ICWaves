from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly


def _next_power_of_2(x) -> int:
    """
    Equivalent to 2^nextpow2 in MATLAB.
    """
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def eeg_autocorr(
    signal: NDArray[np.float64],
    sfreq: float,
) -> NDArray[np.float32]:

    _, n_points_ = signal.shape

    if n_points_ / sfreq > 5:
        autocorr = _eeg_autocorr_welch(signal, sfreq)
    else:
        autocorr = _eeg_autocorr(signal, sfreq)

    return autocorr


def _eeg_autocorr_welch(
    signal: NDArray[np.float64],
    sfreq: float,
) -> NDArray[np.float32]:
    """Autocorrelation feature applied on raw object with at least 5 * fs samples.

    MATLAB: 'eeg_autocorr_welch.m'.
    """

    # in MATLAB, 'pct_data' variable is never provided and is always initialized
    # to 100. 'pct_data' is only used in an 'if' statement reached if 'pct_data'
    # is different than 100.. thus, 'pct_data' is not used by this autocorrelation
    # function and is omitted here.

    # setup constants
    n_chan, n_points_ = signal.shape
    n_points = min(n_points_, int(sfreq * 3))
    nfft = _next_power_of_2(2 * n_points - 1)
    cutoff = np.floor(n_points_ / n_points) * n_points
    range_ = np.ceil(np.arange(0, cutoff - n_points + n_points / 2, n_points / 2))
    index = np.tile(range_, (n_points, 1)).T + np.arange(0, n_points)
    # python uses 0-index and matlab uses 1-index
    # python is 0-index while matlab is 1-index, thus (1:n_points) becomes
    # np.arange(0, n_points) since 'index' is used to select from arrays.
    index = index.T.astype(int)

    # separate data segments
    temp = np.hstack([signal[:, index[:, k]] for k in range(index.shape[-1])])
    segments = temp.reshape(n_chan, *index.shape, order="F")

    """
    # Just in case, here is the 'if' statement when 'pct_data' is different
    # than 100.

    n_seg = index.shape[1]
    # In MATLAB: n_seg = size(index, 2) * EEG.trials;
    # However, this function is only called on RAW dataset with EEG.trials
    # equal to 1.

    # in MATLAB: 'subset = randperm(n_seg, ceil(n_seg * pct_data / 100));'
    # which is basically: 'subset = randperm(n_seg, n_seg);'
    # np.random.seed() can be used to fix the seed to the same value as MATLAB,
    # but since the 'randperm' equivalent in numpy does not exist, it is not
    # possible to reproduce the output in python.
    # 'subset' is used to select from arrays and is 0-index in Python while its
    # 1-index in MATLAB.
    subset = np.random.permutation(range(n_seg))  # 0-index
    temp = np.hstack([icaact[:, index[:, k]] for k in range(index.shape[-1])])
    temp = temp.reshape(ncomp, *index.shape, order='F')
    segments = temp[:, :, subset]
    """

    # calc autocorrelation
    ac = np.zeros((n_chan, nfft))
    for it in range(n_chan):
        x = np.fft.fft(segments[it, :, :], nfft, axis=0)
        ac[it, :] = np.mean(np.power(np.abs(x), 2), axis=1)
    ac = np.fft.ifft(ac)

    # normalize
    # In MATLAB, 2 scenarios are defined:
    # - EEG.pnts < EEG.srate, which never occurs since then raw provided to
    # this autocorrelation function last at least 5 second.
    # - EEG.pnts > EEG.srate, implemented below.
    ac = ac[:, : int(sfreq) + 1]
    # build the (3-line!) denominator
    arr1 = np.arange(n_points, n_points - int(sfreq), -1)
    arr1 = np.hstack([arr1, [np.max([1, n_points - int(sfreq)])]])
    den = np.tile(ac[:, 0], (arr1.size, 1))
    den = den.T * arr1 / n_points
    # finally..
    ac = np.divide(ac, den)

    # resample to 1 second at 100 samples/sec
    resamp = _resample(ac, sfreq)
    resamp = resamp[:, 1:]
    return 0.99 * np.real(resamp).astype(np.float32)


def _resample(ac: NDArray[np.float64], sfreq: Union[int, float]) -> NDArray[np.float64]:
    """Resample the autocorrelation feature.

    The comment in EEGLAB is:
        resample to 1 second at 100 samples/sec

    Which translates by: the output array must be of shape (n_comp, 101), thus
    the resampling up variable is set to 100, and down variable must respect:
        100 < ac.T.shape[0] * 100 / down <= 101
    If the instance sampling frequency is an integer, then down is equal to the
    sampling frequency.

    Parameters
    ----------
    ac : array
        Array of shape (n_comp, samples).
    fs : int | float
        Sampling frequency of the MNE instance.
    """
    down = int(sfreq)
    if 101 < ac.shape[1] * 100 / down:
        down += 1
    return resample_poly(ac.T, 100, down).T


def _eeg_autocorr(
    signal: NDArray[np.float64],
    sfreq: float,
) -> NDArray[np.float32]:
    """Autocorr applied on raw object without enough sampes for eeg_autocorr_welch.

    MATLAB: 'eeg_autocorr.m'.
    """

    # in MATLAB, 'pct_data' variable is neither provided or used, thus it is
    # omitted here.
    n_chan, n_points_ = signal.shape
    nfft = _next_power_of_2(2 * n_points_ - 1)

    c = np.zeros((n_chan, nfft))
    for it in range(n_chan):
        # in MATLAB, 'mean' does nothing here. It looks like it was included
        # for a case where epochs are provided, which never happens with this
        # autocorrelation function.
        x = np.power(np.abs(np.fft.fft(signal[it, :], n=nfft)), 2)
        c[it, :] = np.real(np.fft.ifft(x))

    if n_points_ < sfreq:
        zeros = np.zeros((c.shape[0], int(sfreq) - n_points_ + 1))
        ac = np.hstack([c[:, :n_points_], zeros])
    else:
        ac = c[:, : int(sfreq) + 1]

    # normalize by 0-tap autocorrelation
    ac = np.divide(ac.T, ac[:, 0]).T

    # resample to 1 second at 100 samples/sec
    resamp = _resample(ac, sfreq)
    resamp = resamp[:, 1:]
    return 0.99 * resamp.astype(np.float32)
