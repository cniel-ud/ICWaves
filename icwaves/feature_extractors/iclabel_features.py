import numpy as np
from numpy.typing import NDArray

from icwaves.feature_extractors.autocorr import eeg_autocorr
from icwaves.feature_extractors.psd import eeg_psd


def get_iclabel_features(
    signal: NDArray[np.float64],
    sfreq: float,
    use_autocorr: bool = True,
) -> NDArray[np.float32]:

    out = eeg_psd(signal, sfreq)

    if use_autocorr:
        autocorr = eeg_autocorr(signal, sfreq)
        out = np.hstack((out, autocorr))

    return out





