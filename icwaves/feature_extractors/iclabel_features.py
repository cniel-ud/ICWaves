from typing import Optional
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


def get_iclabel_features_per_segment(
    signal: NDArray[np.float64],
    sfreq: float,
    use_autocorr: bool = True,
    segment_len: Optional[int] = None,
) -> NDArray[np.float32]:
    n_time_series, time_series_len = signal.shape

    # TODO: parametrize this inside eeg_psd and eeg_autocorr
    n_features = 200

    if segment_len is None:
        segment_len = time_series_len

    n_segments = time_series_len // segment_len
    features = np.zeros((n_segments, n_time_series, n_features), dtype=np.float32)
    for segment_ind in range(n_segments):
        start = segment_ind * segment_len
        end = start + segment_len
        segment = signal[:, start:end]
        features[segment_ind] = get_iclabel_features(segment, sfreq, use_autocorr)

    # reshape `features` to (n_time_series, n_segments, n_features)
    return features.transpose((1, 0, 2))
