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
    if out.size == 0:
        return out

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

    n_features = 200 if use_autocorr else 100

    if segment_len is None:
        segment_len = time_series_len

    n_segments = time_series_len // segment_len
    features = np.zeros((n_segments, n_time_series, n_features), dtype=np.float32)
    for segment_ind in range(n_segments):
        start = segment_ind * segment_len
        end = start + segment_len
        segment = signal[:, start:end]
        feature = get_iclabel_features(segment, sfreq, use_autocorr)
        if feature.size > 0:
            features[segment_ind] = feature

    # drop segments that are all-zeros
    all_zeros = np.all(features == 0.0, axis=(1, 2))
    features = features[~all_zeros]

    # reshape `features` to (n_time_series, n_segments, n_features)
    features = features.transpose((1, 0, 2))

    # raise an error if all segments of a time series are zero
    all_zeros = np.all(features == 0.0, axis=(1, 2))
    if all_zeros.any():
        raise ValueError("There are all-zero feature vectors")

    return features
