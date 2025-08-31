# imports
import scipy
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, sosfilt, welch, freqz, sosfreqz, filtfilt, lfilter
from scipy.fft import rfft, rfftfreq, irfft
from typing import List, Tuple

from generate_cmmn_filters import compute_filter_original


def main():
    frolich_subj_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']


    frolich_filepath = Path('../data/frolich_256/frolich_extract_256_hz')


    frolich_data = []
    for subj in frolich_subj_list:
        frolich_data.append(loadmat(frolich_filepath / f'frolich_extract_{subj}_256_hz.mat')['X'])


    # compute filters and save them
    save_path = Path('../data/frolich_filters')
    save_path.mkdir(parents=True, exist_ok=True)

    with np.load(save_path / 'emotion_unnormed_ch_avg_barycenter.npz') as saved_data:
        unnormed_emotion_barycenter = saved_data['arr_0']

    # compute original filter
    # unnormed psds, normed barycenter
    freq_filter, time_filter = compute_filter_original(frolich_data, unnormed_emotion_barycenter)

    # save filters
    print('Saving filters for frolich original method...')
    for i, filter in enumerate(freq_filter):
        np.savez(save_path / f'frolich_original_ch_avg_barycenter_freq_filter_unnormed_{frolich_subj_list[i]}.npz', filter)

    for i, filter in enumerate(time_filter):
        np.savez(save_path / f'frolich_original_ch_avg_barycenter_time_filter_unnormed_{frolich_subj_list[i]}.npz', filter)




if __name__ == "__main__":
    main()
