# imports
import scipy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, sosfilt, welch, freqz, sosfreqz, filtfilt, lfilter
from scipy.fft import rfft, rfftfreq, irfft
from typing import List, Tuple

# self contained script for caviness job. see sbatch file

# all helper functions from gdrive notebook here. I know this is messy right now

def psd(data, fs=256, nperseg=256):
    """
    Compute the Power Spectral Density (PSD) of the given data.

    Parameters:
    - data: list of numpy arrays, each containing EEG data for a subject
    - fs: sampling frequency (default: 256 Hz)
    - nperseg: length of each segment for Welch's method (default: 256)

    Returns:
    - f: array of sample frequencies (x-axis)
    - Pxx: power spectral density of the data (y-axis)
    """

    f, Pxx = welch(data, fs=fs, nperseg=nperseg)
    return f, Pxx

def compute_normed_barycenter(data, psds=None):
  """

  """

  normalized_psds = []
  if psds is None:
    psds = []
  for i, subj in enumerate(data):
      if psds is None:
          f, Pxx = psd(subj)
          psds.append(Pxx)
          normalized_psds.append(Pxx / np.sum(Pxx))
      else:
          normalized_psds.append(psds[i] / np.sum(psds[i]))

  # now average all together
  per_subj_avgs = []
  for subj in normalized_psds:
      avg = np.mean(subj, axis=0)
      per_subj_avgs.append(np.mean(subj, axis=0)) # necessary due to inhomogenous dimensions

  barycenter = np.mean(per_subj_avgs, axis=0)

  return barycenter

def compute_filter_original(data, barycenter, psds=None):
  """
  Compute the filter to transform the given data to the barycenter.

  Parameters:
  - data: list of numpy arrays, each containing EEG data for a subject in the target data (cue / frolich here)
  - barycenter: numpy array containing the normed barycenter of the data

  Returns:
  - freq_filter: numpy array containing the filter in the frequency domain
  - time_filter: numpy array containing the filter in the time domain
  """

  if psds is None:
    psds = []
    for subj in data:
        f, Pxx = psd(subj)
        psds.append(Pxx)

  # here we collapse channels down. our paradigm does not include channel - channel correspondence between domains
  avg_psds_per_subj = [
    np.mean(subj, axis=0) for subj in psds
  ]

  # computing filters now. irfft
  freq_filter_per_subj = []
  for avg_psd_per_subj in avg_psds_per_subj:
    freq_filter = np.sqrt(barycenter) / np.sqrt(avg_psd_per_subj)
    freq_filter_per_subj.append(freq_filter)

  time_filter_per_subj = []
  for subj in freq_filter_per_subj:
    time_filter = np.fft.irfft(subj) # note, now with real values only.
    time_filter_per_subj.append(time_filter)

  return freq_filter_per_subj, time_filter_per_subj

def subj_subj_matching(source_psds, target_psds) -> List[int]:
  """
  Subsidiary function for subj-subj filter computation function below.

  Parameters
  ----------
  source_psds: list of numpy arrays, each containing the PSD of the data for a subject in the source domain. shape (n_channels, n_freqs)
  target_psds: list of numpy arrays, each containing the PSD of the data for a subject in the target domain. shape (n_channels, n_freqs)

  Returns
  -------
  A list of indices, where the i-th element is the index of the source subject that best matches the i-th target subject.
  Ex: target subj 1 and 2 match to source domain 1, target subj 3 matches to source domain 2: [1, 1, 2]
  """

  # average both right away. we don't assume that domains must match # of channels, so do dist matching based on
  correct_source_psds = [np.mean(subj, axis=0) for subj in source_psds]
  correct_target_psds = [np.mean(subj, axis=0) for subj in target_psds]

  # pre-normalized length output. for debug
  # for i, target_psd in enumerate(correct_target_psds):
  #   print(f"For frolich subject {i}, the PSD has length { (1/np.sqrt(2)) * np.sqrt(np.sum(target_psd**2))}")

  # for i, source_psd in enumerate(correct_source_psds):
  #   print(f"For emotion subject {i}, the PSD has length { (1/np.sqrt(2)) * np.sqrt(np.sum(source_psd**2))}")


  # Normalizing so that Hellinger dist makes sense
  # make them sum to one
  for i, target_psd in enumerate(correct_target_psds):
    correct_target_psds[i] = target_psd / np.sum(target_psd)
  for i, source_psd in enumerate(correct_source_psds):
    correct_source_psds[i] = source_psd / np.sum(source_psd)

  # now check again

  # print dist to check normalization. for debug
  # for i, target_psd in enumerate(correct_target_psds):
  #   print(f"For frolich subject {i}, the PSD has length {np.sqrt(np.sum(target_psd**2))}")

  # for i, source_psd in enumerate(correct_source_psds):
  #   print(f"For emotion subject {i}, the PSD has length {np.sqrt(np.sum(source_psd**2))}")

  subj_subj_matches = []
  for i, target_psd in enumerate(correct_target_psds): # frolich
    min_dist = float('inf')
    min_idx = -1
    for j, source_psd in enumerate(correct_source_psds): # emotion
      # dist = np.sqrt(np.sum((np.sqrt(source_psd) - np.sqrt(target_psd))**2)) # just L2
      dist = (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(source_psd) - np.sqrt(target_psd))**2)) # Hellinger

      # print("distance: ", dist)

      if dist < min_dist:
        # print("distance was lower than min_dist of: ", min_dist)
        min_dist = dist
        min_idx = j
    subj_subj_matches.append(min_idx)

    print(f"Frolich subject {i} has been matched to Emotion subject {min_idx}")

  return subj_subj_matches

def compute_filter_subj_subj(target_psds, source_psds, subj_subj_matches):
  """
  Compute the filter to transform the given data to the barycenter, with subject to subject matching.

  Parameters:
  - target_psds: list of numpy arrays, each containing the PSD of the data for a subject in the target domain. shape (n_channels, n_freqs)
  - source_psds: list of numpy arrays, each containing the PSD of the data for a subject in the source domain. shape (n_channels, n_freqs)
  - subj_subj_matches: list of indices, where the i-th element is the index of the source subject that best matches the i-th target subject.

  Returns:
  - freq_filter: numpy array containing the filter in the frequency domain for each target subject
  - time_filter: numpy array containing the filter in the time domain for each target subject
  """


  freq_filter_per_subj = []
  time_filter_per_subj = []


  averaged_source_psds = []
  averaged_target_psds = []

  # do not assume that there is a channel - channel correspondence, etc. collapse to average PSD
  for psd in source_psds:
    averaged_source_psds.append(np.mean(psd, axis=0))
  for psd in target_psds:
    averaged_target_psds.append(np.mean(psd, axis=0))


  for i, subj in enumerate(averaged_target_psds):
    source_psd = averaged_source_psds[subj_subj_matches[i]]

    freq_filter = np.sqrt(source_psd) / np.sqrt(subj) # sqrt of emotion psd / sqrt of frolich psd
    time_filter = np.fft.irfft(freq_filter) # note: changed from ifft to irfft

    freq_filter_per_subj.append(freq_filter)
    time_filter_per_subj.append(time_filter)

  return freq_filter_per_subj, time_filter_per_subj

def transform_original(data, time_filter):
  """
  Transform the given data using the given filter.

  Parameters:
  - data: list of numpy arrays, each containing EEG data for a subject in the target data
  - time_filter: list containing the filter in the time domain for each subject

  Returns:
  - transformed_data: list of numpy arrays, each containing the transformed EEG data for a subject
  """

  # Do the transformation on a channel basis

  transformed_data = []
  for i, subj in enumerate(data):
    num_channels = subj.shape[0]
    subj_norm = np.zeros_like(subj)

    for chan in range(num_channels):

      #subj_norm[chan] = np.convolve(subj[chan], time_filter[i], mode='same') # note dr. b wants changed to full b/c of weird reverse shifting
      subj_norm[chan] = np.convolve(subj[chan], time_filter[i], mode='full')[:len(subj[chan])]

    transformed_data.append(subj_norm)

  return transformed_data

def transform_data_subj_subj(data, time_filter_subj_subj):
  """
  Transform data with knowledge that the time filter is a list, with each specific to the individual target domain subject

  Parameters
  ----------
  data
  time_filter_subj_subj

  Returns
  -------

  """

  transformed_data = []
  for i, subj in enumerate(data):
    subj_norm = np.zeros(subj.shape)
    num_channels = subj.shape[0]
    time_filter = time_filter_subj_subj[i] # subj specific time filter

    for chan in range(num_channels):
      # print(subj[chan].shape)
      # print(time_filter[i].shape)
      # subj_norm[chan] = np.convolve(subj[chan], time_filter[chan], mode='same') # here also change this to full, matching the above
      subj_norm[chan] = np.convolve(subj[chan], time_filter[chan], mode='full')[:len(subj[chan])]

    transformed_data.append(subj_norm)

  return transformed_data






# now actual script material

def main(make_psds=False):
    """
    If first time running, set make_psds to True. Otherwise don't re-generate every time.
    """

    # note emotion dataset missing 22
    emotion_subj_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '23',
        '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
    frolich_subj_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']


    emotion_filepath = Path('../data/emotion_256/raw_data_and_IC_labels')
    frolich_filepath = Path('../data/frolich_256/frolich_extract_256_hz')

    # grab raw data
    emotion_data = []
    for subj in emotion_subj_list:
    emotion_data.append(loadmat(emotion_filepath / f'subj-{subj}.mat')['data'])

    frolich_data = []
    for subj in frolich_subj_list:
    frolich_data.append(loadmat(frolich_filepath / f'frolich_extract_{subj}_256_hz.mat')['X'])

    if make_psds:
    (emotion_filepath / 'psds').mkdir(parents=True, exist_ok=True)
    (frolich_filepath / 'psds').mkdir(parents=True, exist_ok=True)

    for i, subj in enumerate(emotion_data):
        f, Pxx = psd(subj)
        np.savez(emotion_filepath / 'psds' / f'subj-{emotion_subj_list[i]}_psds', Pxx)

    for i, subj in enumerate(frolich_data):
        f, Pxx = psd(subj)
        np.savez(frolich_filepath / 'psds' / f'frolich_extract_{frolich_subj_list[i]}_256_hz_psds', Pxx)

    emotion_data_psds_raw = []
    frolich_data_psds_raw = []

    if (emotion_filepath / 'psds').exists():
    for subj in emotion_subj_list:
        emotion_data_psds_raw.append(np.load(emotion_filepath / 'psds' / f'subj-{subj}_psds.npz')['arr_0'])

    if (frolich_filepath / 'psds').exists():
    for subj in frolich_subj_list:
        frolich_data_psds_raw.append(np.load(frolich_filepath / 'psds' / f'frolich_extract_{subj}_256_hz_psds.npz')['arr_0'])


    # compute normed barycenter
    normed_emotion_barycenter = compute_normed_barycenter(emotion_data, psds=emotion_data_psds_raw)

    # compute filters and save them
    save_path = Path('../data/frolich_filters')
    save_path.mkdir(parents=True, exist_ok=True)

    # compute original filter
    freq_filter, time_filter = compute_filter_original(frolich_data, normed_emotion_barycenter)

    # save filters
    for i, filter in enumerate(freq_filter):
        np.savez(save_path / f'frolich_original_freq_filter_{frolich_subj_list[i]}.npz', filter)

    for i, filter in enumerate(time_filter):
        np.savez(save_path / f'frolich_original_time_filter_{frolich_subj_list[i]}.npz', filter)

    
    # compute subj-subj filter
    subj_subj_matches = subj_subj_matching(emotion_data_psds_raw, frolich_data_psds_raw)
    freq_filter_subj_subj, time_filter_subj_subj = compute_filter_subj_subj(frolich_data_psds_raw, emotion_data_psds_raw, subj_subj_matches)

    # save subj-subj filters
    for i, filter in enumerate(freq_filter_subj_subj):
        np.savez(save_path / f'frolich_subj_subj_freq_filter_{frolich_subj_list[i]}.npz', filter)

    for i, filter in enumerate(time_filter_subj_subj):
        np.savez(save_path / f'frolich_subj_subj_time_filter_{frolich_subj_list[i]}.npz', filter)



if __name__ == "__main__":
    main(make_psds=True)
