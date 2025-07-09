# %%
"""
Dr. B has recommended to do the notebook here in colab so that we all have the same data and can easily run it. I think it's a good idea so I'm porting it over now. Still saving to the git repo though.

Later, once this is complete, will refresh on the newly refactored codebase to do BoW on this and then Carlos can test it with his shiny new clfs.

Right now this is finished, currently setting up a Caviness job to run through everything.


Post NER note - moved to notebooks, make sure that the filepaths work correclty when re-visualized. 
"""

# %%

# %%
# imports
import scipy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, sosfilt, welch, freqz, sosfreqz, filtfilt, lfilter
from scipy.fft import rfft, rfftfreq, irfft
from typing import List, Tuple





# %%
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

# Define a function to plot PSD
def plot_psd(data, fs=256, nperseg=256, psds=None, title='PSD', save_path=None):
    """
    Plot the Power Spectral Density (PSD) of the given data.

    Note: right now this is straight averaging, not the barycenter norming I do later.
    If I want to do the norming, apply that first and then this will not average for me.

    Parameters:
    - data: list of numpy arrays, each containing EEG data for a subject
    - fs: sampling frequency (default: 256 Hz)
    - nperseg: length of each segment for Welch's method (default: 256)
    - save_path: if provided, save the plot to this path as PDF
    """
    plt.figure(figsize=(12, 8))

    for i, subj_data in enumerate(data):
        # If subj_data is multi-dimensional, average across the channels

        """
        Note, the below two are different because in the first I average the channels before taking the PSD,
        and in the second I average the channels after taking the PSD.

        Maybe should ask this to Dr. B and Carlos. Which is better since they do seem to result in some differences

        In the rest, when calculating the barycenter and computing the filter, I average AFTER computing the PSD. Follow that convention.

        Clean below later, but keep it as is for now to ask Dr. B and Carlos about it in the meeting later.
        """
        if psds is None:
            # If subj_data is multi-dimensional, average across the channels
            if subj_data.ndim > 1:
                subj_data = np.mean(subj_data, axis=0)
            f, Pxx = psd(subj_data, fs=fs, nperseg=nperseg)

            plt.plot(f, 10 * np.log10(Pxx), label=f'Subject {i+1}')
        else:
            viz_psds = [np.zeros_like(psd) for psd in psds]
            if psds[i].ndim > 1:
                viz_psds[i] = np.mean(psds[i], axis=0)

            f = np.linspace(0, 128, 129)

            plt.plot(f, 10 * np.log10(viz_psds[i]), label=f'Subject {i+1}')


    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB/Hz)')
    plt.title(title)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved plot to {save_path}")
    
    plt.show()

# %%
def plot_raw_signals(data, fs=256, title='Raw Signals', all_channels=False, save_path=None):
  """
  Plot the raw signals of the given data.

  Parameters:
  - data: list of numpy arrays, each containing EEG data for a subject
  - fs: sampling frequency (default: 256 Hz)
  - save_path: if provided, save the plot to this path as PDF
  """
  plt.figure(figsize=(12, 8))

  for i, subj_data in enumerate(data):
    if all_channels:
      # Check if subj_data has multiple channels
      if subj_data.ndim > 1:
        for channel in range(subj_data.shape[1]):
          plt.plot(np.arange(subj_data.shape[0]) / fs, subj_data[:, channel])#, label=f'Subject {i+1}, Channel {channel+1}')
      else:
        # If subj_data is 1-dimensional, plot it directly
        plt.plot(np.arange(subj_data.shape[0]) / fs, subj_data, label=f'Subject {i+1}, Channel 1')
    else:
      # average all channels together
      subj_data = np.mean(subj_data, axis=0) # changed from 1 to 0

      # now plot
      plt.plot(np.arange(subj_data.shape[0]) / fs, subj_data, label=f'Subject {i+1}')

  plt.title(title)
  plt.legend()
  plt.ylabel('Voltage (uV)')
  plt.xlabel('Time (s)')
  
  if save_path:
      plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                 facecolor='white', edgecolor='none')
      print(f"Saved plot to {save_path}")
  
  plt.show()

# %%
def plot_polysomnograph(data, channel_indices, fs=256, time_window=None, spacing_factor=2.0, title=None, save_path=None):
    """
    Plot EEG data in a polysomnograph-style format with stacked channels for a single subject.
    Parameters:
    - data: numpy array of shape (n_channels, n_samples) containing EEG data for one subject
    - channel_indices: list of channel indices to plot (e.g., [1,2,3,15,16,63,64])
    - fs: sampling frequency (default: 256 Hz)
    - time_window: tuple of (start_time, end_time) in seconds. If None, plots all data.
    - spacing_factor: factor to adjust vertical spacing between channels (default: 2.0)
    - title: plot title (optional)
    - save_path: if provided, save the plot to this path as PDF
    """
    plt.figure(figsize=(15, len(channel_indices) * 1.5))    # Apply time window if specified
    if time_window is not None:
        start_sample = int(time_window[0] * fs)
        end_sample = int(time_window[1] * fs)
        data = data[:, start_sample:end_sample]    # Calculate time vector
    time = np.arange(data.shape[1]) / fs    # Plot each channel
    for i, chan_idx in enumerate(channel_indices):
        # Get channel data (subtract 1 from index since data is 0-indexed)
        channel = data[chan_idx - 1]        # Normalize the channel
        normalized_channel = channel / np.max(np.abs(channel))        # Plot with offset
        offset = -i * spacing_factor
        plt.plot(time, normalized_channel + offset, label=f'Channel {chan_idx}')

    plt.xlabel('Time (s)')
    plt.yticks([])  # Remove y-axis ticks since they're arbitrary
    if title:
        plt.title(title)    # Add channel labels on the y-axis
    channel_positions = [-i * spacing_factor for i in range(len(channel_indices))]
    plt.yticks(channel_positions, [f'Ch {idx}' for idx in channel_indices])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved plot to {save_path}")
    
    plt.show()



# %%
def compute_normed_barycenter(data, psds=None):
  """

  """

  # Original approach - commented out
  # normalized_psds = []
  # if psds is None:
  #   psds = []
  # for i, subj in enumerate(data):
  #     if psds is None:
  #         f, Pxx = psd(subj)
  #         psds.append(Pxx)
  #         normalized_psds.append(Pxx / np.sum(Pxx))
  #     else:
  #         normalized_psds.append(psds[i] / np.sum(psds[i]))

  # # now average all together
  # per_subj_avgs = []
  # for subj in normalized_psds:
  #     avg = np.mean(subj, axis=0)
  #     per_subj_avgs.append(np.mean(subj, axis=0)) # necessary due to inhomogenous dimensions

  # barycenter = np.mean(per_subj_avgs, axis=0)

  # New approach - L1 normalization after channel averaging
  if psds is None:
    psds = []
    for subj in data:
        f, Pxx = psd(subj)
        psds.append(Pxx)

  # First average across channels for each subject
  per_subj_avgs = []
  for subj in psds:
      avg = np.mean(subj, axis=0)
      per_subj_avgs.append(avg)

  # Then do L1 normalization on each subject's averaged PSD
  normalized_psds = []
  for subj_avg in per_subj_avgs:
      normalized_psds.append(subj_avg / np.sum(subj_avg))

  # Finally average across subjects
  barycenter = np.mean(normalized_psds, axis=0)

  return barycenter


def plot_barycenter(barycenter, title='Normed Barycenter', save_path=None):
  """
  Plot the normed barycenter of the given data.

  Parameters:
  - barycenter: numpy array containing the normed barycenter of the data
  - save_path: if provided, save the plot to this path as PDF
  """
  plt.figure(figsize=(12, 8))
  f = np.linspace(0, 128, 129)
  plt.plot(f, 10 * np.log10(barycenter))

  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Power Spectral Density (dB/Hz)')
  plt.title(title)
  
  if save_path:
      plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                 facecolor='white', edgecolor='none')
      print(f"Saved plot to {save_path}")
  
  plt.show()

# %%
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




# also some plotting funcs
def plot_freq_filter(freq_filter, fs=256, title='Frequency Filters', save_path=None):
  """
  Plot the filter in the frequency domain.

  Parameters:
  - freq_filter: numpy array containing the filter in the frequency domain
  - fs: sampling frequency (default: 256 Hz)
  - title: plot title (default: 'Frequency Filters')
  - save_path: if provided, save the plot to this path as PDF

  """

  # for subj subj matching, the freq and time filters have channel components still.
  # average before plotting if so

  if freq_filter[0].ndim > 1:
    freq_filter = [np.mean(subj, axis=0) for subj in freq_filter]

  plt.figure(figsize=(12, 8))
  f = np.linspace(0, fs/2, freq_filter[0].shape[-1])
  for i, subj in enumerate(freq_filter):
      plt.plot(f, 10 * np.log10(np.abs(subj.T)), label=f'Subject {i+1}')

  plt.xlabel('Frequency (Hz)')
  plt.ylabel('Filter (dB)')
  plt.title(title)
  plt.legend()
  
  if save_path:
      plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                 facecolor='white', edgecolor='none')
      print(f"Saved plot to {save_path}")
  
  plt.show()

def plot_time_filter(time_filter, fs=256, title='Time Domain Filter', save_path=None):
  """
  Plot the filter in the time domain.

  Parameters:
  - time_filter: numpy array containing the filter in the time domain
  - fs: sampling frequency (default: 256 Hz)
  - title: plot title (default: 'Time Domain Filter')
  - save_path: if provided, save the plot to this path as PDF
  """

  # for subj subj matching, the freq and time filters have channel components still.
  # average before plotting if so

  if time_filter[0].ndim > 1:
    time_filter = [np.mean(subj, axis=0) for subj in time_filter]


  plt.figure(figsize=(12, 8))
  t = np.arange(len(time_filter[0])) / fs
  for i, subj in enumerate(time_filter):
      plt.plot(t, np.real(subj), label=f'Subject {i+1}')

  plt.xlabel('Time (s)')
  plt.ylabel('Amplitude')
  plt.title(title)
  plt.legend()
  
  if save_path:
      plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                 facecolor='white', edgecolor='none')
      print(f"Saved plot to {save_path}")
  
  plt.show()


# %%
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


def transform_data_subj_subj_single(subj_data, time_filter):
  """
  Transform data for a single subject using subject-specific time filter

  Parameters
  ----------
  subj_data : numpy array
    EEG data for a single subject, shape (n_channels, n_samples)
  time_filter : numpy array
    Time filter specific to this subject, shape (n_channels, filter_length)

  Returns
  -------
  subj_norm : numpy array
    Transformed EEG data for the subject, same shape as input
  """
  
  subj_norm = np.zeros(subj_data.shape)
  num_channels = subj_data.shape[0]

  for chan in range(num_channels):
    subj_norm[chan] = np.convolve(subj_data[chan], time_filter[chan], mode='full')[:len(subj_data[chan])]

  return subj_norm


def transform_original_single(subj_data, time_filter):
  """
  Transform the given single subject data using the given filter.

  Parameters:
  - subj_data: numpy array containing EEG data for a single subject, shape (n_channels, n_samples)
  - time_filter: numpy array containing the filter in the time domain for this subject

  Returns:
  - subj_norm: numpy array containing the transformed EEG data for the subject
  """

  num_channels = subj_data.shape[0]
  subj_norm = np.zeros_like(subj_data)

  for chan in range(num_channels):
    subj_norm[chan] = np.convolve(subj_data[chan], time_filter, mode='full')[:len(subj_data[chan])]

  return subj_norm







# ----------------------------------------------------
# above is util functions, below actually load and viz
# %%
# Full subject lists for HPC processing
# right now I put this at 4 for quick testing. change soon
emotion_subj_list = [f'{i:02d}' for i in range(1, 36) if i != 22]  # 01-35 minus 22
frolich_subj_list = [f'{i:02d}' for i in range(1, 13)]  # 01-12

# Output directory for PDF figures
output_dir = Path('./ner_figures')
output_dir.mkdir(parents=True, exist_ok=True)

# %%
make_psds = False # change to True if running the notebook for the first time.

# HPC file paths
data_dir = Path('../data')
emotion_filepath = data_dir / 'emotion_256' / 'raw_data_and_IC_labels'
frolich_filepath = data_dir / 'frolich_256' / 'frolich_extract_256_hz'

emotion_transformed_filepath = data_dir / 'emotion_256' / 'emotion_data_cmmn'
frolich_transformed_filepath = data_dir / 'frolich_256' / 'frolich_extract_256_hz_cmmn'
emotion_transformed_psd_filepath = data_dir / 'emotion_256' / 'emotion_data_cmmn' / 'psds'
frolich_transformed_subj_subj_psd_filepath = data_dir / 'frolich_256' / 'frolich_extract_256_hz_cmmn' / 'subj_subj_psds'
frolich_transformed_original_psd_filepath = data_dir / 'frolich_256' / 'frolich_extract_256_hz_cmmn' / 'original_psds'

# PSD file paths  
emotion_normed_psd_filepath = emotion_filepath / 'psds_normed'
frolich_normed_psd_filepath = frolich_filepath / 'psds_normed'

emotion_raw_psd_filepath = emotion_filepath / 'psds'
frolich_raw_psd_filepath = frolich_filepath / 'psds'

# Create all necessary directories
emotion_transformed_filepath.mkdir(parents=True, exist_ok=True)
frolich_transformed_filepath.mkdir(parents=True, exist_ok=True)
emotion_transformed_psd_filepath.mkdir(parents=True, exist_ok=True)
frolich_transformed_subj_subj_psd_filepath.mkdir(parents=True, exist_ok=True)
frolich_transformed_original_psd_filepath.mkdir(parents=True, exist_ok=True)

# grab raw data
emotion_data = []
for subj in emotion_subj_list:
  emotion_data.append(loadmat(emotion_filepath / f'subj-{subj}.mat')['data'])

frolich_data = []
for subj in frolich_subj_list:
  frolich_data.append(loadmat(frolich_filepath / f'frolich_extract_{subj}_256_hz.mat')['X'])

if make_psds:
  emotion_raw_psd_filepath.mkdir(parents=True, exist_ok=True)
  frolich_raw_psd_filepath.mkdir(parents=True, exist_ok=True)

  for i, subj in enumerate(emotion_data):
    f, Pxx = psd(subj)
    np.savez(emotion_raw_psd_filepath / f'subj-{emotion_subj_list[i]}_psds', Pxx)

  for i, subj in enumerate(frolich_data):
    f, Pxx = psd(subj)
    np.savez(frolich_raw_psd_filepath / f'frolich_extract_{frolich_subj_list[i]}_256_hz_psds', Pxx)

emotion_data_psds_raw = []
frolich_data_psds_raw = []

emotion_data_psds_normed = []
frolich_data_psds_normed = []

if (emotion_raw_psd_filepath).exists():
  for subj in emotion_subj_list:
    emotion_data_psds_raw.append(np.load(emotion_raw_psd_filepath / f'subj-{subj}_psds.npz')['arr_0'])

if (frolich_raw_psd_filepath).exists():
  for subj in frolich_subj_list:
    frolich_data_psds_raw.append(np.load(frolich_raw_psd_filepath / f'frolich_extract_{subj}_256_hz_psds.npz')['arr_0'])

# Load normed PSDs
if emotion_normed_psd_filepath.exists():
  for subj in emotion_subj_list:
    emotion_data_psds_normed.append(np.load(emotion_normed_psd_filepath / f'subj-{subj}_psds_normed.npz')['arr_0'])

if frolich_normed_psd_filepath.exists():
  for subj in frolich_subj_list:
    frolich_data_psds_normed.append(np.load(frolich_normed_psd_filepath / f'frolich_extract_{subj}_256_hz_psds_normed.npz')['arr_0'])




"""
## Frolich Filters Visualization

Load in the data filters for emotion -> emotion and visualize. Do four of them, similarly to the rest.
"""

# %%
# HPC filter paths
filters_filepath = data_dir / 'frolich_filters'

emotion_original_time_filters = []
emotion_original_freq_filters = []

for subj in emotion_subj_list:  # Load filters for all subjects
  time_filter_path = filters_filepath / f'emotion_normed_psds_normed_barycenter_time_filter_{subj}.npz'
  freq_filter_path = filters_filepath / f'emotion_normed_psds_normed_barycenter_freq_filter_{subj}.npz'
  
  if time_filter_path.exists():
    emotion_original_time_filters.append(np.load(time_filter_path)['arr_0'])
  if freq_filter_path.exists():
    emotion_original_freq_filters.append(np.load(freq_filter_path)['arr_0'])

if emotion_original_time_filters:
  plot_time_filter(emotion_original_time_filters, title='Emotion Barycenter Mapping Time Filters', save_path=output_dir / 'emotion_original_time_filters.pdf')
if emotion_original_freq_filters:
  plot_freq_filter(emotion_original_freq_filters, title='Emotion Barycenter Mapping Frequency Filters', save_path=output_dir / 'emotion_original_freq_filters.pdf')


frolich_original_time_filters = []
frolich_original_freq_filters = []
frolich_subj_subj_freq_filters = []
frolich_subj_subj_time_filters = []

for subj in frolich_subj_list:
  frolich_original_time_filters.append(np.load(filters_filepath / f'frolich_original_time_filter_{subj}.npz')['arr_0'])
  frolich_original_freq_filters.append(np.load(filters_filepath / f'frolich_original_freq_filter_{subj}.npz')['arr_0'])
  frolich_subj_subj_freq_filters.append(np.load(filters_filepath / f'frolich_subj_subj_freq_filter_{subj}.npz')['arr_0'])
  frolich_subj_subj_time_filters.append(np.load(filters_filepath / f'frolich_subj_subj_time_filter_{subj}.npz')['arr_0'])

if frolich_original_time_filters:
  plot_time_filter(frolich_original_time_filters, title='Cue Barycenter Mapping Time Filters', save_path=output_dir / 'frolich_original_time_filters.pdf')
if frolich_original_freq_filters:
  plot_freq_filter(frolich_original_freq_filters, title='Cue Barycenter Mapping Frequency Filters', save_path=output_dir / 'frolich_original_freq_filters.pdf')

# also plot frolich subj subj filters
if frolich_subj_subj_time_filters:
  plot_time_filter(frolich_subj_subj_time_filters, title='Cue Subj-to-subj Mapping Time Filters', save_path=output_dir / 'frolich_subj_subj_time_filters.pdf')
if frolich_subj_subj_freq_filters:
  plot_freq_filter(frolich_subj_subj_freq_filters, title='Cue Subj-to-subj Mapping Frequency Filters', save_path=output_dir / 'frolich_subj_subj_freq_filters.pdf')




"""
List of things to plot in this omnibus plotting file:

- Raw PSDs for emotion and frolich
- Normed Barycenter for emotion
- Filters for frolich to emotion - loaded in, not recomputed
- Transformed data for frolich
"""

# raw psds for emoiton and frolich
plot_psd(emotion_data, psds=emotion_data_psds_raw, title='Emotion Data, Unnormalized PSDs', save_path=output_dir / 'emotion_psd_raw.pdf')
plot_psd(frolich_data, psds=frolich_data_psds_raw, title='Cue Data, Unnormalized PSDs', save_path=output_dir / 'frolich_psd_raw.pdf')

# normed barycenter for emotion
normed_emotion_barycenter = np.load(data_dir / 'frolich_filters' / 'emotion_normed_barycenter.npz')['arr_0']
plot_barycenter(normed_emotion_barycenter, title='Normalized Emotion Barycenter', save_path=output_dir / 'emotion_normed_barycenter.pdf')

# NOTE: believe my currently computer frolich subj subj filters are good

# let me try the above and then I'll come back and plot the transformed data

# will also want to print out the subj to subj correspondence, polysomnograph of sample channels for a few different subjects






# now I want to transform the data using the provided filters, store it in the transformed data file directories, and then calculate psds and whatnot

# %%
# transform the data using the provided filters
for i, subj in enumerate(frolich_subj_list):

  # if the transformed data and psds already exist, skip
  if (frolich_transformed_filepath / f'frolich_extract_{subj}_256_hz_subj_subj_cmmn.npz').exists() and (frolich_transformed_subj_subj_psd_filepath / f'frolich_extract_{subj}_256_hz_subj_subj_cmmn_psds.npz').exists() and (frolich_transformed_filepath / f'frolich_extract_{subj}_256_hz_original_cmmn.npz').exists() and (frolich_transformed_original_psd_filepath / f'frolich_extract_{subj}_256_hz_original_cmmn_psds.npz').exists():
    continue

  # print progress
  print(f'Transforming frolich data for subject {subj} ({i+1}/{len(frolich_subj_list)})')

  time_filter = frolich_subj_subj_time_filters[i]
  transformed_data = transform_data_subj_subj_single(frolich_data[i], time_filter)
  np.savez(frolich_transformed_filepath / f'frolich_extract_{subj}_256_hz_subj_subj_cmmn.npz', transformed_data)

  # save psds
  f, Pxx = psd(transformed_data)
  np.savez(frolich_transformed_subj_subj_psd_filepath / f'frolich_extract_{subj}_256_hz_subj_subj_cmmn_psds.npz', Pxx)

  # now filter with original filter
  time_filter = frolich_original_time_filters[i]
  transformed_data = transform_original_single(frolich_data[i], time_filter)
  np.savez(frolich_transformed_filepath / f'frolich_extract_{subj}_256_hz_original_cmmn.npz', transformed_data)

  # save psds
  f, Pxx = psd(transformed_data)
  np.savez(frolich_transformed_original_psd_filepath / f'frolich_extract_{subj}_256_hz_original_cmmn_psds.npz', Pxx)

  print(f'Transformed frolich data for subject {subj} and calculated and saved psds')



# transform the emotion data using original filters
for i, subj in enumerate(emotion_subj_list):
   
  # skip is already done
  if (emotion_transformed_filepath / f'emotion_data_{subj}_256_hz_original_cmmn.npz').exists() and (emotion_transformed_psd_filepath / f'emotion_data_{subj}_256_hz_original_cmmn_psds.npz').exists():
    continue

  # print progress
  print(f'Transforming emotion data for subject {subj} ({i+1}/{len(emotion_subj_list)})')

  time_filter = emotion_original_time_filters[i]
  transformed_data = transform_original_single(emotion_data[i], time_filter)
  np.savez(emotion_transformed_filepath / f'emotion_data_{subj}_256_hz_original_cmmn.npz', transformed_data)

  # save psds
  f, Pxx = psd(transformed_data)
  np.savez(emotion_transformed_psd_filepath / f'emotion_data_{subj}_256_hz_original_cmmn_psds.npz', Pxx)

  print(f'Transformed emotion data for subject {subj} and calculated and saved psds')


# now visualize all of the above
frolich_transformed_subj_subj_psd_filepath.mkdir(parents=True, exist_ok=True)
frolich_transformed_original_psd_filepath.mkdir(parents=True, exist_ok=True)
emotion_transformed_psd_filepath.mkdir(parents=True, exist_ok=True)

frolich_transformed_subj_subj_psds = []
frolich_transformed_original_psds = []
emotion_transformed_psds = []

for subj in frolich_subj_list:
  frolich_transformed_subj_subj_psds.append(np.load(frolich_transformed_subj_subj_psd_filepath / f'frolich_extract_{subj}_256_hz_subj_subj_cmmn_psds.npz')['arr_0'])
  frolich_transformed_original_psds.append(np.load(frolich_transformed_original_psd_filepath / f'frolich_extract_{subj}_256_hz_original_cmmn_psds.npz')['arr_0'])

for subj in emotion_subj_list:
  emotion_transformed_psds.append(np.load(emotion_transformed_psd_filepath / f'emotion_data_{subj}_256_hz_original_cmmn_psds.npz')['arr_0'])

# plot the psds
plot_psd(frolich_transformed_subj_subj_psds, title='Cue Transformed - Subj-to-subj Mapping', save_path=output_dir / 'frolich_transformed_subj_subj_psds.pdf')
plot_psd(frolich_transformed_original_psds, title='Cue Transformed - Barycenter Mapping', save_path=output_dir / 'frolich_transformed_original_psds.pdf')
plot_psd(emotion_transformed_psds, title='Emotion Transformed - Barycenter Mapping', save_path=output_dir / 'emotion_transformed_psds.pdf')





# TODO remove after update figures for NER arXiv submission, may need some of below comments
# %%
"""
# **Computation & Plotting**

Below are plots of base data first, then the computation process.
"""

# %%
# Plot both raw and normed PSDs for comparison
# plot_psd(emotion_data, psds=emotion_data_psds_raw, title='Emotion Data - Raw PSDs', save_path=output_dir / 'emotion_psd_raw.pdf')
# if emotion_data_psds_normed:
#     plot_psd(emotion_data, psds=emotion_data_psds_normed, title='Emotion Data - Normed PSDs', save_path=output_dir / 'emotion_psd_normed.pdf')

# %%
# Use normed PSDs for barycenter computation
#psds_to_use = emotion_data_psds_normed if emotion_data_psds_normed else emotion_data_psds_raw
# psds_to_use = emotion_data_psds_raw
# normed_emotion_barycenter = compute_normed_barycenter(emotion_data, psds=psds_to_use)
# plot_barycenter(normed_emotion_barycenter, save_path=output_dir / 'emotion_barycenter.pdf')

# %%
# Plot both raw and normed PSDs for Frolich data
# plot_psd(frolich_data, psds=frolich_data_psds_raw, title='Frolich Data - Raw PSDs', save_path=output_dir / 'frolich_psd_raw.pdf')
# if frolich_data_psds_normed:
#     plot_psd(frolich_data, psds=frolich_data_psds_normed, title='Frolich Data - Normed PSDs', save_path=output_dir / 'frolich_psd_normed.pdf')

# %%
# # Plotting filters for cue -> emotion (using normed PSDs if available)
# frolich_psds_to_use = frolich_data_psds_normed if frolich_data_psds_normed else frolich_data_psds_raw
# freq_filter, time_filter = compute_filter_original(frolich_data, normed_emotion_barycenter, psds=frolich_psds_to_use)

# # %%
# plot_freq_filter(freq_filter, save_path=output_dir / 'frolich_to_emotion_freq_filter.pdf')
# plot_time_filter(time_filter, save_path=output_dir / 'frolich_to_emotion_time_filter.pdf') # note: still looks wonky but the transformed data seems to be fine

# %%
# transformed_data = transform_original(frolich_data, time_filter)

# %%
# # plot transformed data
# plot_psd(frolich_data, title='Frolich Data Original', save_path=output_dir / 'frolich_original_psd.pdf')
# plot_psd(transformed_data, title='Transformed Frolich Data Original CMMN Method', save_path=output_dir / 'frolich_transformed_original_psd.pdf')

# %%
# # plot raw frolich and transformed frolich
# plot_raw_signals(frolich_data, title='Frolich Data', save_path=output_dir / 'frolich_raw_signals.pdf')
# plot_raw_signals(transformed_data, title='Transformed Frolich Data', save_path=output_dir / 'frolich_transformed_signals.pdf')

# %%
# # Example usage for a single subject:
# channel_list = [1, 16, 64]  # specify which channels you want to see
# plot_polysomnograph(frolich_data[0], channel_indices=channel_list, time_window=(0, 5), title='First Frolich Subject, 5 seconds, Original', save_path=output_dir / 'frolich_polysomnograph_original.pdf')  # show first 60 seconds, for first subject of frolich

# plot_polysomnograph(transformed_data[0], channel_indices=channel_list, time_window=(0, 5), title='First Frolich Subject, 5 seconds, Transformed', save_path=output_dir / 'frolich_polysomnograph_transformed.pdf')  # show first 60 seconds, for first subject of frolich

# # %%
# # plot with all channels

# # there are very many channels, so just do with 1 subject to start

# plot_raw_signals([frolich_data[0]], title='Frolich Data', all_channels=True, save_path=output_dir / 'frolich_all_channels_original.pdf')
# plot_raw_signals([transformed_data[0]], title='Transformed Frolich Data', all_channels=True, save_path=output_dir / 'frolich_all_channels_transformed.pdf')

# %%
# subj_subj_matches = subj_subj_matching(psds_to_use, frolich_psds_to_use)

# freq_filter_subj_subj, time_filter_subj_subj = compute_filter_subj_subj(frolich_psds_to_use, psds_to_use, subj_subj_matches)

# transformed_data_subj_subj = transform_data_subj_subj(frolich_data, time_filter_subj_subj)

# # %%
# plot_freq_filter(freq_filter_subj_subj, save_path=output_dir / 'frolich_to_emotion_subj_subj_freq_filter.pdf')
# plot_time_filter(time_filter_subj_subj, save_path=output_dir / 'frolich_to_emotion_subj_subj_time_filter.pdf')

# # %%
# # plot original frolich data as well
# plot_psd(frolich_data, title='Frolich Data', save_path=output_dir / 'frolich_data_for_subj_subj.pdf')
# plot_psd(transformed_data_subj_subj, title='Transformed Frolich Data. Subj-Subj Matching', save_path=output_dir / 'frolich_transformed_subj_subj_psd.pdf')

# # %%
# # Example usage for a single subject:
# channel_list = [1, 16, 32, 48, 64]  # specify which channels you want to see
# plot_polysomnograph(frolich_data[0], channel_indices=channel_list, time_window=(0, 5), title='First Frolich Subject, 5 seconds, Original', save_path=output_dir / 'frolich_polysomnograph_original_subj_subj.pdf')  # show first 60 seconds, for first subject of frolich

# plot_polysomnograph(transformed_data_subj_subj[0], channel_indices=channel_list, time_window=(0, 5), title='First Frolich Subject, 5 seconds, Transformed Subj-Subj', save_path=output_dir / 'frolich_polysomnograph_transformed_subj_subj.pdf')  # show first 60 seconds, for first subject of frolich

# %%

print("\n" + "="*60)
print("VISUALIZATION COMPLETE!")
print("="*60)
print(f"All figures saved to: {output_dir}")
print(f"Processed {len(emotion_subj_list)} emotion subjects and {len(frolich_subj_list)} Fröhlich subjects")

# Print which PSDs were used
print(f"\nPSD Usage:")
if emotion_data_psds_normed:
    print(f"  - Emotion: Using NORMED PSDs for processing")
else:
    print(f"  - Emotion: Using RAW PSDs (normed not found)")

if frolich_data_psds_normed:
    print(f"  - Fröhlich: Using NORMED PSDs for processing")
else:
    print(f"  - Fröhlich: Using RAW PSDs (normed not found)")

print("\nFigures generated:")

for fig_file in sorted(output_dir.glob("*.pdf")):
    print(f"  - {fig_file.name}")

print(f"\nTotal PDF files: {len(list(output_dir.glob('*.pdf')))}")
print("Ready for NER paper submission!")
