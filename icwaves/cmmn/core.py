"""
Core CMMN (Convolutional Monge Mapping Normalization) functions for domain adaptation.

This module contains the implementation of CMMN for EEG signal processing and domain adaptation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from typing import List, Tuple, Optional
from pathlib import Path


# PSD Computation Functions
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


# Barycenter Computation
def compute_normed_barycenter(data, psds=None):
    """
    Compute the normalized barycenter of PSDs across subjects.
    
    Parameters:
    - data: list of numpy arrays, each containing EEG data for a subject
    - psds: pre-computed PSDs (optional)
    
    Returns:
    - barycenter: numpy array containing the normed barycenter of the data
    """
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


# Filter Computation Functions
def compute_filter_original(data, barycenter, psds=None):
    """
    Compute the filter to transform the given data to the barycenter.

    Parameters:
    - data: list of numpy arrays, each containing EEG data for a subject in the target data (cue / frolich here)
    - barycenter: numpy array containing the normed barycenter of the data
    - psds: pre-computed PSDs (optional)

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
        time_filter = np.fft.irfft(subj)  # note, now with real values only.
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

    # Normalizing so that Hellinger dist makes sense
    # make them sum to one
    for i, target_psd in enumerate(correct_target_psds):
        correct_target_psds[i] = target_psd / np.sum(target_psd)
    for i, source_psd in enumerate(correct_source_psds):
        correct_source_psds[i] = source_psd / np.sum(source_psd)

    subj_subj_matches = []
    for i, target_psd in enumerate(correct_target_psds):  # frolich
        min_dist = float('inf')
        min_idx = -1
        for j, source_psd in enumerate(correct_source_psds):  # emotion
            # Hellinger distance
            dist = (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(source_psd) - np.sqrt(target_psd))**2))

            if dist < min_dist:
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

        freq_filter = np.sqrt(source_psd) / np.sqrt(subj)  # sqrt of emotion psd / sqrt of frolich psd
        time_filter = np.fft.irfft(freq_filter)  # note: changed from ifft to irfft

        freq_filter_per_subj.append(freq_filter)
        time_filter_per_subj.append(time_filter)

    return freq_filter_per_subj, time_filter_per_subj


# Transformation Functions
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
            # note dr. b wants changed to full b/c of weird reverse shifting
            subj_norm[chan] = np.convolve(subj[chan], time_filter[i], mode='full')[:len(subj[chan])]

        transformed_data.append(subj_norm)

    return transformed_data


def transform_data_subj_subj(data, time_filter_subj_subj):
    """
    Transform data with knowledge that the time filter is a list, with each specific to the individual target domain subject

    Parameters
    ----------
    data: list of numpy arrays, each containing EEG data for a subject
    time_filter_subj_subj: list of time filters, one per subject

    Returns
    -------
    transformed_data: list of transformed EEG data arrays
    """
    transformed_data = []
    for i, subj in enumerate(data):
        subj_norm = np.zeros(subj.shape)
        num_channels = subj.shape[0]
        time_filter = time_filter_subj_subj[i]  # subj specific time filter

        for chan in range(num_channels):
            # here also change this to full, matching the above
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


# Plotting Functions
def plot_psd(data, fs=256, nperseg=256, psds=None, title='PSD', save_path=None):
    """
    Plot the Power Spectral Density (PSD) of the given data.

    Note: right now this is straight averaging, not the barycenter norming I do later.
    If I want to do the norming, apply that first and then this will not average for me.

    Parameters:
    - data: list of numpy arrays, each containing EEG data for a subject
    - fs: sampling frequency (default: 256 Hz)
    - nperseg: length of each segment for Welch's method (default: 256)
    - psds: pre-computed PSDs (optional)
    - title: plot title
    - save_path: if provided, save the plot to this path as PDF
    """
    plt.figure(figsize=(12, 8))

    for i, subj_data in enumerate(data):
        if psds is None:
            # If subj_data is multi-dimensional, average across the channels
            if subj_data.ndim > 1:
                subj_data = np.mean(subj_data, axis=0)
            f, Pxx = psd(subj_data, fs=fs, nperseg=nperseg)

            plt.plot(f, 10 * np.log10(Pxx), label=f'Subject {i+1}')
        else:
            viz_psds = [np.zeros_like(psd_val) for psd_val in psds]
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


def plot_raw_signals(data, fs=256, title='Raw Signals', all_channels=False, save_path=None):
    """
    Plot the raw signals of the given data.

    Parameters:
    - data: list of numpy arrays, each containing EEG data for a subject
    - fs: sampling frequency (default: 256 Hz)
    - title: plot title
    - all_channels: if True, plot all channels separately
    - save_path: if provided, save the plot to this path as PDF
    """
    plt.figure(figsize=(12, 8))

    for i, subj_data in enumerate(data):
        if all_channels:
            # Check if subj_data has multiple channels
            if subj_data.ndim > 1:
                for channel in range(subj_data.shape[1]):
                    plt.plot(np.arange(subj_data.shape[0]) / fs, subj_data[:, channel])
            else:
                # If subj_data is 1-dimensional, plot it directly
                plt.plot(np.arange(subj_data.shape[0]) / fs, subj_data, label=f'Subject {i+1}, Channel 1')
        else:
            # average all channels together
            subj_data = np.mean(subj_data, axis=0)  # changed from 1 to 0

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
    plt.figure(figsize=(15, len(channel_indices) * 1.5))
    
    # Apply time window if specified
    if time_window is not None:
        start_sample = int(time_window[0] * fs)
        end_sample = int(time_window[1] * fs)
        data = data[:, start_sample:end_sample]
    
    # Calculate time vector
    time = np.arange(data.shape[1]) / fs
    
    # Plot each channel
    for i, chan_idx in enumerate(channel_indices):
        # Get channel data (subtract 1 from index since data is 0-indexed)
        channel = data[chan_idx - 1]
        
        # Normalize the channel
        normalized_channel = channel / np.max(np.abs(channel))
        
        # Plot with offset
        offset = -i * spacing_factor
        plt.plot(time, normalized_channel + offset, label=f'Channel {chan_idx}')

    plt.xlabel('Time (s)')
    plt.yticks([])  # Remove y-axis ticks since they're arbitrary
    if title:
        plt.title(title)
    
    # Add channel labels on the y-axis
    channel_positions = [-i * spacing_factor for i in range(len(channel_indices))]
    plt.yticks(channel_positions, [f'Ch {idx}' for idx in channel_indices])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_barycenter(barycenter, title='Normed Barycenter', save_path=None):
    """
    Plot the normed barycenter of the given data.

    Parameters:
    - barycenter: numpy array containing the normed barycenter of the data
    - title: plot title
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