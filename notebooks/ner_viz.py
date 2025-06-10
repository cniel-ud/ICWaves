#!/usr/bin/env python3
"""
NER Paper Visualization Script

This script generates barycenter visualizations for the NER conference paper submission.
It processes both emotion and Fröhlich datasets, computing normed barycenters for all subjects
and creating publication-ready visualizations.

Usage:
    python ner_viz.py [--data_dir DATA_DIR] [--output_dir OUTPUT_DIR] [--all_subjects]

Author: ICWaves Research Team
Date: 2025
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import welch
import seaborn as sns
from tqdm import tqdm
import warnings

# Set plotting style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

def load_subject_data(filepath, subj_id, dataset_type='emotion'):
    """
    Load data for a single subject.
    
    Parameters:
    -----------
    filepath : Path
        Directory containing the subject data files
    subj_id : str
        Subject identifier (e.g., '01', '02', etc.)
    dataset_type : str
        Either 'emotion' or 'frolich'
        
    Returns:
    --------
    data : numpy.ndarray
        EEG data for the subject, shape (n_channels, n_samples)
    """
    try:
        if dataset_type == 'emotion':
            filename = f'subj-{subj_id}.mat'
            data = loadmat(filepath / filename)['data']
        elif dataset_type == 'frolich':
            filename = f'frolich_extract_{subj_id}_256_hz.mat'
            data = loadmat(filepath / filename)['X']
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        return data
    except Exception as e:
        print(f"Warning: Could not load subject {subj_id} from {dataset_type} dataset: {e}")
        return None

def compute_psd(data, fs=256, nperseg=256):
    """
    Compute Power Spectral Density using Welch's method.
    
    Parameters:
    -----------
    data : numpy.ndarray
        EEG data, shape (n_channels, n_samples)
    fs : float
        Sampling frequency in Hz
    nperseg : int
        Length of each segment for Welch's method
        
    Returns:
    --------
    f : numpy.ndarray
        Frequency array
    psd : numpy.ndarray
        Power spectral density, shape (n_channels, n_frequencies)
    """
    f, psd = welch(data, fs=fs, nperseg=nperseg)
    return f, psd

def compute_normed_barycenter(data, psds=None):
  """

  """

  normalized_psds = []
  if psds is None:
    psds = []
  for i, subj in enumerate(data):
      if psds is None:
          f, Pxx = compute_psd(subj)
          psds.append(Pxx)
          normalized_psds.append(Pxx / np.sum(Pxx))
      else:
          normalized_psds.append(psds[i] / np.sum(psds[i]))

  # now average all together
  per_subj_avgs = []
  for subj in normalized_psds:
      per_subj_avgs.append(np.mean(subj, axis=0)) # necessary due to inhomogenous dimensions

  barycenter = np.mean(per_subj_avgs, axis=0)

  return barycenter

def compute_filter_original(data, barycenter, psds=None):
    """
    Compute the filter to transform the given data to the barycenter.

    Parameters:
    - data: list of numpy arrays, each containing EEG data for a subject in the target data (cue / frolich here)
    - barycenter: numpy array containing the normed barycenter of the data
    - psds: list of numpy arrays, each containing the PSD of the data for a subject in the target data. shape (n_channels, n_freqs)

    Returns:
    - freq_filter: numpy array containing the filter in the frequency domain
    - time_filter: numpy array containing the filter in the time domain
    """

    if psds is None:
        psds = []
        for subj in data:
            f, Pxx = compute_psd(subj)
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

def subj_subj_matching(source_psds, target_psds):
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
    for i, target_psd in enumerate(correct_target_psds):
        min_dist = np.inf
        min_idx = -1
        for j, source_psd in enumerate(correct_source_psds):
            # Use Hellinger distance (sqrt of 1 - sum of sqrt of products)
            dist = np.sqrt(1 - np.sum(np.sqrt(target_psd * source_psd)))
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        subj_subj_matches.append(min_idx)

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

def plot_freq_filter(freq_filter, f, title="Frequency Filters", save_path=None):
    """
    Plot the filter in the frequency domain.

    Parameters:
    - freq_filter: list of numpy arrays containing the filter in the frequency domain
    - f: frequency array
    - title: plot title
    - save_path: if provided, save the plot to this path
    """

    # for subj subj matching, the freq and time filters have channel components still.
    # average before plotting if so

    if freq_filter[0].ndim > 1:
        freq_filter = [np.mean(subj, axis=0) for subj in freq_filter]

    plt.figure(figsize=(12, 8))
    for i, subj in enumerate(freq_filter):
        plt.plot(f, 10 * np.log10(np.abs(subj)), label=f'Subject {i+1:02d}', alpha=0.8)

    plt.xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    plt.ylabel('Filter Magnitude (dB)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, f[-1])
    
    # Publication-quality styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved frequency filter plot to {save_path}")
    
    plt.show()

def transform_data(data, time_filters):
    """
    Transform the given data using the given time domain filters.
    
    Parameters:
    - data: list of numpy arrays, each containing EEG data for a subject
    - time_filters: list of time domain filters, one per subject
    
    Returns:
    - transformed_data: list of numpy arrays, each containing the transformed EEG data
    """
    transformed_data = []
    
    for i, subj in enumerate(data):
        if i >= len(time_filters):
            print(f"Warning: No filter available for subject {i+1}, skipping transformation")
            transformed_data.append(subj)
            continue
            
        num_channels = subj.shape[0]
        subj_transformed = np.zeros_like(subj)
        time_filter = time_filters[i]
        
        for chan in range(num_channels):
            # Apply convolution with 'full' mode and trim to original length
            subj_transformed[chan] = np.convolve(subj[chan], time_filter, mode='full')[:len(subj[chan])]
        
        transformed_data.append(subj_transformed)
    
    return transformed_data

def load_filters(filters_dir, subject_list, filter_type='time', method='original'):
    """
    Load pre-computed filters from disk using the specific naming convention.
    
    Parameters:
    - filters_dir: Path to directory containing filter files
    - subject_list: List of subject IDs
    - filter_type: 'time' or 'freq'
    - method: 'original' or 'subj_subj'
    
    Returns:
    - filters: List of loaded filters
    """
    filters = []
    
    for subj in subject_list:
        if method == 'original':
            if filter_type == 'time':
                filename = f'emotion_normed_psds_normed_barycenter_time_filter_{subj}.npz'
            else:
                filename = f'emotion_normed_psds_normed_barycenter_freq_filter_{subj}.npz'
        elif method == 'subj_subj':
            if filter_type == 'time':
                filename = f'emotion_subj_subj_time_filter_{subj}.npz'
            else:
                filename = f'emotion_subj_subj_freq_filter_{subj}.npz'
        
        filter_path = filters_dir / filename
        
        if filter_path.exists():
            loaded_filter = np.load(filter_path)['arr_0']
            filters.append(loaded_filter)
        else:
            print(f"Warning: Filter file {filename} not found")
            filters.append(None)
    
    return filters

def load_psds(psd_filepath, subject_list, dataset_type='emotion'):
    """
    Load pre-computed PSDs from disk using the specific naming convention.
    
    Parameters:
    - psd_filepath: Path to directory containing PSD files
    - subject_list: List of subject IDs
    - dataset_type: 'emotion' or 'frolich'
    
    Returns:
    - psds: List of loaded PSDs
    """
    psds = []
    
    for subj in subject_list:
        if dataset_type == 'emotion':
            # emotion_psds: subj-{XX}_psds_normed.npz where XX is 01 to 35, minus 22
            filename = f'subj-{subj}_psds_normed.npz'
        elif dataset_type == 'frolich':
            # frolich_psds: frolich_extract_{XX}_256_hz_psds_normed.npz where XX is 01 to 12
            filename = f'frolich_extract_{subj}_256_hz_psds_normed.npz'
        
        psd_path = psd_filepath / filename
        
        if psd_path.exists():
            loaded_psd = np.load(psd_path)['arr_0']
            psds.append(loaded_psd)
        else:
            print(f"Warning: PSD file {filename} not found")
            psds.append(None)
    
    return psds

def plot_time_filter(time_filter, fs=256, title="Time Domain Filters", save_path=None):
    """
    Plot the filter in the time domain.

    Parameters:
    - time_filter: list of numpy arrays containing the filter in the time domain
    - fs: sampling frequency (default: 256 Hz)
    - title: plot title
    - save_path: if provided, save the plot to this path
    """

    # for subj subj matching, the freq and time filters have channel components still.
    # average before plotting if so

    if time_filter[0].ndim > 1:
        time_filter = [np.mean(subj, axis=0) for subj in time_filter]

    plt.figure(figsize=(12, 8))
    t = np.arange(len(time_filter[0])) / fs
    for i, subj in enumerate(time_filter):
        plt.plot(t, np.real(subj), label=f'Subject {i+1:02d}', alpha=0.8)

    plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Publication-quality styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved time filter plot to {save_path}")
    
    plt.show()

def plot_barycenter(barycenter, f, title="Normed Barycenter", save_path=None):
    """
    Plot the normed barycenter with publication-quality formatting.
    
    Parameters:
    -----------
    barycenter : numpy.ndarray
        Barycenter power spectral density
    f : numpy.ndarray
        Frequency array
    title : str
        Plot title
    save_path : Path or str, optional
        If provided, save the plot to this path
    """
    plt.figure(figsize=(12, 8))
    
    # Convert to dB scale for better visualization
    barycenter_db = 10 * np.log10(barycenter)
    
    plt.plot(f, barycenter_db, linewidth=2.5, color='#2E86C1')
    plt.xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    plt.ylabel('Power Spectral Density (dB/Hz)', fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, f[-1])
    
    # Add subtle styling for publication quality
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved plot to {save_path}")
    
    plt.show()

def plot_comparison_barycenters(emotion_barycenter, frolich_barycenter, f, save_path=None):
    """
    Plot both barycenters for comparison.
    
    Parameters:
    -----------
    emotion_barycenter : numpy.ndarray
        Emotion dataset barycenter
    frolich_barycenter : numpy.ndarray
        Fröhlich dataset barycenter
    f : numpy.ndarray
        Frequency array
    save_path : Path or str, optional
        If provided, save the plot to this path
    """
    plt.figure(figsize=(14, 8))
    
    # Convert to dB scale
    emotion_db = 10 * np.log10(emotion_barycenter)
    frolich_db = 10 * np.log10(frolich_barycenter)
    
    plt.plot(f, emotion_db, linewidth=2.5, label='Emotion Dataset', 
             color='#E74C3C', alpha=0.8)
    plt.plot(f, frolich_db, linewidth=2.5, label='Fröhlich Dataset', 
             color='#3498DB', alpha=0.8)
    
    plt.xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    plt.ylabel('Power Spectral Density (dB/Hz)', fontsize=14, fontweight='bold')
    plt.title('Comparison of Normed Barycenters', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, f[-1])
    
    # Publication-quality styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved comparison plot to {save_path}")
    
    plt.show()

def plot_individual_psds(psds_list, f, dataset_name, save_path=None, max_subjects=10):
    """
    Plot individual subject PSDs (averaged across channels) for visualization.
    
    Parameters:
    -----------
    psds_list : list of numpy.ndarray
        List of PSDs for each subject
    f : numpy.ndarray
        Frequency array
    dataset_name : str
        Name of the dataset for the title
    save_path : Path or str, optional
        If provided, save the plot to this path
    max_subjects : int
        Maximum number of subjects to plot (for readability)
    """
    plt.figure(figsize=(14, 8))
    
    # Limit the number of subjects plotted for clarity
    n_subjects_to_plot = min(len(psds_list), max_subjects)
    
    for i in range(n_subjects_to_plot):
        # Average across channels for each subject
        avg_psd = np.mean(psds_list[i], axis=0)
        psd_db = 10 * np.log10(avg_psd)
        
        plt.plot(f, psd_db, alpha=0.7, linewidth=1.5, 
                label=f'Subject {i+1:02d}' if n_subjects_to_plot <= 8 else None)
    
    plt.xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    plt.ylabel('Power Spectral Density (dB/Hz)', fontsize=14, fontweight='bold')
    plt.title(f'{dataset_name} Dataset - Individual Subject PSDs', 
              fontsize=16, fontweight='bold', pad=20)
    
    if n_subjects_to_plot <= 8:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.xlim(0, f[-1])
    
    # Publication-quality styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_linewidth(1.5)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved individual PSDs plot to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Generate NER paper visualizations')
    parser.add_argument('--data_dir', type=str, default='../data', 
                       help='Root directory containing the data')
    parser.add_argument('--output_dir', type=str, default='./ner_figures', 
                       help='Directory to save output figures')
    parser.add_argument('--all_subjects', action='store_true', 
                       help='Process all subjects (default: subset for testing)')
    parser.add_argument('--fs', type=float, default=256.0, 
                       help='Sampling frequency in Hz')
    parser.add_argument('--nperseg', type=int, default=256, 
                       help='Window length for PSD computation')
    
    args = parser.parse_args()
    
    # Set up paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    emotion_filepath = data_dir / 'emotion_256' / 'raw_data_and_IC_labels'
    frolich_filepath = data_dir / 'frolich_256' / 'frolich_extract_256_hz'

    emotion_psd_filepath = emotion_filepath / 'psds_normed'
    frolich_psd_filepath = frolich_filepath / 'psds_normed'

    filters_filepath = data_dir / 'frolich_filters'

    '''
    Naming conventions from how I previously created things:
    emotion_psds: subj-{XX}_psds_normed.npz where XX is 01 to 35, minus 22
    frolich_psds: frolich_extract_{XX}_256_hz.npz where XX is 01 to 12
    
    emotion_filters: emotion_normed_psds_normed_barycenter_freq_filter_{XX}.npz
    
    # frolich filters need to be recalculated, just visualize the above for now
    '''
    
    # Define subject lists
    if args.all_subjects:
        emotion_subj_list = [f'{i:02d}' for i in range(1, 36) if i != 22]  # 35 subjects minus 22
        frolich_subj_list = [f'{i:02d}' for i in range(1, 13)]  # 12 subjects
    else:
        # Subset for testing/development (exclude 22 if present)
        emotion_subj_list = ['01', '02', '03', '04', '05']
        frolich_subj_list = ['01', '02', '03', '04']
    
    print(f"Processing {len(emotion_subj_list)} emotion subjects and {len(frolich_subj_list)} Fröhlich subjects")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load and process emotion dataset
    print("\n" + "="*60)
    print("LOADING EMOTION DATASET")
    print("="*60)
    
    emotion_data = []
    
    for subj in tqdm(emotion_subj_list, desc="Loading emotion subjects"):
        data = load_subject_data(emotion_filepath, subj, 'emotion')
        if data is not None:
            emotion_data.append(data)
    
    print(f"Successfully loaded {len(emotion_data)} emotion subjects")
    
    # Load and process Fröhlich dataset
    print("\n" + "="*60)
    print("LOADING FRÖHLICH DATASET")
    print("="*60)
    
    frolich_data = []
    
    for subj in tqdm(frolich_subj_list, desc="Loading Fröhlich subjects"):
        data = load_subject_data(frolich_filepath, subj, 'frolich')
        if data is not None:
            frolich_data.append(data)
    
    print(f"Successfully loaded {len(frolich_data)} Fröhlich subjects")
    
    # Compute barycenters (this will calculate PSDs internally as needed)
    print("\n" + "="*60)
    print("COMPUTING BARYCENTERS")
    print("="*60)
    
    if emotion_data:
        print("Computing emotion dataset barycenter...")
        emotion_barycenter = compute_normed_barycenter(emotion_data, psds=None)
        
        # For plotting, we need to compute PSDs to get frequency array
        print("Computing PSDs for plotting...")
        f, temp_psd = compute_psd(emotion_data[0], fs=args.fs, nperseg=args.nperseg)
        
        print("Plotting emotion barycenter...")
        plot_barycenter(emotion_barycenter, f, 
                       title="Emotion Dataset - Normed Barycenter",
                       save_path=output_dir / "emotion_barycenter.pdf")
    
    if frolich_data:
        print("Computing Fröhlich dataset barycenter...")
        frolich_barycenter = compute_normed_barycenter(frolich_data, psds=None)
        
        # For plotting, we need to compute PSDs to get frequency array  
        print("Computing PSDs for plotting...")
        f, temp_psd = compute_psd(frolich_data[0], fs=args.fs, nperseg=args.nperseg)
        
        print("Plotting Fröhlich barycenter...")
        plot_barycenter(frolich_barycenter, f, 
                       title="Fröhlich Dataset - Normed Barycenter",
                       save_path=output_dir / "frolich_barycenter.pdf")
    
    # Create comparison plot if both datasets are available
    if emotion_data and frolich_data:
        print("Creating comparison plot...")
        plot_comparison_barycenters(emotion_barycenter, frolich_barycenter, f,
                                  save_path=output_dir / "barycenter_comparison.pdf")

    # Transform data using pre-computed filters
    print("\n" + "="*60)
    print("DATA TRANSFORMATION")
    print("="*60)
    
    # Example usage - uncomment to use your pre-computed data:
    """
    # Load pre-computed PSDs using your specific naming convention
    print("Loading pre-computed PSDs...")
    emotion_psds = load_psds(emotion_psd_filepath, emotion_subj_list, 'emotion')
    frolich_psds = load_psds(frolich_psd_filepath, frolich_subj_list, 'frolich')
    
    # Plot original PSDs from your pre-computed files
    if emotion_psds and any(psd is not None for psd in emotion_psds):
        # Get frequency array from first valid PSD
        valid_psd = next(psd for psd in emotion_psds if psd is not None)
        f_loaded = np.linspace(0, args.fs/2, valid_psd.shape[-1])
        
        print("Plotting pre-computed emotion PSDs...")
        plot_individual_psds([psd for psd in emotion_psds if psd is not None], 
                           f_loaded, "Pre-computed Emotion",
                           save_path=output_dir / "precomputed_emotion_psds.pdf")
    
    if frolich_psds and any(psd is not None for psd in frolich_psds):
        # Get frequency array from first valid PSD  
        valid_psd = next(psd for psd in frolich_psds if psd is not None)
        f_loaded = np.linspace(0, args.fs/2, valid_psd.shape[-1])
        
        print("Plotting pre-computed Fröhlich PSDs...")
        plot_individual_psds([psd for psd in frolich_psds if psd is not None],
                           f_loaded, "Pre-computed Fröhlich", 
                           save_path=output_dir / "precomputed_frolich_psds.pdf")
    
    # Load and visualize emotion filters (available)
    if emotion_data:
        print("Loading pre-computed emotion filters...")
        emotion_time_filters = load_filters(filters_filepath, emotion_subj_list, 
                                          filter_type='time', method='original')
        emotion_freq_filters = load_filters(filters_filepath, emotion_subj_list,
                                          filter_type='freq', method='original')
        
        # Plot filters if available
        valid_time_filters = [f for f in emotion_time_filters if f is not None]
        valid_freq_filters = [f for f in emotion_freq_filters if f is not None]
        
        if valid_time_filters:
            print("Plotting emotion time domain filters...")
            plot_time_filter(valid_time_filters, args.fs,
                           title="Emotion Time Domain Filters",
                           save_path=output_dir / "emotion_time_filters.pdf")
        
        if valid_freq_filters:
            print("Plotting emotion frequency domain filters...")
            # Create frequency array for filter plotting
            f_filter = np.linspace(0, args.fs/2, len(valid_freq_filters[0]))
            plot_freq_filter(valid_freq_filters, f_filter,
                           title="Emotion Frequency Domain Filters",
                           save_path=output_dir / "emotion_freq_filters.pdf")
        
        print(f"Successfully loaded {len(valid_time_filters)} time filters and {len(valid_freq_filters)} freq filters")
    
    # Note: Fröhlich filters need to be recalculated as mentioned in your comment
    print("Note: Fröhlich filters need to be recalculated (as noted in your comment)")
    """
    
    print("Ready for transformation! Uncomment the section above to use your pre-computed data.")
    print("Available functions:")
    print("  - load_psds(): Load your pre-computed PSD files using exact naming convention")
    print("  - load_filters(): Load your pre-computed filter files")
    print("  - transform_data(): Apply time domain filters to EEG data")
    print("  - plot_individual_psds(): Visualize PSDs from your files")
    print("  - plot_freq_filter() and plot_time_filter(): Visualize your filters")

    # Save numerical results
    print("\n" + "="*60)
    print("SAVING NUMERICAL RESULTS")
    print("="*60)
    
    if emotion_data:
        np.savez(output_dir / "emotion_barycenter.npz", 
                barycenter=emotion_barycenter, frequencies=f)
        print(f"Saved emotion barycenter to {output_dir / 'emotion_barycenter.npz'}")
    
    if frolich_data:
        np.savez(output_dir / "frolich_barycenter.npz", 
                barycenter=frolich_barycenter, frequencies=f)
        print(f"Saved Fröhlich barycenter to {output_dir / 'frolich_barycenter.npz'}")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"All figures saved to: {output_dir}")
    print("\nFigures generated:")
    
    for fig_file in output_dir.glob("*.pdf"):
        print(f"  - {fig_file.name}")
    
    print(f"\nData files saved:")
    for data_file in output_dir.glob("*.npz"):
        print(f"  - {data_file.name}")

if __name__ == "__main__":
    main()
