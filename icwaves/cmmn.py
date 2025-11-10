"""
CMMN (Convolutional Monge Mapping Normalization) - Single file implementation.

This module provides domain adaptation for EEG signals using optimal transport-based
spectral normalization techniques.
"""

import numpy as np
from scipy.signal import welch
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple
import warnings


class CMMNProcessor:
    """CMMN (Convolutional Monge Mapping Normalization) processor.

    This class handles all CMMN operations including filter generation,
    data transformation, and saving/loading filters.

    Args:
        method: CMMN method ('normed-barycenter', 'unnormed-barycenter', 'subj-to-subj')
        sampling_rate: Sampling rate in Hz
        nperseg: Length of FFT segments
        verbose: Print progress messages

    Example:
        >>> processor = CMMNProcessor(method='normed-barycenter')
        >>> processor.fit('/path/to/source', '/path/to/target')
        >>> filtered_data = processor.transform(processor.target_data)
        >>> processor.save_filters('./filters')
    """

    def __init__(self,
                 method: str = 'normed-barycenter',
                 sampling_rate: int = 256,
                 nperseg: int = 256,
                 verbose: bool = True):
        """Initialize CMMN processor."""
        self.method = method
        self.sampling_rate = sampling_rate
        self.nperseg = nperseg
        self.verbose = verbose

        # Data storage
        self.source_data = None
        self.target_data = None
        self.source_psds = None
        self.target_psds = None
        self.barycenter = None
        self.filters = None
        self.subj_matches = None

    def fit(self,
            source_data: Union[Dict[str, np.ndarray], Path, str],
            target_data: Union[Dict[str, np.ndarray], Path, str]) -> 'CMMNProcessor':
        """Fit CMMN filters from source to target domain.

        Args:
            source_data: Source domain data (dict, path to directory, or path to .npz)
            target_data: Target domain data (dict, path to directory, or path to .npz)

        Returns:
            Self for method chaining
        """
        # Load data
        self.source_data = self._load_data(source_data, "source")
        self.target_data = self._load_data(target_data, "target")

        # Compute PSDs
        if self.verbose:
            print("Computing PSDs...")
        self.source_psds = self._compute_psds(self.source_data)
        self.target_psds = self._compute_psds(self.target_data)

        # Generate filters based on method
        if self.verbose:
            print(f"Generating {self.method} filters...")

        if self.method == 'normed-barycenter':
            self._generate_normed_barycenter_filters()
        elif self.method == 'unnormed-barycenter':
            self._generate_unnormed_barycenter_filters()
        elif self.method == 'subj-to-subj':
            self._generate_subj_to_subj_filters()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        if self.verbose:
            print(f"Generated filters for {len(self.filters['time'])} subjects")

        return self

    def transform(self, data: Union[Dict[str, np.ndarray], np.ndarray]) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Apply CMMN filters to transform data.

        Args:
            data: Data to transform (dict or array)

        Returns:
            Transformed data in same format as input
        """
        if self.filters is None:
            raise RuntimeError("No filters available. Call fit() first.")

        # Handle different input formats
        return_array = isinstance(data, np.ndarray)

        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                data_dict = {"subj_00": data}
            elif data.ndim == 3:
                data_dict = {f"subj_{i:02d}": data[i] for i in range(len(data))}
            else:
                raise ValueError(f"Array must be 2D or 3D, got {data.ndim}D")
        else:
            data_dict = data

        # Transform each subject
        transformed = {}
        time_filters = self.filters['time']

        for subj_id, subj_data in data_dict.items():
            # Get appropriate filter
            if subj_id in time_filters:
                filter_id = subj_id
            else:
                # Use first available filter if subject not found
                filter_id = list(time_filters.keys())[0]
                if len(time_filters) > 1 and self.verbose:
                    print(f"Warning: No specific filter for {subj_id}, using {filter_id}")

            # Apply transformation
            if self.method == 'subj-to-subj':
                transformed[subj_id] = self._transform_subj_to_subj_single(
                    subj_data, time_filters[filter_id]
                )
            else:
                transformed[subj_id] = self._transform_original_single(
                    subj_data, time_filters[filter_id]
                )

        # Return in same format as input
        if return_array:
            if len(transformed) == 1:
                return list(transformed.values())[0]
            else:
                return np.stack(list(transformed.values()))
        else:
            return transformed

    def fit_transform(self,
                     source_data: Union[Dict[str, np.ndarray], Path, str],
                     target_data: Union[Dict[str, np.ndarray], Path, str]) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Fit and transform in one step.

        Args:
            source_data: Source domain data
            target_data: Target domain data

        Returns:
            Transformed target data
        """
        self.fit(source_data, target_data)
        return self.transform(self.target_data)

    def save_filters(self, output_path: Union[Path, str]) -> None:
        """Save filters to disk.

        Args:
            output_path: Directory to save filters
        """
        if self.filters is None:
            raise RuntimeError("No filters to save. Call fit() first.")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save each subject's filters
        for subj_id in self.filters['time'].keys():
            np.savez(output_path / f"subj-{subj_id}.npz",
                    time_filter=self.filters['time'][subj_id],
                    freq_filter=self.filters['freq'][subj_id],
                    method=self.method,
                    sampling_rate=self.sampling_rate)

        # Save barycenter if available
        if self.barycenter is not None:
            np.savez(output_path / "barycenter.npz",
                    barycenter=self.barycenter,
                    method=self.method)

        if self.verbose:
            print(f"Saved filters to {output_path}")

    def load_filters(self, filter_path: Union[Path, str]) -> 'CMMNProcessor':
        """Load pre-computed filters.

        Args:
            filter_path: Directory containing filter files

        Returns:
            Self for method chaining
        """
        filter_path = Path(filter_path)
        if not filter_path.exists():
            raise FileNotFoundError(f"Filter path does not exist: {filter_path}")

        freq_filters = {}
        time_filters = {}

        # Load all subject filter files
        for filter_file in sorted(filter_path.glob("subj-*.npz")):
            subj_id = filter_file.stem.replace("subj-", "")
            data = np.load(filter_file)
            freq_filters[subj_id] = data['freq_filter']
            time_filters[subj_id] = data['time_filter']

        self.filters = {'freq': freq_filters, 'time': time_filters}

        # Try to load barycenter
        barycenter_file = filter_path / "barycenter.npz"
        if barycenter_file.exists():
            data = np.load(barycenter_file)
            self.barycenter = data['barycenter']

        if self.verbose:
            print(f"Loaded {len(time_filters)} filters from {filter_path}")

        return self

    # ========== Private methods ==========

    def _load_data(self, data: Union[Dict[str, np.ndarray], Path, str], domain: str) -> Dict[str, np.ndarray]:
        """Load data from various formats."""
        if isinstance(data, dict):
            return data

        data_path = Path(data)
        if not data_path.exists():
            raise FileNotFoundError(f"{domain} data path not found: {data_path}")

        loaded_data = {}

        if data_path.is_file() and data_path.suffix == '.npz':
            # Load from single NPZ file
            loaded = np.load(data_path, allow_pickle=True)
            if 'data' in loaded:
                raw_data = loaded['data']
                if isinstance(raw_data, np.ndarray):
                    for i in range(len(raw_data)):
                        loaded_data[f"subj_{i:02d}"] = raw_data[i]
                else:
                    loaded_data = dict(raw_data)
            else:
                # Assume each key is a subject
                for key in loaded.keys():
                    loaded_data[key] = loaded[key]

        elif data_path.is_dir():
            # Load from directory of subject files
            for subj_file in sorted(data_path.glob("*.np[yz]")):
                subj_id = subj_file.stem
                if subj_file.suffix == '.npz':
                    data = np.load(subj_file, allow_pickle=True)
                    if 'data' in data:
                        loaded_data[subj_id] = data['data']
                    else:
                        # Get first array
                        keys = list(data.keys())
                        if keys:
                            loaded_data[subj_id] = data[keys[0]]
                else:
                    loaded_data[subj_id] = np.load(subj_file)

        else:
            raise ValueError(f"Unsupported data format for {data_path}")

        if self.verbose:
            print(f"Loaded {len(loaded_data)} subjects from {domain} domain")

        return loaded_data

    def _compute_psds(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute PSDs for all subjects."""
        psds = {}
        for subj_id, subj_data in data.items():
            _, Pxx = welch(subj_data, fs=self.sampling_rate, nperseg=self.nperseg)
            psds[subj_id] = Pxx
        return psds

    def _generate_normed_barycenter_filters(self):
        """Generate filters using normalized barycenter."""
        # Convert PSDs to array
        source_psd_array = np.stack(list(self.source_psds.values()))

        # Compute normalized barycenter
        self.barycenter = self._compute_normed_barycenter(source_psd_array)

        # Generate filters for each target subject
        freq_filters = {}
        time_filters = {}

        for subj_id, subj_psd in self.target_psds.items():
            # Average across channels
            avg_subj_psd = np.mean(subj_psd, axis=0)

            # Compute filter
            freq_filter = np.sqrt(self.barycenter) / np.sqrt(avg_subj_psd)
            time_filter = np.fft.irfft(freq_filter)

            freq_filters[subj_id] = freq_filter
            time_filters[subj_id] = time_filter

        self.filters = {'freq': freq_filters, 'time': time_filters}

    def _generate_unnormed_barycenter_filters(self):
        """Generate filters using unnormalized barycenter."""
        # Compute unnormalized barycenter
        source_psd_array = np.stack(list(self.source_psds.values()))
        avg_psds = []
        for subj_psd in source_psd_array:
            avg_psd = np.mean(subj_psd, axis=0)  # Average across channels
            avg_psds.append(avg_psd)
        self.barycenter = np.mean(avg_psds, axis=0)

        # Generate filters
        freq_filters = {}
        time_filters = {}

        for subj_id, subj_psd in self.target_psds.items():
            # Compute filter using unnormalized barycenter
            avg_subj_psd = np.mean(subj_psd, axis=0)
            freq_filter = np.sqrt(self.barycenter) / np.sqrt(avg_subj_psd + 1e-10)
            time_filter = np.fft.irfft(freq_filter)

            freq_filters[subj_id] = freq_filter
            time_filters[subj_id] = time_filter

        self.filters = {'freq': freq_filters, 'time': time_filters}

    def _generate_subj_to_subj_filters(self):
        """Generate filters using subject-to-subject matching."""
        # Convert to arrays
        source_psd_array = np.stack(list(self.source_psds.values()))
        target_psd_array = np.stack(list(self.target_psds.values()))
        target_ids = list(self.target_psds.keys())

        # Perform matching
        self.subj_matches = self._subj_subj_matching(source_psd_array, target_psd_array)

        # Generate filters
        freq_filters = {}
        time_filters = {}

        # Average PSDs across channels
        averaged_source_psds = [np.mean(psd, axis=0) for psd in source_psd_array]
        averaged_target_psds = [np.mean(psd, axis=0) for psd in target_psd_array]

        for i, target_id in enumerate(target_ids):
            source_psd = averaged_source_psds[self.subj_matches[i]]
            target_psd = averaged_target_psds[i]

            freq_filter = np.sqrt(source_psd) / np.sqrt(target_psd)
            time_filter = np.fft.irfft(freq_filter)

            freq_filters[target_id] = freq_filter
            time_filters[target_id] = time_filter

        self.filters = {'freq': freq_filters, 'time': time_filters}

    # ========== Core CMMN algorithms (from core.py) ==========

    def _compute_normed_barycenter(self, psds: np.ndarray) -> np.ndarray:
        """Compute the normalized barycenter of PSDs across subjects.

        Parameters:
            psds: Array of PSDs, shape (n_subjects, n_channels, n_freqs)

        Returns:
            barycenter: Normalized barycenter
        """
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

    def _subj_subj_matching(self, source_psds: np.ndarray, target_psds: np.ndarray) -> List[int]:
        """Match each target subject to best source subject using Hellinger distance.

        Parameters:
            source_psds: Source PSDs, shape (n_source_subjects, n_channels, n_freqs)
            target_psds: Target PSDs, shape (n_target_subjects, n_channels, n_freqs)

        Returns:
            List of source subject indices for each target subject
        """
        # Average across channels
        correct_source_psds = [np.mean(subj, axis=0) for subj in source_psds]
        correct_target_psds = [np.mean(subj, axis=0) for subj in target_psds]

        # Normalize to probability distributions
        for i, target_psd in enumerate(correct_target_psds):
            correct_target_psds[i] = target_psd / np.sum(target_psd)
        for i, source_psd in enumerate(correct_source_psds):
            correct_source_psds[i] = source_psd / np.sum(source_psd)

        subj_subj_matches = []
        for i, target_psd in enumerate(correct_target_psds):
            min_dist = float('inf')
            min_idx = -1
            for j, source_psd in enumerate(correct_source_psds):
                # Hellinger distance
                dist = (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(source_psd) - np.sqrt(target_psd))**2))

                if dist < min_dist:
                    min_dist = dist
                    min_idx = j
            subj_subj_matches.append(min_idx)

            if self.verbose:
                print(f"Target subject {i} matched to source subject {min_idx}")

        return subj_subj_matches

    def _transform_original_single(self, subj_data: np.ndarray, time_filter: np.ndarray) -> np.ndarray:
        """Transform single subject data using the given filter.

        Parameters:
            subj_data: EEG data, shape (n_channels, n_samples)
            time_filter: Time domain filter

        Returns:
            Transformed data, same shape as input
        """
        num_channels = subj_data.shape[0]
        subj_norm = np.zeros_like(subj_data)

        for chan in range(num_channels):
            subj_norm[chan] = np.convolve(subj_data[chan], time_filter, mode='full')[:len(subj_data[chan])]

        return subj_norm

    def _transform_subj_to_subj_single(self, subj_data: np.ndarray, time_filter: np.ndarray) -> np.ndarray:
        """Transform single subject data using subject-specific filter.

        For subject-to-subject matching, the time_filter is a 1D array that applies
        to all channels (since we averaged across channels during filter generation).

        Parameters:
            subj_data: EEG data, shape (n_channels, n_samples)
            time_filter: Time domain filter (1D array)

        Returns:
            Transformed data, same shape as input
        """
        num_channels = subj_data.shape[0]
        subj_norm = np.zeros_like(subj_data)

        for chan in range(num_channels):
            subj_norm[chan] = np.convolve(subj_data[chan], time_filter, mode='full')[:len(subj_data[chan])]

        return subj_norm


# ========== Utility function for direct IC filtering ==========

def apply_filter_to_ic(ic_data: np.ndarray, filter_array: np.ndarray) -> np.ndarray:
    """Apply CMMN filter to IC activation data.

    Args:
        ic_data: IC activation data (1D or 2D array)
        filter_array: Filter to apply

    Returns:
        Filtered IC data
    """
    if ic_data.ndim == 1:
        # Single IC
        filtered = np.convolve(ic_data, filter_array, mode='full')[:len(ic_data)]
        return filtered

    elif ic_data.ndim == 2:
        # Multiple ICs
        n_ics, n_samples = ic_data.shape
        filtered = np.zeros_like(ic_data)

        for i in range(n_ics):
            filtered[i] = np.convolve(ic_data[i], filter_array, mode='full')[:n_samples]

        return filtered

    else:
        raise ValueError(f"IC data must be 1D or 2D, got {ic_data.ndim}D")