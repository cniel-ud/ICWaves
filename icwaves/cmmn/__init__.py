"""
CMMN (Convolutional Monge Mapping Normalization) module for domain adaptation.
"""

from .core import (
    # PSD Functions
    psd,
    compute_normed_barycenter,
    
    # Filter Functions
    compute_filter_original,
    compute_filter_subj_subj,
    subj_subj_matching,
    
    # Transformation Functions
    transform_original,
    transform_data_subj_subj,
    transform_data_subj_subj_single,
    transform_original_single,
    
    # Plotting Functions
    plot_psd,
    plot_raw_signals,
    plot_polysomnograph,
    plot_barycenter,
    plot_freq_filter,
    plot_time_filter,
)

__all__ = [
    'psd',
    'compute_normed_barycenter',
    'compute_filter_original',
    'compute_filter_subj_subj',
    'subj_subj_matching',
    'transform_original',
    'transform_data_subj_subj',
    'transform_data_subj_subj_single',
    'transform_original_single',
    'plot_psd',
    'plot_raw_signals',
    'plot_polysomnograph',
    'plot_barycenter',
    'plot_freq_filter',
    'plot_time_filter',
]