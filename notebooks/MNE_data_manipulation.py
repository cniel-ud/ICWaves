#%%
import mne
from mne.fixes import _safe_svd
from scipy.io import loadmat
from mne_icalabel.iclabel.features import _eeg_rpsd, _compute_ica_activations
#%%

# Load the .mat file
mat_data = loadmat('../data/emotion_study/icact_iclabel/test_subj-01.mat')

# Extract the EEG data and relevant metadata
eeg_data = mat_data['data']
sampling_rate = mat_data['srate'][0][0]
icaweights = mat_data['icaweights']
icasphere = mat_data['icasphere']

# Display the shape of the EEG data and the sampling rate to confirm
eeg_data.shape, sampling_rate
# %%
# Create the mne.Info object
ch_names = ['EEG {}'.format(i+1) for i in range(eeg_data.shape[0])]
ch_types = ['eeg'] * len(ch_names)

# Create the mne.Info object again
info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)

# Create the RawArray object
raw = mne.io.RawArray(eeg_data, info)

# Temporarily store the data to restore after filtering
original_data = raw._data.copy()

# Use the filter method to update the metadata
raw.filter(l_freq=1, h_freq=100, fir_design='firwin', skip_by_annotation='edge', verbose=False)

# Restore the original data
raw._data = original_data

# Display a summary of the Raw object
raw.info
# %%
# Create an ICA object
n_components = icaweights.shape[0]
ica = mne.preprocessing.ICA(method='infomax', fit_params=dict(extended=True), n_components=n_components)
ica.n_pca_components = None
ica.n_components_ = n_components
W = icaweights @ icasphere
u, s, v = _safe_svd(W, full_matrices=False)
ica.unmixing_matrix_ = u * s
ica.pca_components_ = v
ica.pca_explained_variance_ = s * s
ica.info = info
ica._update_mixing_matrix()
ica._update_ica_names()
ica.reject_ = None
# %%
icaact = _compute_ica_activations(raw, ica)
psd = _eeg_rpsd(raw, ica, icaact)
# %%
