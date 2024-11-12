# %%
# Import necessary libraries
import os
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from icwaves.feature_extractors.psd import eeg_psd
from icwaves.feature_extractors.autocorr import eeg_autocorr
from icwaves.feature_extractors.iclabel_features import get_iclabel_features

# %%
HOME = os.path.expanduser("~")
ICWaves_repo = Path(HOME, "personal_repos", "ICWaves")
dpath = ICWaves_repo / "data/emotion_study/raw_data_and_IC_labels"
subj_ids = [8, 9]
fnames = [f"subj-{i:02}.mat" for i in subj_ids]
file_list = [dpath.joinpath(f) for f in fnames]
mat_dict = loadmat(file_list[0])
data = mat_dict["data"]
icaweights = mat_dict["icaweights"]
icasphere = mat_dict["icasphere"]
labels = mat_dict["labels"] - 1
noisy_labels = mat_dict["noisy_labels"]
expert_label_mask = mat_dict["expert_label_mask"].astype(bool)
srate = mat_dict["srate"].item(0)
ica_activations = icaweights @ icasphere @ data
# %%
psd_ica = eeg_psd(ica_activations, srate)
# %%
autcorr_ica = eeg_autocorr(ica_activations, srate)
# %%
iclabel_features = get_iclabel_features(ica_activations, srate)

# %%
