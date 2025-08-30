# ICWaves: Cross-Dataset EEG Independent Component Classification

This repository contains the implementation of methods for automatic EEG independent component (IC) classification using Bag-of-Waves (BoWav) features and Convolutional Monge Mapping Normalization (CMMN) for domain adaptation.

## Overview

EEG independent component analysis (ICA) separates brain signals from artifacts, but classifiers trained on one dataset often fail to generalize across different recording conditions. This work addresses cross-dataset generalization through:

- **Bag-of-Waves (BoWav)** features that capture waveform morphology via shift-invariant clustering
- **Extended CMMN** for spectral normalization between datasets with different channel counts
- **Domain adaptation** enabling robust classification across recording equipment and environments

## Key Features

- Cross-dataset IC classification matching ICLabel performance (F1=0.91 vs 0.89)
- Channel-averaged CMMN enabling adaptation between datasets with different electrode counts
- Subject-to-subject mapping for optimal spectral alignment
- Interpretable waveform dictionaries learned via shift-invariant k-means
- Support for datasets with different channel counts and line noise characteristics

## Publications

This repository implements methods described in:

1. **"Labeling EEG Components with a Bag of Waveforms from Learned Dictionaries"** (ICLR 2023 Workshop) - Original BoWav method for IC classification
2. **"Convolutional Monge Mapping between EEG Datasets to Support Independent Component Labeling"** - Work in progress
3. **"Cross-Dataset EEG Independent Component Classification using Bag-of-Waves Features and Convolutional Monge Mapping Normalization"** - Work in progress

### Related Work
- **"Interpretable EEG biomarkers for neurological disease models in mice using bag-of-waves classifiers"** - Application of BoWav to genotype prediction from EEG

## Installation

```bash
git clone https://github.com/cniel-ud/ICWaves.git
cd ICWaves
pip install -r requirements.txt
pip install -e .
```

## Datasets

The experiments use two publicly available datasets:

- **Imagined Emotion Dataset** ([OpenNeuro](https://openneuro.org/datasets/ds002721)): 32 subjects, 134-235 channels, 935 expert-labeled ICs
- **Cue Dataset** ([OpenNeuro](https://openneuro.org/datasets/ds006563)): 12 subjects, 64 channels, 389 expert-labeled ICs

## Usage

### Basic IC Classification

```python
# Train dictionaries for BoWav features. Class labels go from 1 to 7.
python scripts/train_dict.py --path-to-config-file config/train_dict.txt --class-label 1

# Run hyperparameter optimization. One job for each split-candidate combination.
python scripts/train_single_hpo_job.py --feature_extractor bowav --path-to-config-file config/train_classifier.txt --job-id 0

# Train final model with best parameters
python scripts/train_after_hpo.py --feature_extractor bowav --path-to-config-file config/train_classifier.txt

# Evaluate classifier performance using the evaluation notebook
from icwaves.evaluation.evaluation import eval_classifier_per_subject_brain_F1, load_classifier
from icwaves.evaluation.config import EvalConfig
from icwaves.data.loading import get_feature_extractor, load_data_bundles

# Create evaluation configuration
config = EvalConfig(
    eval_dataset='cue',
    feature_extractor='bowav',
    classifier_type='random_forest',
    validation_segment_length=300,
    cmmn_filter='subj_to_subj'
)

# Load data and feature extractor
data_bundles = load_data_bundles(config)
feature_extractor = get_feature_extractor('bowav', data_bundles)

# Load trained classifier
clf, best_params = load_classifier(config.path_to_classifier['bowav'])

# Run cross-dataset evaluation
results = eval_classifier_per_subject_brain_F1(
    config,
    {'bowav': clf},
    {'bowav': feature_extractor},
    validation_times=np.array([10, 300]),  # seconds
    data_bundles=data_bundles,
    agg_method={'bowav': best_params['input_or_output_aggregation_method']},
    train_segment_len=best_params['training_segment_length'],
    results_file=<path-to-results-file>  # where results will be saved
)
```

### CMMN Domain Adaptation (WIP)

```python
from icwaves.cmmn import


```

### Feature Extraction Options

```python
from icwaves.data.loading import get_feature_extractor, load_data_bundles

# Create configuration
config = EvalConfig(
    eval_dataset='cue',
    feature_extractor='bowav',
    classifier_type='random_forest',
    validation_segment_length=300
)

# Load data bundles for feature extraction
data_bundles = load_data_bundles(config)

# Get BoWav feature extractor
bowav_extractor = get_feature_extractor('bowav', data_bundles)
bowav_features = bowav_extractor(
    time_series={'bowav': data_bundles['bowav'].data},
    segment_len={'bowav': 300}  # seconds
)

# Get PSD + Autocorrelation feature extractor (baseline)
config.feature_extractor = 'psd_autocorr'
data_bundles = load_data_bundles(config)
psd_autocorr_extractor = get_feature_extractor('psd_autocorr', data_bundles)
psd_autocorr_features = psd_autocorr_extractor(
    time_series={'psd_autocorr': data_bundles['psd_autocorr'].data},
    segment_len={'psd_autocorr': 300}  # seconds
)

```

## Experimental Results

### Cross-Dataset Performance (Emotion → Cue)

| Method | Features | CMMN | F1 Score (Brain) |
|--------|----------|------|------------------|
| ICLabel | Spatial+Spectral | N/A | 0.89 ± 0.05 |
| Ours | BoWav | Subject-to-subject | **0.91 ± 0.08** |
| Ours | BoWav | No filtering | 0.89 ± 0.09 |
| Baseline | PSD+Autocorr | Normalized Barycenter | 0.86 ± 0.08 |

### Key Findings

- BoWav + CMMN achieves competitive performance with ICLabel using only temporal features
- Subject-to-subject CMMN mapping outperforms barycenter approaches
- Method successfully handles datasets with different channel counts (235 vs 64)
- Robust to different line noise characteristics (60Hz vs 50Hz)

## Repository Structure

```
ICWaves/
├── icwaves/              # Main package
│   ├── feature_extractors/  # BoWav and PSD/autocorr features
│   ├── sikmeans/            # Shift-invariant k-means implementation
│   ├── model_selection/     # Hyperparameter optimization
│   ├── evaluation/          # Performance metrics and validation
│   ├── data/               # Data handling utilities
│   └── config/             # Configuration files
├── scripts/              # Training and HPO scripts
├── notebooks/            # Analysis and evaluation scripts
├── matlab/               # MATLAB scripts for data processing
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── setup.py             # Package installation
└── README.md            # This file
```

Note: Some directories like `data/`, `results/`, `SLURM/`, and `.venv/` are excluded from version control via `.gitignore`.

## Dependencies

- Python 3.8+
- NumPy
- SciPy
- scikit-learn
- pandas
- matplotlib
- joblib
- tqdm
- threadpoolctl

See `requirements.txt` for specific versions.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{
mendoza-cardenas2023labeling,
title={Labeling {EEG} Components with a Bag of Waveforms from Learned Dictionaries},
author={Carlos H Mendoza-Cardenas and Austin Meek and Austin J. Brockmeier},
booktitle={ICLR 2023 Workshop on Time Series Representation Learning for Health},
year={2023},
url={https://openreview.net/forum?id=i8zQLneFvn}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- University of Delaware General University Research Fund
- DARWIN computing system (NSF Grant 1919839)
- Laura Frølich, Tobias Andersen, Klaus Gramann, and Morten Mørup for dataset access

## Contact

- Austin Meek - ajmeek@udel.edu
- Austin J. Brockmeier - ajbrock@udel.edu

Department of Computer and Information Sciences
Department of Electrical and Computer Engineering
University of Delaware
