from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List
from icwaves.file_utils import build_base_classifier_name
from typing import Optional
from copy import deepcopy

SUPPORTED_CLASSIFIERS = ["random_forest", "logistic", "ensembled_logistic"]
SUPPORTED_DATASETS = ["emotion_study", "cue"]
SUPPORTED_FEATURES = ["bowav", "psd_autocorr", "bowav_psd_autocorr"]
SUPPORTED_CMNN_FILTERS = ["normed-barycenter", "unnormed-barycenter", "subj_to_subj"]


@dataclass
class EvalConfig:
    """Configuration for the evaluation pipeline.

    This configuration assumes that both the BoWav dictionaries and the
    classifier was trained on the 'emotion_study' dataset, using subjects
    8 to 35 (excluding subject 22, which is missing).
    """

    eval_dataset: str  # 'emotion_study' or 'cue'
    train_config: Namespace  # Namespace object with args used to train the classifier
    root: Path
    cmmn_filter: Optional[str] = None

    def _flatten_train_config(self) -> None:
        self.validation_segment_length = self.train_config.validation_segment_length
        self.classifier_type = self.train_config.classifier_type
        self.minutes_per_ic = self.train_config.minutes_per_ic
        self.feature_extractor = self.train_config.feature_extractor
        if "bowav" in self.feature_extractor:
            self.window_length = self.train_config.window_length
            self.num_clusters = self.train_config.num_clusters
            self.centroid_length = self.train_config.centroid_length
            self.codebook_minutes_per_ic = self.train_config.codebook_minutes_per_ic
            self.codebook_ics_per_subject = self.train_config.codebook_ics_per_subject
            self.use_idf = self.train_config.use_idf

    def __post_init__(self):
        if self.eval_dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unknown eval dataset {self.eval_dataset}")
        if self.train_config.classifier_type not in SUPPORTED_CLASSIFIERS:
            raise ValueError(
                f"Unknown classifier type {self.train_config.classifier_type}"
            )
        if self.train_config.feature_extractor not in SUPPORTED_FEATURES:
            raise ValueError(
                f"Unknown feature extractor {self.train_config.feature_extractor}"
            )
        if (
            self.cmmn_filter is not None
            and self.cmmn_filter not in SUPPORTED_CMNN_FILTERS
        ):
            raise ValueError(f"Unknown cmmn filter {self.cmmn_filter}")
        if self.cmmn_filter == "subj_to_subj" and self.eval_dataset != "cue":
            raise ValueError(f"cmmn filter only supported for cue dataset")

        # Wheter we use codebooks that were trained using CMMN-filtered data
        if self.train_config.cmmn_filter == "normed-barycenter":
            self.cmmn_subfolder = "normed_filtered"
        else:
            self.cmmn_subfolder = "unfiltered"

        self._flatten_train_config()

    @property
    def subj_ids(self) -> List[int]:
        if self.eval_dataset == "emotion_study":
            return list(range(1, 8))  # test subjects
        elif self.eval_dataset == "cue":
            return list(range(1, 13))

    @property
    def path_to_train_output(self) -> Path:
        path = self.root / "results/emotion_study"
        return path

    @property
    def path_to_eval_data(self) -> Path:
        path = self.root / f"data/{self.eval_dataset}"
        return path

    @property
    def path_to_raw_data(self) -> Path:
        path = self.path_to_eval_data / "raw_data_and_IC_labels"
        return path

    @property
    def path_to_cmmn_filters(self) -> Path:
        if self.cmmn_filter is None:
            return None
        if self.eval_dataset == "cue":
            path = self.path_to_eval_data / f"cmmn_filters_resampled/{self.cmmn_filter}"
        else:
            path = self.path_to_eval_data / f"cmmn_filters/{self.cmmn_filter}"
        return path

    @property
    def path_to_preprocessed_data(self) -> Path:
        path = self.path_to_eval_data / "preprocessed_data"
        return path

    @property
    def path_to_centroid_assignments(self) -> Path:
        path = self.path_to_eval_data / "centroid_assignments" / self.cmmn_subfolder
        return path

    @property
    def path_to_codebooks(self) -> Path:
        if "bowav" in self.train_config.feature_extractor:
            if self.eval_dataset == "emotion_study":
                return self.path_to_train_output / "dictionaries" / self.cmmn_subfolder
            else:  # cue
                return (
                    self.path_to_train_output
                    / "dictionaries_resampled"
                    / self.cmmn_subfolder
                )
        else:
            raise ValueError(
                f"Codebooks not available for {self.train_config.feature_extractor}"
            )

    @property
    def path_to_classifier(self) -> dict[str, Path]:
        # Base path for all classifiers
        base_path = self.path_to_train_output / "classifier"

        if self.train_config.classifier_type == "ensembled_logistic":
            bowav_cfg = deepcopy(self.train_config)
            bowav_cfg.feature_extractor = "bowav"
            bowav_cfg.classifier_type = "logistic"
            psd_cfg = deepcopy(self.train_config)
            psd_cfg.feature_extractor = "psd_autocorr"
            psd_cfg.classifier_type = "logistic"
            return {
                "bowav": base_path
                / f"train_{build_base_classifier_name(bowav_cfg)}.pkl",
                "psd_autocorr": base_path
                / f"train_{build_base_classifier_name(psd_cfg)}.pkl",
            }
        else:
            return {
                self.feature_extractor: base_path
                / f"train_{build_base_classifier_name(self.train_config)}.pkl"
            }
