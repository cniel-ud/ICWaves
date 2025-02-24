from dataclasses import dataclass
from pathlib import Path
from typing import List
from icwaves.file_utils import get_validation_segment_length_string


@dataclass
class EvalConfig:
    """Configuration for the evaluation pipeline.

    This configuration assumes that both the BoWav dictionaries and the
    classifier was trained on the 'emotion_study' dataset, using subjects
    8 to 35 (excluding subject 22, which is missing).
    """

    eval_dataset: str  # 'emotion_study' or 'cue'
    classifier_type: str  # 'random_forest' or 'logistic'
    feature_extractor: str  # 'bowav' or 'psd_autocorr'
    root: Path
    window_length: float = 1.5
    minutes_per_ic: float = 50.0
    num_clusters: int = 128
    centroid_length: float = 1.0
    validation_segment_length: int = -1
    codebook_minutes_per_ic: float = 50.0
    codebook_ics_per_subject: int = 2

    def __post_init__(self):
        if self.eval_dataset not in ["emotion_study", "cue"]:
            raise ValueError(f"Unknown eval dataset {self.eval_dataset}")
        if self.classifier_type not in ["random_forest", "logistic"]:
            raise ValueError(f"Unknown classifier type {self.classifier_type}")
        if self.feature_extractor not in [
            "bowav",
            "psd_autocorr",
            "bowav_psd_autocorr",
        ]:
            raise ValueError(f"Unknown feature extractor {self.feature_extractor}")

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
    def path_to_preprocessed_data(self) -> Path:
        path = self.path_to_eval_data / "preprocessed_data"
        return path

    @property
    def path_to_centroid_assignments(self) -> Path:
        path = self.path_to_eval_data / "centroid_assignments"
        return path

    @property
    def path_to_codebooks(self) -> Path:
        if "bowav" in self.feature_extractor:
            if self.eval_dataset == "emotion_study":
                return self.path_to_train_output / "dictionaries"
            else:  # cue
                return self.path_to_train_output / "dictionaries_resampled"
        else:
            raise ValueError(f"Codebooks not available for {self.feature_extractor}")

    @property
    def path_to_classifier(self) -> Path:
        valseglen = get_validation_segment_length_string(
            int(self.validation_segment_length)
        )
        path = (
            self.path_to_train_output
            / "classifier"
            / f"train_{self.classifier_type}_{self.feature_extractor}_valSegLen{valseglen}.pkl"
        )
        return path
