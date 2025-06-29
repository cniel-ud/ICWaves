from dataclasses import dataclass
from pathlib import Path
from typing import List
from icwaves.file_utils import get_cmmn_suffix, get_validation_segment_length_string
from typing import Optional

SUPPORTED_CLASSIFIERS = ["random_forest", "logistic", "ensembled_logistic"]
SUPPORTED_DATASETS = ["emotion_study", "cue"]
SUPPORTED_FEATURES = ["bowav", "psd_autocorr", "bowav_psd_autocorr"]
SUPPORTED_CMNN_FILTERS = ["original", "subj_to_subj"]


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
    cmmn_filter: Optional[str] = None
    is_cmmn_filter_resampled: bool = False
    is_classifier_trained_on_normalized_data: bool = False

    def __post_init__(self):
        if self.eval_dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unknown eval dataset {self.eval_dataset}")
        if self.classifier_type not in SUPPORTED_CLASSIFIERS:
            raise ValueError(f"Unknown classifier type {self.classifier_type}")
        if self.feature_extractor not in SUPPORTED_FEATURES:
            raise ValueError(f"Unknown feature extractor {self.feature_extractor}")
        if (
            self.cmmn_filter is not None
            and self.cmmn_filter not in SUPPORTED_CMNN_FILTERS
        ):
            raise ValueError(f"Unknown cmmn filter {self.cmmn_filter}")
        if self.cmmn_filter is not None and self.eval_dataset != "cue":
            raise ValueError(f"cmmn filter only supported for cue dataset")

        # TODO: if the classifier is not trained on filtered 'emotion' data,
        # should we still use the filtered codebooks when computing bowav for 'cue'?
        if self.is_classifier_trained_on_normalized_data:
            self.cmmn_subfolder = "normed_filtered"
        else:
            self.cmmn_subfolder = "unfiltered"

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
        if self.is_cmmn_filter_resampled:
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
        if "bowav" in self.feature_extractor:
            if self.eval_dataset == "emotion_study":
                return self.path_to_train_output / "dictionaries" / self.cmmn_subfolder
            else:  # cue
                return (
                    self.path_to_train_output
                    / "dictionaries_resampled"
                    / self.cmmn_subfolder
                )
        else:
            raise ValueError(f"Codebooks not available for {self.feature_extractor}")

    @property
    def path_to_classifier(self) -> dict[str, Path]:
        valseglen = get_validation_segment_length_string(
            int(self.validation_segment_length)
        )
        if self.is_classifier_trained_on_normalized_data:
            cmmn_suffix = get_cmmn_suffix(
                self.cmmn_filter, self.is_cmmn_filter_resampled
            )
        else:
            cmmn_suffix = get_cmmn_suffix(None)
        # Base path for all classifiers
        base_path = self.path_to_train_output / "classifier"

        # Common filename pattern
        filename_pattern = "train_{classifier}_{feature}_valSegLen{valseglen}{cmmn}.pkl"

        if self.classifier_type == "ensembled_logistic":
            return {
                "bowav": base_path
                / f"{filename_pattern}".format(
                    classifier="logistic",
                    feature="bowav",
                    valseglen=valseglen,
                    cmmn=cmmn_suffix,
                ),
                "psd_autocorr": base_path
                / f"{filename_pattern}".format(
                    classifier="logistic",
                    feature="psd_autocorr",
                    valseglen=valseglen,
                    cmmn=cmmn_suffix,
                ),
            }
        else:
            return {
                self.feature_extractor: base_path
                / f"{filename_pattern}".format(
                    classifier=self.classifier_type,
                    feature=self.feature_extractor,
                    valseglen=valseglen,
                    cmmn=cmmn_suffix,
                )
            }
