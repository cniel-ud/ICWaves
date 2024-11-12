from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class EvalConfig:
    dataset_name: str  # 'emotion_study' or 'cue'
    classifier_type: str  # 'random_forest' or 'logistic'
    feature_extractor: str  # 'bowav' or 'psd_autocorr'
    root: Path
    train_dataset: Optional[str] = None  # For cross-dataset evaluation
    window_length: float = 1.5
    minutes_per_ic: float = 50.0
    num_clusters: int = 128
    centroid_length: float = 1.0
    validation_segment_length = -1

    @property
    def subj_ids(self) -> List[int]:
        if self.dataset_name == "emotion_study":
            return list(range(1, 8))  # test subjects
        elif self.dataset_name == "cue":
            return list(range(1, 13))
        raise ValueError(f"Unknown dataset {self.dataset_name}")

    @property
    def paths(self):
        base = self.root
        valseglen = (
            "none"
            if self.validation_segment_length == -1
            else int(self.validation_segment_length)
        )
        results_file = Path(
            base,
            "results",
            f"{self.dataset_name}",
            f"final_{self.classifier_type}_{self.feature_extractor}_valSegLen{valseglen}.pkl",
        )
        # results/{train_dataset}/classifier
        train_dataset = self.train_dataset or self.dataset_name
        # Save final model and results
        return {
            "raw_data": base / f"data/{self.dataset_name}/raw_data_and_IC_labels",
            "results_file": results_file,
            "preprocessed": base / f"data/{self.dataset_name}/preprocessed_data",
            "codebooks": base / f"results/{train_dataset}/dictionaries",
            "centroid_assignments": base
            / f"data/{self.dataset_name}/centroid_assignments",
        }
