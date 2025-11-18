# %%
from pathlib import Path
import numpy as np

from icwaves.data.loading import load_data_bundles
from icwaves.evaluation.config import EvalConfig
from icwaves.factories import create_feature_extractor
from icwaves.feature_extractors.utils import convert_segment_length
from icwaves.file_utils import (
    get_cmmn_suffix,
    parse_config_file_args,
)
from icwaves.evaluation.evaluation import get_base_results_filename, load_estimator


# %%
def sl2min(sl):
    return "50min" if sl == -1 else "5min"


dataset = "emotion_study"
classifiers = ["random_forest", "logistic"]
val_segment_lengths = [300, -1]
train_cmmn_filter = None
use_idf = True
root = Path.home() / "personal_repos/ICWaves"
config_folder = root / "config/_private"

# %%
# Loop through all combinations of classifier and val_segment_length
for classifier in classifiers:
    for val_segment_length in val_segment_lengths:
        print(
            f"\nProcessing classifier: {classifier}, val_segment_length: {val_segment_length}"
        )

        cmmn_str = get_cmmn_suffix(train_cmmn_filter)
        idf_str = "_idf" if use_idf else ""
        config_file_name = (
            f"{classifier}_{sl2min(val_segment_length)}{cmmn_str}{idf_str}.txt"
        )
        print(f"Reading config file: {config_file_name}")
        config_path = config_folder / config_file_name
        train_config = parse_config_file_args(config_path, "bowav")

        config = EvalConfig(
            eval_dataset=dataset,
            train_config=train_config,
            root=root,
            cmmn_filter=None,
        )

        data_bundles = load_data_bundles(train_config)
        centroid_assignments = data_bundles["bowav"].data
        labels = data_bundles["bowav"].labels
        expert_label_mask = data_bundles["bowav"].expert_label_mask
        subj_ind = data_bundles["bowav"].subj_ind
        srate = data_bundles["bowav"].srate

        n_centroids = train_config.num_clusters
        feature_extractor = create_feature_extractor("bowav", n_centroids=n_centroids)

        clf_path = config.path_to_classifier["bowav"]
        estimator, best_params = load_estimator(clf_path)
        print(f"Best params: {best_params}")

        test_segment_length = convert_segment_length(
            float(val_segment_length),
            train_config.feature_extractor,
            srate,
            train_config.window_length,
        )[0]

        emotion_bowav_dir = root / "data/emotion_study/bowav/train/"
        agg_method = best_params["input_or_output_aggregation_method"]
        train_segment_length = best_params["training_segment_length"]

        segment = {"bowav": centroid_assignments}
        if agg_method == "majority_vote":
            bowav = feature_extractor(segment, train_segment_length)
        else:
            bowav = feature_extractor(segment, test_segment_length)

        output_base_filename = get_base_results_filename(config)
        emotion_bowav_file = emotion_bowav_dir / f"{output_base_filename}.npz"
        print(f"Saving to: {emotion_bowav_file}")
        with open(emotion_bowav_file, "wb") as f:
            np.savez(
                f,
                bowav=bowav,
                labels=labels,
                expert_label_mask=expert_label_mask,
                subj_ind=subj_ind,
            )
        print(f"Successfully saved {emotion_bowav_file}")

# %%
