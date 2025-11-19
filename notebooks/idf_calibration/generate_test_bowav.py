# %%
import os

os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["NUMEXPR_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"

from pathlib import Path
import numpy as np

from icwaves.data.loading import load_data_bundles
from icwaves.evaluation.config import EvalConfig
from icwaves.evaluation.utils import get_eval_cmmn_filter_options
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


dataset = "epic"
minutes_per_ic = 10 if dataset == "epic" else None

train_cmmn_filter = None
use_idf = True
root = Path.home() / "personal_repos/ICWaves"
config_folder = root / "config/_private"
for classifier in ["logistic", "random_forest"]:
    for valseglen in [300, -1]:

        if dataset == "epic":
            if valseglen == -1:
                testseglen = 600  # cap to 10 minutes
            else:  # 5 minutes
                testseglen = valseglen
        else:
            testseglen = valseglen

        cmmn_str = get_cmmn_suffix(train_cmmn_filter)
        idf_str = "_idf" if use_idf else ""
        config_file_name = f"{classifier}_{sl2min(valseglen)}{cmmn_str}{idf_str}.txt"
        print(f"Reading config file: {config_file_name}")
        config_path = config_folder / config_file_name
        train_config = parse_config_file_args(config_path, "bowav")
        # %%
        eval_cmmn_filter_options = get_eval_cmmn_filter_options(
            dataset, train_cmmn_filter
        )
        for cmmn_filter in eval_cmmn_filter_options:
            config = EvalConfig(
                eval_dataset=dataset,
                train_config=train_config,
                root=root,
                cmmn_filter=cmmn_filter,
                minutes_per_ic=minutes_per_ic,
            )
            print(
                f"Path to centroid assignments: {config.path_to_centroid_assignments}"
            )
            data_bundles = load_data_bundles(config)
            centroid_assignments = data_bundles["bowav"].data
            labels = data_bundles["bowav"].labels
            expert_label_mask = data_bundles["bowav"].expert_label_mask
            subj_ind = data_bundles["bowav"].subj_ind
            if hasattr(data_bundles["bowav"], "zero_window_mask"):
                zero_window_mask = data_bundles["bowav"].zero_window_mask
            else:
                zero_window_mask = None
            # %%
            n_centroids = train_config.num_clusters
            feature_extractor = create_feature_extractor(
                "bowav", n_centroids=n_centroids
            )

            # %%
            clf_path = config.path_to_classifier["bowav"]
            _, best_params = load_estimator(clf_path)
            # %%
            srate = data_bundles["bowav"].srate
            converted_test_segment_lengths = convert_segment_length(
                float(testseglen),
                train_config.feature_extractor,
                srate,
                train_config.window_length,
            )

            # %%
            # We generate data under "test_segment" to make ad-hoc predictions on a smaller
            # segment. This was used to quickly prototype this, so we might be able to clean
            # this later. TODO.
            for use_full_time_series in [True, False]:
                data_folder = root / "data"
                bowav_folder = data_folder / dataset / "bowav"
                seglen_dir = "full" if use_full_time_series else "test_segment"
                bowav_folder = bowav_folder / seglen_dir
                bowav_folder.mkdir(parents=True, exist_ok=True)
                agg_method = best_params["input_or_output_aggregation_method"]
                train_seglen = best_params["training_segment_length"]

                for converted_testseglen in converted_test_segment_lengths:
                    if use_full_time_series:
                        segment = centroid_assignments
                    else:
                        segment = centroid_assignments[
                            :, :, : converted_testseglen["bowav"]
                        ]
                        if zero_window_mask is not None:
                            zero_window_mask = zero_window_mask[
                                :, : converted_testseglen["bowav"]
                            ]
                    segment = {"bowav": segment}
                    if zero_window_mask is not None:
                        segment["zero_window_mask"] = zero_window_mask
                    if agg_method == "majority_vote":
                        bowav = feature_extractor(segment, train_seglen)
                    else:
                        bowav = feature_extractor(segment, converted_testseglen)

                    output_base_filename = get_base_results_filename(config)
                    emotion_bowav_file = bowav_folder / f"{output_base_filename}.npz"
                    with emotion_bowav_file.open("wb") as f:
                        np.savez(
                            f,
                            bowav=bowav,
                            labels=labels,
                            expert_label_mask=expert_label_mask,
                            subj_ind=subj_ind,
                        )

# %%
# TODO: do we need to add the subject mask and expert label mask with the bowav features?
