"""
TF-IDF Analysis Verification Script
Reproduces all key evidence for why TF-IDF scaling hurts cross-dataset performance
"""

# %%
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import f1_score
from copy import deepcopy


def compute_subject_f1_score(y_true, y_pred, expert_mask):
    """Compute F1 scores per subject on expert-annotated samples only and return mean"""

    if expert_mask.sum() == 0:  # Skip if no expert annotations for this subject
        return 0.0
    subj_y_true = y_true[expert_mask]
    subj_y_pred = y_pred[expert_mask]
    if len(np.unique(subj_y_true)) > 1:  # Only if subject has brain class
        f1 = f1_score(subj_y_true, subj_y_pred, labels=[0], average=None)[0]
        return f1
    else:
        print(f"Subject has only one class: {np.unique(subj_y_true)}")
        return 0.0


# Setup paths
root = Path(__file__).parents[2]

# Load trained classifier with TF-IDF scaler
classifier_path = (
    root
    / "results/emotion_study/classifier"
    / "train_random_forest_bowav_valSegLen300_cmmn-None_idf.pkl"
)
with open(classifier_path, "rb") as f:
    results = pickle.load(f)

# Check the structure of the best_estimator
print("Best estimator structure:", type(results["best_estimator"]))
if hasattr(results["best_estimator"], "named_steps"):
    print("Named steps:", list(results["best_estimator"].named_steps.keys()))
    pipeline = results["best_estimator"]
else:
    # If it's not a pipeline, it might be the classifier directly
    raise ValueError(
        "The loaded object is not a pipeline and does not have named_steps."
    )

# Load training data (emotion_study)
output_base_filename = "random_forest_bowav_valSegLen300_cmmn-None_idf"
train_path = root / "data/emotion_study/bowav/train" / f"{output_base_filename}.npz"
with np.load(train_path, allow_pickle=True) as f:
    bowav_train = f["bowav"]

bowav_train = bowav_train.reshape(-1, bowav_train.shape[-1])
# Load test data (cue) using the full time series (50 min)
# We want to use the full time series for idf calibration
# cmmn_filter_options = [None, "unnormed-barycenter", "subj_to_subj"]
cmmn_filter = None
test_path = root / "data/cue/bowav/full" / f"{output_base_filename}.npz"
with np.load(test_path, allow_pickle=True) as f:
    bowav_full_cue = f["bowav"]
    subj_ind_full_cue = f["subj_ind"]

# print shape of bowav_full_cue
print(f"Shape of bowav using all 50-min ICs: {bowav_full_cue.shape}")

# n_seg = bowav_full_cue.shape[1]
# bowav_full_cue = bowav_full_cue.reshape(-1, bowav_full_cue.shape[-1])
# subj_ind_full_cue = np.repeat(subj_ind_full_cue, n_seg)

# %%
test_path = root / "data/cue/bowav/test_segment" / f"{output_base_filename}.npz"
with np.load(test_path, allow_pickle=True) as f:
    bowav_cue = f["bowav"]
    labels = f["labels"]
    expert_label_mask = f["expert_label_mask"]
    subj_ind = f["subj_ind"]

print(f"Shape of bowav using first 5-min of the ICs: {bowav_cue.shape}")
n_ics, n_seg, n_feat = bowav_cue.shape
bowav_cue = bowav_cue.reshape(-1, bowav_cue.shape[-1])
labels = np.repeat(labels, n_seg)
expert_label_mask = np.repeat(expert_label_mask, n_seg)
subj_ind = np.repeat(subj_ind, n_seg)
n_subj = np.unique(subj_ind).shape[0]

f1_original = []
f1_clipped = []
f1_raw = []
# Calculate term frequency ratios (source/target)
train_mean_vals = np.mean(bowav_train, axis=0)
for sid in np.unique(subj_ind):
    subj_mask = subj_ind == sid
    subj_mask_full_cue = subj_ind_full_cue == sid
    cue_mean_vals = np.mean(bowav_full_cue[subj_mask_full_cue], axis=(0, 1))
    tf_ratio = np.divide(
        train_mean_vals,
        cue_mean_vals,
        out=np.ones_like(train_mean_vals),  # Default to 1 for zero division
        where=cue_mean_vals != 0,
    )

    # Create log-smoothed scaled IDF weights (new approach)
    clipped_tf_ratio = np.clip(tf_ratio, 0.1, 10.0)

    # Create clipped IDF weights without logarithmic smoothing
    clipped_idf = pipeline["scaler"].idf_ * clipped_tf_ratio

    # %%

    # Test clipped TF-IDF scaling (without logarithmic smoothing)
    calibrated_pipeline = deepcopy(pipeline)
    calibrated_pipeline["scaler"].idf_ = clipped_idf
    calibrated_pipeline["scaler"]._tfidf_transformer.idf_ = clipped_idf

    pred_clipped = calibrated_pipeline.predict(bowav_cue[subj_mask])
    f1_clipped.append(
        compute_subject_f1_score(
            labels[subj_mask], pred_clipped, expert_label_mask[subj_mask]
        )
    )

    # Test original TF-IDF scaling
    # pipeline["scaler"].idf_ = og_idf
    pred_original = pipeline.predict(bowav_cue[subj_mask])
    f1_original.append(
        compute_subject_f1_score(
            labels[subj_mask], pred_original, expert_label_mask[subj_mask]
        )
    )

    # Test raw features (no scaling)
    pred_raw = pipeline["clf"].predict(bowav_cue[subj_mask])
    f1_raw.append(
        compute_subject_f1_score(
            labels[subj_mask], pred_raw, expert_label_mask[subj_mask]
        )
    )

print(f"F1 score for brain class (label=0) on cue dataset:")
print(f"  Raw features: {np.mean(f1_raw):.3f}")
print(f"  Original TF-IDF: {np.mean(f1_original):.3f}")
print(f"  clipped_tf_ratio: {np.mean(f1_clipped):.3f}")
