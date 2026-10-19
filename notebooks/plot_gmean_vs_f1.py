"""
This script plots the metrics GMean vs Brain F1 score on the EPIC
dataset. The purpose of this is to show that those two metrics are
strictly positively correlated. E.g., the point with the highest F1
score is not the one with the highest GMean score. This is expected,
as the F1 score do no take into account true negatives (as specificity
does).
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Get the script directory and construct absolute paths
script_dir = Path(__file__).parent
project_root = script_dir.parent

# Load data
metrics = pd.read_csv(
    project_root / "results/epic/evaluation/global_metrics_summary.csv"
)
f1_scores = pd.read_csv(project_root / "results/mean_std_brain_f1_raw.csv")

# Filter F1 scores for EPIC dataset with prediction_window=10
f1_epic = f1_scores[
    (f1_scores["eval_dataset"] == "epic") & (f1_scores["prediction_window"] == 10.0)
].copy()

# Create merge key with consistent types
metrics["merge_key"] = (
    metrics["feature_extractor"]
    + "_"
    + metrics["classifier_type"]
    + "_"
    + metrics["validation_segment_len"].astype(float).astype(str)
    + "_"
    + metrics["cmmn_filter"].fillna("None")
)

f1_epic["merge_key"] = (
    f1_epic["feature_extractor"]
    + "_"
    + f1_epic["classifier_type"]
    + "_"
    + f1_epic["validation_segment_len"].astype(str)
    + "_"
    + f1_epic["cmmn_filter"].fillna("None")
)

# Merge datasets
merged = pd.merge(metrics, f1_epic, on="merge_key")

print(f"Found {len(merged)} matching configurations")

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(merged["gmean"], merged["mean_f1"], alpha=0.6, s=100)

plt.xlabel("Geometric Mean (Sensitivity & Specificity)", fontsize=12)
plt.ylabel("Mean Brain F1 Score", fontsize=12)
plt.title("EPIC Dataset: Gmean vs Mean F1 Score", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
