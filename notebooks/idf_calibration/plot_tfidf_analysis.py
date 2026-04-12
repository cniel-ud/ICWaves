# %%
"""
Visualization script for TF-IDF analysis findings
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# %%

# Setup
root = Path(__file__).parents[2]

# Load data
output_base_filename = "random_forest_bowav_valSegLen300_cmmn-None_idf"
classifier_path = (
    root / "results/emotion_study/classifier" / f"train_{output_base_filename}.pkl"
)
with open(classifier_path, "rb") as f:
    results = pickle.load(f)
scaler = results["best_estimator"]["scaler"]

# Load training data (emotion_study)
train_dir = root / "data/emotion_study/bowav/train"
train_path = train_dir / f"{output_base_filename}.npz"
with np.load(train_path, allow_pickle=True) as f:
    bowav_train = f["bowav"]

bowav_train = bowav_train.reshape(-1, bowav_train.shape[-1])

# Load test data (cue) using the full time series (50 min)
# We want to use the full time series for idf calibration
test_dir = root / "data/cue/bowav/full"
test_path = test_dir / f"{output_base_filename}.npz"
with np.load(test_path, allow_pickle=True) as f:
    bowav_full_cue = f["bowav"]

bowav_full_cue = bowav_full_cue.reshape(-1, bowav_full_cue.shape[-1])

# Calculate key metrics
doc_freq_train_norm = (bowav_train > 0).sum(axis=0) / bowav_train.shape[0]
doc_freq_cue_norm = (bowav_full_cue > 0).sum(axis=0) / bowav_full_cue.shape[0]
freq_ratio = np.divide(
    doc_freq_cue_norm,
    doc_freq_train_norm,
    out=np.zeros_like(doc_freq_cue_norm),
    where=doc_freq_train_norm != 0,
)

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(
    "TF-IDF Scaling Analysis: Why Raw Features Outperform",
    fontsize=14,
    fontweight="bold",
)

# Plot 1: IDF distribution
axes[0, 0].hist(scaler.idf_, bins=30, alpha=0.7, color="blue")
axes[0, 0].set_xlabel("IDF Weight")
axes[0, 0].set_ylabel("Number of Features")
axes[0, 0].set_title("IDF Weight Distribution\n(Learned from emotion_study)")
axes[0, 0].axvline(
    scaler.idf_.mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {scaler.idf_.mean():.2f}",
)
axes[0, 0].legend()

# Plot 2: Frequency ratio vs IDF
scatter = axes[0, 1].scatter(scaler.idf_, freq_ratio, alpha=0.6, s=20)
axes[0, 1].set_xlabel("IDF Weight")
axes[0, 1].set_ylabel("Frequency Ratio (Cue/Train)")
axes[0, 1].set_title("Problem: High IDF Features\nBecome Common in Cue")
axes[0, 1].axhline(1, color="red", linestyle="--", alpha=0.5)
axes[0, 1].axhline(
    2, color="orange", linestyle="--", alpha=0.5, label="2x more frequent"
)
axes[0, 1].set_yscale("log")
axes[0, 1].legend()

# Plot 3: Document frequency comparison
valid_mask = (doc_freq_train_norm > 0) & (doc_freq_cue_norm > 0)
axes[1, 0].scatter(
    doc_freq_train_norm[valid_mask], doc_freq_cue_norm[valid_mask], alpha=0.6, s=20
)
axes[1, 0].plot([0, 1], [0, 1], "r--", alpha=0.5, label="Perfect match")
axes[1, 0].set_xlabel("Training Frequency")
axes[1, 0].set_ylabel("Cue Frequency")
axes[1, 0].set_title("Document Frequency Mismatch\n(Correlation: X)")
axes[1, 0].legend()

# Plot 4: Codebook analysis
n_codebooks = 7
n_centroids = 128
codebook_ratios = []
codebook_idfs = []

for cb in range(n_codebooks):
    start_idx = cb * n_centroids
    end_idx = start_idx + n_centroids
    cb_ratio = freq_ratio[start_idx:end_idx].mean()
    cb_idf = scaler.idf_[start_idx:end_idx].mean()
    codebook_ratios.append(cb_ratio)
    codebook_idfs.append(cb_idf)

bars = axes[1, 1].bar(range(n_codebooks), codebook_ratios, color="skyblue", alpha=0.7)
axes[1, 1].set_xlabel("Codebook")
axes[1, 1].set_ylabel("Mean Frequency Ratio")
axes[1, 1].set_title("Codebook-wise Distribution Shift")
axes[1, 1].set_xticks(range(n_codebooks))

# Add IDF values as text on bars
for i, (bar, idf) in enumerate(zip(bars, codebook_idfs)):
    height = bar.get_height()
    axes[1, 1].text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.5,
        f"IDF: {idf:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

plt.tight_layout()
plt.savefig("tfidf_analysis_plots.png", dpi=300, bbox_inches="tight")
plt.show()

# Summary statistics
print("KEY STATISTICS:")
print(f"Features >10x more frequent in cue: {(freq_ratio > 10).sum()}")
print(
    f"Features <0.1x as frequent in cue: {((freq_ratio < 0.1) & (doc_freq_cue_norm > 0)).sum()}"
)
print(f"Most extreme increase: {freq_ratio.max():.1f}x (Feature {freq_ratio.argmax()})")
print(f"Codebook 1 most affected: {codebook_ratios[1]:.1f}x average ratio")

# %%
