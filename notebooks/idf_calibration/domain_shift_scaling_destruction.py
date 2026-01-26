# %%
"""
Analysis of how extreme domain shifts destroy feature scaling
and why 10x is a reasonable tolerance threshold
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Load data
root = Path(__file__).parents[2]
classifier_path = (
    root
    / "results/emotion_study/classifier"
    / "train_random_forest_bowav_valSegLen300_cmmn-None_idf.pkl"
)
with open(classifier_path, "rb") as f:
    results = pickle.load(f)
scaler = results["best_estimator"]["scaler"]

train_path = root / "data/emotion_study/bowav/train/5min" / "5min.npz"
with np.load(train_path, allow_pickle=True) as f:
    bowav_train = f["bowav"]
bowav_train = bowav_train.reshape(-1, bowav_train.shape[-1])

test_path = root / "data/cue/bowav/full/5min/cmmn-None" / "5min.npz"
with np.load(test_path, allow_pickle=True) as f:
    bowav_full_cue = f["bowav"]
bowav_full_cue = bowav_full_cue.reshape(-1, bowav_full_cue.shape[-1])

# Calculate tf_ratio and scaling factors
train_mean_vals = np.mean(bowav_train, axis=0)
cue_mean_vals = np.mean(bowav_full_cue, axis=0)
tf_ratio = np.divide(
    train_mean_vals,
    cue_mean_vals,
    out=np.ones_like(train_mean_vals),
    where=cue_mean_vals != 0,
)

print("HOW EXTREME DOMAIN SHIFTS DESTROY FEATURE SCALING")
print("=" * 55)

# 1. Demonstrate the scaling destruction mechanism
print("\n1. SCALING DESTRUCTION MECHANISM")
print("-" * 35)

# Original IDF weights
original_weights = scaler.idf_
clipped_tf_ratio = np.clip(tf_ratio, 0.1, 10)
naive_weights = scaler.idf_ * tf_ratio
clipped_weights = scaler.idf_ * clipped_tf_ratio

print(
    f"Original IDF range: [{original_weights.min():.3f}, {original_weights.max():.3f}]"
)
print(f"Naive scaling range: [{naive_weights.min():.3f}, {naive_weights.max():.3f}]")
print(
    f"Clipped scaling range: [{clipped_weights.min():.3f}, {clipped_weights.max():.3f}]"
)

# Show the amplification factor
amplification = naive_weights.max() / original_weights.max()
print(f"Maximum amplification factor: {amplification:.0f}x")

# 2. Identify the most destructive features
print("\n2. MOST DESTRUCTIVE FEATURES")
print("-" * 30)

# Features with extreme tf_ratios
extreme_indices = np.argsort(tf_ratio)[::-1][:10]  # Top 10 most extreme

print("Top 10 most extreme tf_ratios (features that became very rare in cue):")
for i, idx in enumerate(extreme_indices):
    cb = idx // 128
    centroid = idx % 128
    print(
        f"  {i+1:2d}. Feature {idx} (CB{cb}, C{centroid}): "
        f"tf_ratio: {tf_ratio[idx]:8.1f}, "
        f"IDF: {scaler.idf_[idx]:.3f}, "
        f"After naive scaling: {naive_weights[idx]:8.1f}, "
        f"After clipped scaling: {clipped_weights[idx]:8.1f}, "
    )

# 3. Show how these features dominate
print("\n3. FEATURE DOMINANCE ANALYSIS")
print("-" * 30)

# Calculate feature contributions to total scaling
total_naive_weight = np.sum(naive_weights)
total_clipped_weight = np.sum(clipped_weights)

# Top 10 most extreme features' contribution
top10_naive_contrib = np.sum(naive_weights[extreme_indices]) / total_naive_weight
top10_clipped_contrib = np.sum(clipped_weights[extreme_indices]) / total_clipped_weight

print(f"Top 10 extreme features contribute:")
print(f"  Naive scaling: {top10_naive_contrib:.1%} of total weight")
print(f"  Clipped scaling: {top10_clipped_contrib:.1%} of total weight")

# 4. Why 10x is reasonable
print("\n4. WHY 10x IS A REASONABLE THRESHOLD")
print("-" * 40)

# Test different thresholds
thresholds = [2, 5, 10, 20, 50, 100]
for thresh in thresholds:
    clipped_at_thresh = np.clip(tf_ratio, 1 / thresh, thresh)
    weights_at_thresh = scaler.idf_ * clipped_at_thresh

    # Calculate how many features are clipped
    clipped_count = ((tf_ratio < 1 / thresh) | (tf_ratio > thresh)).sum()
    clipped_pct = clipped_count / len(tf_ratio) * 100

    # Calculate weight range
    weight_range = weights_at_thresh.max() / weights_at_thresh.min()

    print(
        f"  {thresh:3d}x threshold: {clipped_pct:4.1f}% clipped, weight range: {weight_range:6.1f}x"
    )

# 5. Semantic interpretation
print("\n5. SEMANTIC INTERPRETATION")
print("-" * 30)

print("What different tf_ratio values mean:")
print("  tf_ratio = 0.01 (1/100): Feature 100x MORE common in target → likely artifact")
print("  tf_ratio = 0.1  (1/10):  Feature 10x MORE common in target → suspicious")
print("  tf_ratio = 1:            Feature equally common → ideal")
print("  tf_ratio = 10:           Feature 10x LESS common in target → suspicious")
print("  tf_ratio = 100:          Feature 100x LESS common in target → likely artifact")

# 6. Create visualizations
print("\n6. CREATING VISUALIZATIONS")
print("-" * 25)

# Plot 1: Weight distributions
plt.figure(figsize=(8, 6))
plt.hist(original_weights, bins=30, alpha=0.6, label="Original IDF", color="blue")
plt.hist(clipped_weights, bins=30, alpha=0.6, label="Clipped [0.1, 10]", color="green")
plt.xlabel("Weight Value")
plt.ylabel("Count")
plt.title("Weight Distributions")
plt.legend()
plt.xlim(0, 20)
plt.tight_layout()
plt.savefig("weight_distributions.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 2: Extreme weights (log scale)
plt.figure(figsize=(8, 6))
plt.hist(naive_weights, bins=50, alpha=0.6, label="Naive scaling", color="red")
plt.hist(clipped_weights, bins=50, alpha=0.6, label="Clipped [0.1, 10]", color="green")
plt.xlabel("Weight Value (log scale)")
plt.ylabel("Count")
plt.title("Weight Distributions (Log Scale)")
plt.xscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("weight_distributions_log_scale.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 3: tf_ratio vs resulting weight
plt.figure(figsize=(8, 6))
plt.scatter(tf_ratio, naive_weights, alpha=0.5, s=20, color="red", label="Naive")
plt.scatter(tf_ratio, clipped_weights, alpha=0.5, s=20, color="green", label="Clipped")
plt.axvline(0.1, color="black", linestyle="--", alpha=0.5)
plt.axvline(10, color="black", linestyle="--", alpha=0.5)
plt.xlabel("TF Ratio")
plt.ylabel("Final Weight")
plt.title("TF Ratio vs Final Weight")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.savefig("tf_ratio_vs_weight.png", dpi=300, bbox_inches="tight")
plt.show()

# Plot 4: Cumulative Distribution Function (CDF) of weights
sorted_naive = np.sort(naive_weights)
sorted_clipped = np.sort(clipped_weights)
# Create CDF: cumulative count / total count
cdf_naive = np.arange(1, len(sorted_naive) + 1) / len(sorted_naive)
cdf_clipped = np.arange(1, len(sorted_clipped) + 1) / len(sorted_clipped)

plt.figure(figsize=(8, 6))
plt.plot(sorted_naive, cdf_naive, "r-", label="Naive scaling", linewidth=2)
plt.plot(sorted_clipped, cdf_clipped, "g-", label="Clipped scaling", linewidth=2)
plt.axvline(
    np.median(sorted_naive),
    color="red",
    linestyle="--",
    alpha=0.5,
    label="Naive median",
)
plt.axvline(
    np.median(sorted_clipped),
    color="green",
    linestyle="--",
    alpha=0.5,
    label="Clipped median",
)
plt.xlabel("Weight Value")
plt.ylabel("Cumulative Probability")
plt.title("Cumulative Distribution Function of Weights")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xscale("log")
plt.tight_layout()
plt.savefig("weight_cdf.png", dpi=300, bbox_inches="tight")
plt.show()

# 7. The key insight
print("\n7. KEY INSIGHT: THE TOLERANCE PRINCIPLE")
print("-" * 40)
print("The 10x threshold represents a 'tolerance principle':")
print("• Accept that domains will differ - some features will be more/less common")
print("• But don't let extreme differences (>10x) dominate the entire scaling")
print("• 10x allows for substantial domain differences while preventing artifacts")
print(
    "• It's the sweet spot between being too restrictive (2x) and too permissive (100x)"
)

# Calculate how much "signal" vs "noise" we're dealing with
reasonable_features = (tf_ratio >= 0.1) & (tf_ratio <= 10)
suspicious_features = ((tf_ratio < 0.1) | (tf_ratio > 10)) & (tf_ratio < 100)
artifact_features = tf_ratio >= 100

print(f"\nFeature categorization:")
print(
    f"• Reasonable differences (0.1-10x): {reasonable_features.sum()} ({reasonable_features.mean():.1%})"
)
print(
    f"• Suspicious differences (10-100x): {suspicious_features.sum()} ({suspicious_features.mean():.1%})"
)
print(
    f"• Likely artifacts (>100x): {artifact_features.sum()} ({artifact_features.mean():.1%})"
)

# %%
