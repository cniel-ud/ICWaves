#!/usr/bin/env python3
"""
Why does Heart-codebook centroid 507 rank among the top-10 positive Brain-class
SHAP centroids of the 5-min random forest on Cue?

The covariance used for the ranking is c_t ~ p(1-p) (mu_brain - mu_other), the
difference between the mean SHAP value of the feature on Brain and on non-Brain
segments. This script explains where that difference comes from for 507:

1. Source vs target occurrence rates (raw BoWav rates, Brain vs non-Brain): what
   the forest learned on the Emotion Study, and how the rate shifts on Cue.
2. Scaled feature values seen by the classifier: SHAP background (source IDF)
   vs explained Cue segments (IDF-calibrated).
3. SHAP decomposition for the top-10 positive centroids: mean SHAP on Brain and
   non-Brain segments, their difference, the covariance, the fraction of samples
   with non-zero SHAP, and the rank correlation between rate and SHAP.
4. Context: correlation of the rate of 507 with the top Brain-codebook centroids.

With --ablation, it also checks whether the classifier needs the feature: Brain F1
on Cue (first 5 minutes of each IC, expert-labeled ICs, mean over subjects,
per-subject IDF calibration, as in the paper) after mean-imputing it without
retraining, and after refitting with the saved best hyperparameters and the
feature zeroed, against control centroids of similar Brain-segment prevalence.

Uses the cached BoWav files and the SHAP cache in
results/feature_importance/random_forest_cue/.

Usage:
    python scripts/heart_centroid_analysis.py [--ablation] [--n-controls 3]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from feature_importance import load_data  # noqa: E402

from icwaves.evaluation.config import EvalConfig
from icwaves.evaluation.evaluation import load_estimator
from icwaves.evaluation.utils import get_base_results_filename, make_calibrate_idf_fn
from icwaves.file_utils import parse_config_file_args

app = typer.Typer()

HEART = 507
N_CENTROIDS = 128
BRAIN = 0
CODEBOOKS = ["Brain", "Muscle", "Eye", "Heart", "Line Noise", "Channel Noise", "Other"]


def load_bowav(path):
    d = np.load(path)
    return d["bowav"], d["labels"], d["expert_label_mask"].astype(bool), d["subj_ind"]


def brain_f1_cue(estimator, calibrate, X_eval, labels, expert, subj, transform=None):
    """Paper-style Brain F1: first 5-min segment of each expert-labeled IC,
    per-subject IDF calibration, mean over subjects. Also pooled AUC."""
    f1s, p_all, y_all = [], [], []
    for s in np.unique(subj):
        m = subj == s
        model = calibrate(estimator, m)
        Xs = model["scaler"].transform(X_eval[m][:, 0, :]).toarray()
        if transform is not None:
            Xs = transform(Xs)
        proba = model["clf"].predict_proba(Xs)
        pred = model["clf"].classes_[proba.argmax(1)]
        e = expert[m]
        y = labels[m][e] == BRAIN
        f1s.append(f1_score(y, pred[e] == BRAIN))
        p_all.append(proba[e, list(model["clf"].classes_).index(BRAIN)])
        y_all.append(y)
    return np.mean(f1s), roc_auc_score(np.concatenate(y_all), np.concatenate(p_all))


def refit(estimator, best_params, X_train, y_train, expert_train, seed):
    """Refit the pipeline with the saved best hyperparameters (no HPO)."""
    params = {
        k: v
        for k, v in best_params.items()
        if k.startswith("clf__") or k.startswith("scaler__")
    }
    est = clone(estimator).set_params(**params)
    est.set_params(clf__random_state=seed, clf__n_jobs=-1)
    w = np.ones(len(y_train))
    w[expert_train] = best_params.get("expert_weight", 1)
    n_seg = X_train.shape[1]
    est.fit(
        np.vstack(X_train),
        np.repeat(y_train, n_seg),
        clf__sample_weight=np.repeat(w, n_seg),
    )
    return est


def run_ablation(
    estimator, best_params, calibrate, train, evaluation, rates, is_brain, top, seed,
    n_controls,
):
    X_train, lab_train, exp_train = train
    X_eval, lab_eval, exp_eval, subj_eval = evaluation
    rng = np.random.default_rng(seed)
    presence = (rates[is_brain] > 0).mean(0)
    idx = np.arange(rates.shape[1])
    near = np.where(
        (np.abs(presence - presence[HEART]) < 0.03) & (idx != HEART) & ~np.isin(idx, top)
    )[0]
    controls = sorted(rng.choice(near, size=min(n_controls, len(near)), replace=False))
    print(f"\nAblation (controls with Brain-segment presence within 3% of "
          f"{presence[HEART]:.1%}: {controls})")

    Xtr_scaled = estimator["scaler"].transform(np.vstack(X_train)).toarray()
    rows = []
    f1, auc = brain_f1_cue(estimator, calibrate, X_eval, lab_eval, exp_eval, subj_eval)
    rows.append(("saved model", "-", f1, auc))
    for c in [HEART] + list(controls):
        mean_c = Xtr_scaled[:, c].mean()

        def impute(X, c=c, v=mean_c):
            X = X.copy()
            X[:, c] = v
            return X

        f1, auc = brain_f1_cue(
            estimator, calibrate, X_eval, lab_eval, exp_eval, subj_eval, impute
        )
        rows.append(("no retrain, mean-imputed", c, f1, auc))

    base = refit(estimator, best_params, X_train, lab_train, exp_train, seed)
    f1, auc = brain_f1_cue(base, calibrate, X_eval, lab_eval, exp_eval, subj_eval)
    rows.append(("refit, all features", "-", f1, auc))
    for c in [HEART] + list(controls):
        Xt, Xe = X_train.copy(), X_eval.copy()
        Xt[..., c] = 0.0
        Xe[..., c] = 0.0
        est = refit(estimator, best_params, Xt, lab_train, exp_train, seed)
        f1, auc = brain_f1_cue(est, calibrate, Xe, lab_eval, exp_eval, subj_eval)
        rows.append(("refit, feature dropped", c, f1, auc))
    return pd.DataFrame(rows, columns=["setting", "centroid", "brain_f1", "auc"])


@app.command()
def main(ablation: bool = False, n_controls: int = 3, seed: int = 0):
    root = Path(__file__).absolute().parents[1]
    out_dir = root / "results/feature_importance/random_forest_cue"
    train_config = parse_config_file_args(
        root / "config/_private/random_forest_5min_cmmn-None_idf.txt",
        feature_extractor="bowav",
    )
    cfg = EvalConfig(
        eval_dataset="cue", train_config=train_config, root=root, cmmn_filter=None
    )
    stem = get_base_results_filename(cfg)
    estimator, best_params = load_estimator(cfg.path_to_classifier["bowav"])
    calibrate = make_calibrate_idf_fn(cfg)

    X_eval, lab_eval, exp_eval, subj_eval = load_bowav(
        root / "data/cue/bowav/full" / f"{stem}.npz"
    )
    X_train, lab_train, exp_train, _ = load_bowav(
        root / "data/emotion_study/bowav/train" / f"{stem}.npz"
    )
    pos = pd.read_csv(out_dir / "brain_positive_shap_covariance_welch_centroid_analysis.csv")
    top = pos.centroid_index.astype(int).to_numpy()

    # Segment-level raw rates (all ICs, all segments)
    R_tgt = X_eval.reshape(-1, X_eval.shape[-1])
    brain_tgt = np.repeat(lab_eval, X_eval.shape[1]) == BRAIN
    R_src = X_train.reshape(-1, X_train.shape[-1])
    brain_src = np.repeat(lab_train, X_train.shape[1]) == BRAIN

    # 1. Source vs target rates
    print("1. Mean raw rate, Brain / non-Brain segments (ratio)")
    for c in [HEART, 95, 516]:
        sb, so = R_src[brain_src, c].mean(), R_src[~brain_src, c].mean()
        tb, to = R_tgt[brain_tgt, c].mean(), R_tgt[~brain_tgt, c].mean()
        print(f"   {c:3d} ({CODEBOOKS[c // N_CENTROIDS]:10s}) source {sb:.4f} / {so:.4f} "
              f"({sb / so:4.1f}x) | Cue {tb:.4f} / {to:.4f} ({tb / to:4.1f}x) | "
              f"Cue / source overall {R_tgt[:, c].mean() / R_src[:, c].mean():5.1f}x")

    # 2. Scaled values seen by the classifier (as in the SHAP analysis)
    _, background, explained, _, _, _ = load_data(cfg, sample_size=1000)
    pooled = calibrate(estimator, slice(None))
    B = estimator["scaler"].transform(background).toarray()
    E = pooled["scaler"].transform(explained).toarray()
    print("\n2. Mean scaled value: SHAP background (source IDF) / explained Cue "
          "segments (IDF-calibrated)")
    for c in [HEART, 95, 516]:
        print(f"   {c:3d}: {B[:, c].mean():.4f} / {E[:, c].mean():.4f} "
              f"({E[:, c].mean() / B[:, c].mean():.1f}x)")

    # 3. SHAP decomposition of the covariance for the top-10 positive centroids
    cache = np.load(out_dir / "shap_cache.npz")
    S = cache["shap_values"][:, :, BRAIN]
    y = cache["eval_labels"] == BRAIN
    X_shap = cache["eval_data"]
    rows = []
    for rank, c in enumerate(top, 1):
        s = S[:, c]
        rows.append(
            {
                "rank": rank,
                "centroid": c,
                "codebook": CODEBOOKS[c // N_CENTROIDS],
                "covariance": np.cov(s, y.astype(float))[0, 1],
                "mean_shap_brain": s[y].mean(),
                "mean_shap_other": s[~y].mean(),
                "shap_difference": s[y].mean() - s[~y].mean(),
                "frac_nonzero_shap": (np.abs(s) > 1e-4).mean(),
                "spearman_rate_shap": spearmanr(X_shap[:, c], s).correlation,
                "rate_ratio_cue": R_tgt[brain_tgt, c].mean() / R_tgt[~brain_tgt, c].mean(),
                "rate_ratio_source": R_src[brain_src, c].mean() / R_src[~brain_src, c].mean(),
            }
        )
    df = pd.DataFrame(rows)
    print(f"\n3. SHAP decomposition (covariance ~ p(1-p) x difference, "
          f"p(1-p) = {y.mean() * (1 - y.mean()):.3f})")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df.to_csv(out_dir / "heart_centroid_rank_analysis.csv", index=False)

    # 4. Correlation with the top Brain-codebook centroids
    top_brain = [c for c in top if c // N_CENTROIDS == 0]
    s_brain = R_tgt[:, top_brain].sum(1)
    r, _ = pearsonr(R_tgt[:, HEART], s_brain)
    print(f"\n4. Pearson correlation of the rate of {HEART} with the summed rate of "
          f"{top_brain} on Cue: {r:+.2f}")

    if ablation:
        res = run_ablation(
            estimator, best_params, calibrate, (X_train, lab_train, exp_train),
            (X_eval, lab_eval, exp_eval, subj_eval), R_tgt, brain_tgt, top, seed,
            n_controls,
        )
        print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        res.to_csv(out_dir / "heart_centroid_ablation.csv", index=False)

    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    app()
