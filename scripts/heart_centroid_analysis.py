#!/usr/bin/env python3
"""
Is Heart-codebook centroid 507 acting as a generic "basis function"?

Centroid 507 (Heart codebook) is among the top-10 positive Brain-class SHAP
centroids of the 5-min random forest on Cue. This script tests whether it is a
generic waveform that absorbs Brain windows in the Heart codebook, rather than a
Brain-specific one:

1. Correlation of its rate with the rates of the top Brain-codebook centroids.
2. Presence and mean rate in Brain vs non-Brain segments (vs. Brain centroid 95).
3. Within-codebook share: fraction of the Heart-codebook windows of each segment
   assigned to 507, and how often it is the most used Heart centroid.
4. Ablation, on the Brain F1 of Cue (first 5 minutes of each IC, expert-labeled
   ICs, mean over subjects, per-subject IDF calibration, as in the paper):
   (a) no retrain: set feature 507 to its training mean after scaling;
   (b) refit with the saved best hyperparameters and feature 507 zeroed in train
       and test data (equivalent to removing it);
   both compared with control centroids of similar Brain-segment prevalence, and
   with a refit of the unmodified features using the same random seed.

Uses the cached BoWav files; the SHAP ranking comes from
results/feature_importance/random_forest_cue/.

Usage:
    python scripts/heart_centroid_analysis.py [--n-controls 3]
"""

from pathlib import Path

import numpy as np
import pandas as pd
import typer
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score

from icwaves.evaluation.config import EvalConfig
from icwaves.evaluation.evaluation import load_estimator
from icwaves.evaluation.utils import get_base_results_filename, make_calibrate_idf_fn
from icwaves.file_utils import parse_config_file_args

app = typer.Typer()

HEART = 507
BRAIN_REF = 95
N_CENTROIDS = 128
BRAIN = 0


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


@app.command()
def main(n_controls: int = 3, seed: int = 0):
    root = Path(__file__).absolute().parents[1]
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
    pos = pd.read_csv(
        root
        / "results/feature_importance/random_forest_cue"
        / "brain_positive_shap_covariance_welch_centroid_analysis.csv"
    )
    top_brain = [int(c) for c in pos.centroid_index if c // N_CENTROIDS == 0]

    # Segment-level rates (all ICs, all segments), as in the SHAP analysis
    R = X_eval.reshape(-1, X_eval.shape[-1])
    seg_lab = np.repeat(lab_eval, X_eval.shape[1])
    is_brain = seg_lab == BRAIN

    # 1. Correlation with the top Brain-codebook centroids
    print(f"\n1. Correlation of the rate of {HEART} with the summed rate of the "
          f"top Brain-codebook centroids {top_brain}")
    s_brain = R[:, top_brain].sum(1)
    for name, m in (("all segments", slice(None)), ("Brain segments", is_brain)):
        r, _ = pearsonr(R[m, HEART], s_brain[m])
        rho, _ = spearmanr(R[m, HEART], s_brain[m])
        print(f"   {name:15s} Pearson {r:+.2f}  Spearman {rho:+.2f}")

    # 2. Presence and mean rate
    print("\n2. Presence (rate > 0) and mean rate")
    for c in (HEART, BRAIN_REF):
        pb, pn = (R[is_brain, c] > 0).mean(), (R[~is_brain, c] > 0).mean()
        mb, mn = R[is_brain, c].mean(), R[~is_brain, c].mean()
        print(f"   centroid {c:3d}: present in {pb:5.1%} of Brain / {pn:5.1%} of "
              f"non-Brain segments; rate {mb:.4f} / {mn:.4f} (ratio {mb / mn:.2f})")

    # 3. Within-codebook share
    j = HEART // N_CENTROIDS
    cb = R[:, j * N_CENTROIDS : (j + 1) * N_CENTROIDS]
    share = R[:, HEART] / np.maximum(cb.sum(1), 1e-12)
    top = cb.argmax(1) == HEART % N_CENTROIDS
    print(f"\n3. Share of the Heart-codebook windows assigned to {HEART}")
    for name, m in (("Brain", is_brain), ("non-Brain", ~is_brain)):
        print(f"   {name:9s} median {np.median(share[m]):.1%}, mean {share[m].mean():.1%}; "
              f"most used Heart centroid in {top[m].mean():.1%} of segments")

    # 4. Ablation
    rng = np.random.default_rng(seed)
    presence = (R[is_brain] > 0).mean(0)
    near = np.where(
        (np.abs(presence - presence[HEART]) < 0.03)
        & (np.arange(R.shape[1]) != HEART)
        & ~np.isin(np.arange(R.shape[1]), pos.centroid_index.astype(int))
    )[0]
    controls = sorted(rng.choice(near, size=min(n_controls, len(near)), replace=False))
    print(f"\n4. Ablation (controls with Brain-segment presence within 3% of "
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

    df = pd.DataFrame(rows, columns=["setting", "centroid", "brain_f1", "auc"])
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out = root / "results/feature_importance/random_forest_cue/heart_centroid_ablation.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    app()
