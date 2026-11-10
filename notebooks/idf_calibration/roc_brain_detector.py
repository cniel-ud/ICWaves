"""
Brain / not-brain ROC + AUC analysis of the IDF calibration.

The multi-class BoWav classifier is recast as a BINARY brain-vs-not-brain
detector. The detector score is the aggregated brain-class probability p_brain
that the deployed 7-class argmax decision is built from
(compute_brain_F1_score_per_subject): for majority_vote classifiers it is the
segment-pooled, renormalized posterior (log-probabilities summed over segments,
then softmax) reduced to its brain component; for count_pooling classifiers it is
the posterior of the pooled prediction window. The prediction window is 5 min
(cue) / 10 min (epic) for count_pooling and the whole IC (50 min cue / 10 min
epic) for majority_vote; EPIC ICs are 10 min long. The binary label is y=1 iff
the expert label is Brain (class 0). AUC and the brain-F1 numbers therefore
describe the same probabilities, threshold-free vs at the argmax cutoff.

Two IDF schemes are compared, for every (classifier, val-seg, dataset):
    Original   -- source-trained TF-IDF scaler, no per-subject calibration.
    Corrected  -- variance-matching reweighting of the IDF by sqrt(rho_t), with
                  rho_t = mean_source_t / mean_target_t (see make_corrected_idf_fn);
                  fit from unlabeled target BoWav only.
Labels are used only for scoring.

Outputs (results/idf_calibration/roc/)
  * roc_<classifier>_<5min|50min>.{png,pdf}: ROC overlays, cue and EPIC side by
    side, AUC in the legend.
  * auc_summary.csv: AUC of each scheme with a bootstrap standard error (ICs
    resampled with replacement), and the paired difference
    Delta = AUC_corrected - AUC_original with its paired-bootstrap standard error
    (same resampled ICs for both schemes).
  * roc_scores.npz: cache of the per-IC scores so figures/tables can be
    regenerated without re-running the (slow) evaluation.

Run:
    .venv/bin/python notebooks/idf_calibration/roc_brain_detector.py [cue|epic|both]
    .venv/bin/python notebooks/idf_calibration/roc_brain_detector.py [cue|epic|both] --replot
"""

# %%
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy.special
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

from icwaves.data.loading import get_feature_extractor, load_data_bundles
from icwaves.evaluation.config import EvalConfig
from icwaves.evaluation.evaluation import load_estimator
from icwaves.evaluation.utils import (
    build_features_based_on_aggregation_method,
    get_base_results_filename,
)
from icwaves.feature_extractors.tfidf_rate_scaler import TfidfRateScaler
from icwaves.feature_extractors.utils import convert_segment_length
from icwaves.file_utils import parse_config_file_args

BRAIN = 0
ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "results/idf_calibration/roc"
SCORES_CACHE = OUTDIR / "roc_scores.npz"

# (classifier, config-stem, val-seg length in seconds; -1 = whole IC)
CLASSIFIERS = [
    ("random_forest", "random_forest_5min_cmmn-None_idf", 300),
    ("random_forest", "random_forest_50min_cmmn-None_idf", -1),
    ("logistic", "logistic_5min_cmmn-None_idf", 300),
    ("logistic", "logistic_50min_cmmn-None_idf", -1),
]

SCHEMES = ("original", "corrected")
STYLE = {
    "original": dict(label="Original", color="tab:blue"),
    "corrected": dict(label="Corrected", color="tab:orange"),
}
# Scheme names used by older versions of the score cache.
LEGACY_SCHEME_KEYS = {"source IDF": "original", "shrink0.5": "corrected"}

N_BOOT = 2000
BOOT_SEED = 0


def vtag_of(valseg):
    return "5min" if valseg == 300 else "50min"


# ---------------------------------------------------------------------------
# Setup: config, data, estimator, aggregation, prediction window
# ---------------------------------------------------------------------------
def build(classifier, config_stem, eval_dataset):
    """Load config, data bundles and estimator for one (classifier, dataset)."""
    minutes_per_ic = 10 if eval_dataset == "epic" else None
    train_config = parse_config_file_args(
        ROOT / "config/_private" / f"{config_stem}.txt", "bowav"
    )
    config = EvalConfig(
        eval_dataset=eval_dataset, train_config=train_config, root=ROOT,
        cmmn_filter=None, minutes_per_ic=minutes_per_ic,
    )
    data_bundles = load_data_bundles(config)
    feature_extractor = {"bowav": get_feature_extractor("bowav", data_bundles)}
    clf, best_params = load_estimator(config.path_to_classifier["bowav"])
    agg = {"bowav": best_params["input_or_output_aggregation_method"]}
    train_seglen = best_params["training_segment_length"]

    # Prediction window (seconds): count_pooling -> 5-min (cue) / 10-min (epic);
    # majority_vote -> whole IC (cue trains at 50 min, EPIC ICs are 10 min).
    mins = minutes_per_ic if minutes_per_ic is not None else int(config.minutes_per_ic)
    if agg["bowav"] == "count_pooling":
        pred_seconds = 600.0 if eval_dataset == "epic" else 300.0
    else:
        pred_seconds = mins * 60.0
    conv = convert_segment_length(
        [pred_seconds], config.feature_extractor,
        data_bundles["bowav"].srate, config.window_length,
    )[0]

    X = {k: v.data for k, v in data_bundles.items()}
    db = data_bundles["bowav"]
    if getattr(db, "zero_window_mask", None) is not None:
        X["zero_window_mask"] = db.zero_window_mask
    return config, X, db, feature_extractor, {"bowav": clf}, agg, train_seglen, conv, pred_seconds


# ---------------------------------------------------------------------------
# Corrected scheme: variance-matching IDF reweighting
# ---------------------------------------------------------------------------
def make_corrected_idf_fn(config):
    """Per-subject, unsupervised IDF correction (the 'Corrected' scheme).

    For each centroid t, rho_t = mean_source_t / mean_target_t, where mean_source
    is the mean BoWav rate over the source training set and mean_target the mean
    over the subject's ICs of the target dataset (rho_t = 1 where the target
    mean is 0). Since Var(x_t) scales with the rate for sparse counts, scaling
    feature t by w_t = sqrt(rho_t) scales its variance by rho_t, so the target
    variances mimic the source's. w_t is folded into the scaler's IDF:
    idf_t <- idf_t * sqrt(rho_t). Labels are never used.

    Numerically equivalent to icwaves.evaluation.utils.make_calibrate_idf_fn
    (the paper pipeline); see the notes in the hand-off report for details.
    """

    def calibrate_idf(pipeline, subj_mask):
        assert isinstance(pipeline["scaler"], TfidfRateScaler)
        # The source features were built with the training CMMN filter.
        config_train = deepcopy(config)
        config_train.cmmn_filter = config_train.train_config.cmmn_filter
        base = get_base_results_filename(config_train)

        train_path = config_train.root / "data/emotion_study/bowav/train" / f"{base}.npz"
        with np.load(train_path, allow_pickle=True) as f:
            bowav_train = f["bowav"].reshape(-1, f["bowav"].shape[-1])
        test_path = config.root / f"data/{config.eval_dataset}/bowav/full" / f"{base}.npz"
        with np.load(test_path, allow_pickle=True) as f:
            bowav_full_test = f["bowav"]

        mean_source = bowav_train.mean(axis=0)
        mean_target = bowav_full_test[subj_mask].mean(axis=(0, 1))
        rho = np.divide(
            mean_source, mean_target,
            out=np.ones_like(mean_source), where=mean_target != 0,
        )
        idf_corrected = pipeline["scaler"].idf_ * np.sqrt(np.clip(rho, 1e-9, None))

        corrected = deepcopy(pipeline)
        corrected["scaler"].idf_ = idf_corrected
        corrected["scaler"]._tfidf_transformer.idf_ = idf_corrected
        return corrected

    return calibrate_idf


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def brain_scores(built, calibrate_idf):
    """Per expert-labeled IC: aggregated p_brain (deployed path) and binary
    y (1=Brain). Reproduces compute_brain_F1_score_per_subject's aggregation.
    calibrate_idf=None uses the source IDF unchanged (Original)."""
    (config, X, db, feat_ext, clf_dict, agg, train_seglen, conv, _) = built
    PB, Y = [], []
    for sid in config.subj_ids:
        sm = db.subj_ind == sid
        fd = build_features_based_on_aggregation_method(
            feat_ext, X, conv, train_seglen, agg, sm)
        for ft, fa in fd.items():
            n_ts, n_seg, _ = fa.shape
            fa2 = np.vstack(fa)
            model = clf_dict[ft] if calibrate_idf is None else calibrate_idf(clf_dict[ft], sm)
            bi = list(model.classes_).index(BRAIN)
            proba = model.predict_proba(fa2)
            if agg[ft] == "majority_vote":
                lp = np.log(proba + 1e-12).reshape(n_ts, n_seg, -1).sum(1)
                proba = np.exp(lp - scipy.special.logsumexp(lp, axis=1, keepdims=True))
        mask = db.expert_label_mask[sm]
        PB.append(proba[mask, bi])
        Y.append((db.labels[sm][mask] == BRAIN).astype(int))
    return np.concatenate(PB), np.concatenate(Y)


def _key(classifier, vtag, ds, name):
    return f"{classifier}|{vtag}|{ds}|{name}"


def compute_scores(datasets):
    """Run the evaluation pipeline; return {key: (p, y)} for every
    (classifier, valseg, dataset, scheme)."""
    scored = {}
    for classifier, stem, valseg in CLASSIFIERS:
        vtag = vtag_of(valseg)
        for ds in datasets:
            built = build(classifier, stem, ds)
            fns = {"original": None, "corrected": make_corrected_idf_fn(built[0])}
            for name in SCHEMES:
                scored[_key(classifier, vtag, ds, name)] = brain_scores(built, fns[name])
            print(f"[{classifier} {vtag} {ds}] scored {', '.join(SCHEMES)}")
    return scored


def save_scores(scored):
    flat = {}
    for k, (p, y) in scored.items():
        flat[f"{k}||p"] = np.asarray(p)
        flat[f"{k}||y"] = np.asarray(y)
    np.savez(SCORES_CACHE, **flat)
    print(f"wrote score cache {SCORES_CACHE}")


def load_scores():
    """Reload the cache. Old scheme names are mapped to the current ones and
    entries of any other (no longer supported) scheme are ignored."""
    scored = {}
    with np.load(SCORES_CACHE, allow_pickle=False) as f:
        for k in {k.rsplit("||", 1)[0] for k in f.files}:
            classifier, vtag, ds, name = k.split("|")
            name = LEGACY_SCHEME_KEYS.get(name, name)
            if name in SCHEMES:
                scored[(classifier, vtag, ds, name)] = (f[f"{k}||p"], f[f"{k}||y"])
    return scored


# ---------------------------------------------------------------------------
# Bootstrap standard errors
# ---------------------------------------------------------------------------
def bootstrap_auc(p_orig, p_corr, y, n_boot=N_BOOT, seed=BOOT_SEED):
    """Resample ICs with replacement (the same resample for both schemes) and
    return (se_original, se_corrected, se_delta), where delta = AUC_corrected -
    AUC_original and se_delta is its paired-bootstrap standard error."""
    rng = np.random.default_rng(seed)
    n = len(y)
    a_o, a_c = [], []
    while len(a_o) < n_boot:
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        a_o.append(roc_auc_score(y[idx], p_orig[idx]))
        a_c.append(roc_auc_score(y[idx], p_corr[idx]))
    a_o, a_c = np.array(a_o), np.array(a_c)
    return a_o.std(ddof=1), a_c.std(ddof=1), (a_c - a_o).std(ddof=1)


# ---------------------------------------------------------------------------
def run(datasets, replot=False):
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if replot:
        if not SCORES_CACHE.exists():
            raise FileNotFoundError(
                f"--replot needs a cached {SCORES_CACHE}; run without --replot first.")
        print(f"reloading cached scores from {SCORES_CACHE} (no recompute)")
        scored = load_scores()
        # Use the datasets actually cached, in canonical cue-before-epic order.
        cached_ds = {k[2] for k in scored}
        datasets = [d for d in ("cue", "epic") if d in cached_ds]
    else:
        scored = {tuple(k.split("|")): v for k, v in compute_scores(datasets).items()}
        save_scores({_key(*k): v for k, v in scored.items()})

    # ---- AUC table with bootstrap standard errors ----
    rows = []
    for classifier, stem, valseg in CLASSIFIERS:
        vtag = vtag_of(valseg)
        for ds in datasets:
            p_o, y = scored[(classifier, vtag, ds, "original")]
            p_c, y_c = scored[(classifier, vtag, ds, "corrected")]
            assert np.array_equal(y, y_c)
            auc_o, auc_c = roc_auc_score(y, p_o), roc_auc_score(y, p_c)
            se_o, se_c, se_d = bootstrap_auc(p_o, p_c, y)
            rows.append((classifier, vtag, ds, auc_o, se_o, auc_c, se_c,
                         auc_c - auc_o, se_d, len(y), float(y.mean())))

    print("\n" + "=" * 100)
    print("AUC  (brain vs not-brain, detector score = aggregated p_brain; SE = bootstrap)")
    print("=" * 100)
    print(f"{'classifier':14s} {'valseg':6s} {'dataset':7s} {'original':>15s} "
          f"{'corrected':>15s} {'delta':>15s} {'n':>5s} {'prev':>5s}")
    for r in rows:
        print(f"{r[0]:14s} {r[1]:6s} {r[2]:7s} {r[3]:>8.3f}+-{r[4]:.3f} "
              f"{r[5]:>8.3f}+-{r[6]:.3f} {r[7]:>+8.3f}+-{r[8]:.3f} {r[9]:>5d} {r[10]:>5.2f}")

    csv = OUTDIR / "auc_summary.csv"
    with csv.open("w") as f:
        f.write("classifier,valseg,dataset,auc_original,se_original,auc_corrected,"
                "se_corrected,delta,se_delta,n,prevalence\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.4f},"
                    f"{r[7]:.4f},{r[8]:.4f},{r[9]},{r[10]:.4f}\n")
    print(f"\nwrote {csv}")

    # ---- ROC overlays: one figure per (classifier, valseg), datasets as columns ----
    # Font size matched to the F1-curve figures (icwaves/viz.py: font_size=14) so
    # all paper figures render at a consistent apparent size at \linewidth.
    FONT_SIZE = 14
    for classifier, stem, valseg in CLASSIFIERS:
        vtag = vtag_of(valseg)
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.6), squeeze=False)
        for j, ds in enumerate(datasets):
            ax = axes[0][j]
            for name in SCHEMES:
                p, y = scored[(classifier, vtag, ds, name)]
                if len(np.unique(y)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y, p)
                auc = roc_auc_score(y, p)
                ax.plot(fpr, tpr, lw=1.8, color=STYLE[name]["color"],
                        label=f"{STYLE[name]['label']} (AUC={auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
            ax.set_xlabel("FPR (non-brain called brain)", fontsize=FONT_SIZE)
            ax.set_ylabel("TPR (brain recall)", fontsize=FONT_SIZE)
            ax.set_title(f"{ds}", fontsize=FONT_SIZE)
            ax.tick_params(axis="both", labelsize=FONT_SIZE)
            ax.legend(loc="lower right", fontsize=FONT_SIZE)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        fig.tight_layout()
        out = OUTDIR / f"roc_{classifier}_{vtag}.png"
        fig.savefig(out, dpi=140)
        fig.savefig(out.with_suffix(".pdf"))
        plt.close(fig)
        print(f"  wrote {out}")


# %%
if __name__ == "__main__":
    # Usage:
    #   roc_brain_detector.py [cue|epic|both]            compute scores + plot
    #   roc_brain_detector.py [cue|epic|both] --replot   reload cached scores + plot only
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    replot = "--replot" in sys.argv[1:]
    which = args[0] if args else "both"
    datasets = ["cue", "epic"] if which == "both" else [which]
    run(datasets, replot=replot)
