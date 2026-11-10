"""Within-dataset sanity check of the cross-dataset feature reweighting.

The variance-matching sqrt(rho_t) reweighting stored in the IDF slot is designed
to correct covariate shift when transferring the source-trained BoWav classifier
to a *different* target dataset. As a sanity check, we apply the same reweighting
strategy to the held-out TEST subjects of the SAME dataset used for training
(Imagined Emotion Study). With no cross-dataset shift, the reweighting should not
fundamentally change performance.

Each figure (one per classifier / validation segment length) compares the
"Original" (uncorrected source IDF, *_idf.csv) and "Corrected" (variance-matching
reweighting, *_scaled_idf.csv) brain F1 vs prediction window, on a log-x axis
with the same ticks as icwaves/viz.py and a fixed 0.6-1.0 y-range.

Two output flavors are produced:
  * Publication PDFs (300 dpi, single-column size, +/- std band, no in-figure
    title) written into the paper's img/ directory. Used in the manuscript.
  * Exploratory PNGs (150 dpi, larger, +/- SEM band, with a title) written next
    to the CSVs. Handy for quick on-screen comparison.

Run with `--png-only` or `--pdf-only` to emit just one flavor (default: both).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WIN = "Prediction window [minutes]"
SUBJ = "Subject ID"
F1 = "Brain F1 score"

# Standard x ticks for the log scale (mirrors icwaves/viz.py)
GLOBAL_X_TICKS = np.array([0.15, 0.5, 1, 2, 3, 5, 10, 30, 50])

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
DATA_DIR = os.path.join(REPO_ROOT, "results", "emotion_study", "evaluation")
IMG_DIR = os.path.join(REPO_ROOT, "paper", "icwaves_TBME", "img")

# (file stem, classifier label, l_val label, publication output stem). The
# "Original" (uncorrected) data lives in "<stem>_idf.csv" and the "Corrected"
# (variance-matching reweighting) data in "<stem>_scaled_idf.csv".
FILES = [
    (
        "logistic_bowav_valSegLen300_cmmn-None",
        "Logistic Regression",
        "5",
        "idf_sanity_emotion_logistic_5min",
    ),
    (
        "logistic_bowav_valSegLenNone_cmmn-None",
        "Logistic Regression",
        "50",
        "idf_sanity_emotion_logistic_50min",
    ),
    (
        "random_forest_bowav_valSegLen300_cmmn-None",
        "Random Forest",
        "5",
        "idf_sanity_emotion_random_forest_5min",
    ),
    (
        "random_forest_bowav_valSegLenNone_cmmn-None",
        "Random Forest",
        "50",
        "idf_sanity_emotion_random_forest_50min",
    ),
]


def _summary(path):
    """Per-window mean and spread (std and SEM) of brain F1 over subjects.

    Drops the duplicated first-window block present in the raw CSVs.
    """
    df = pd.read_csv(path).drop_duplicates(subset=[WIN, SUBJ])
    g = df.groupby(WIN)[F1]
    return g.mean(), g.std(), g.sem()


def _plot(ax, orig_mean, corr_mean, orig_err, corr_err):
    for mean, err, label, color in [
        (orig_mean, orig_err, "Original", "tab:blue"),
        (corr_mean, corr_err, "Corrected", "tab:orange"),
    ]:
        ax.plot(mean.index, mean.values, "o-", color=color, label=label)
        ax.fill_between(mean.index, mean - err, mean + err, color=color, alpha=0.2)

    ax.set_xscale("log")
    ax.set_xticks(GLOBAL_X_TICKS, labels=GLOBAL_X_TICKS)
    # Fixed range so every panel (5 and 50 min) shows identical ticks.
    ax.set_xlim(GLOBAL_X_TICKS[0], GLOBAL_X_TICKS[-1])
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel(WIN)
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--pdf-only",
        action="store_true",
        help="only write the publication PDFs into the paper img/ directory",
    )
    group.add_argument(
        "--png-only",
        action="store_true",
        help="only write the exploratory PNGs next to the CSVs",
    )
    args = parser.parse_args()
    make_pdf = not args.png_only
    make_png = not args.pdf_only

    for file_stem, clf, l_val, out_stem in FILES:
        orig_csv = os.path.join(DATA_DIR, file_stem + "_idf.csv")
        corr_csv = os.path.join(DATA_DIR, file_stem + "_scaled_idf.csv")
        orig_mean, orig_std, orig_sem = _summary(orig_csv)
        corr_mean, corr_std, corr_sem = _summary(corr_csv)

        if make_pdf:
            # Publication figure: single-column size, +/- std band, no title.
            fig, ax = plt.subplots(figsize=(6, 4))
            _plot(ax, orig_mean, corr_mean, orig_std, corr_std)
            ax.set_ylabel("Brain F1 score")
            fig.tight_layout()
            out = os.path.join(IMG_DIR, out_stem + ".pdf")
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}")

        if make_png:
            # Exploratory figure: larger, +/- SEM band, with a title.
            fig, ax = plt.subplots(figsize=(8, 5))
            _plot(ax, orig_mean, corr_mean, orig_sem, corr_sem)
            ax.set_ylabel("Brain F1 score (mean over subjects)")
            ax.set_title(f"{clf} — validation segment length: {l_val} min")
            fig.tight_layout()
            out = os.path.join(DATA_DIR, file_stem + "_idf_comparison.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
