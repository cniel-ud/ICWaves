from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import f1_score
from numpy.typing import NDArray
import numpy.typing as npt


def calculate_iclabel_f1_scores(
    dataset_dir: Path,
    subj_ids: list[int],
    validation_segment_lengths: npt.NDArray[np.int_],
):
    """
    Calculate ICLabel F1 scores for different segment lengths and subjects.

    Parameters
    ----------
    dataset_dir : Path
        Directory containing the dataset.
    subj_ids : list[int]
        List of subject IDs to process.
    validation_segment_lengths : npt.NDArray[np.int_]
        Array of validation segment lengths in seconds.

    Returns
    -------
    mean_and_std_df : pd.DataFrame
        Data frame with ICLabel F1 scores, including means and standard deviations.
    """
    columns = [
        "Prediction window [minutes]",
        "Subject ID",
        "Brain F1 score",
        "Number of ICs",
    ]
    df = pd.DataFrame(columns=columns)

    for test_segment_len in validation_segment_lengths:
        subdir = dataset_dir.joinpath(f"IC_labels_at_{test_segment_len:.1f}_seconds")
        for subj_id in subj_ids:
            file = subdir.joinpath(f"subj-{subj_id:02}.mat")
            with file.open("rb") as f:
                data = loadmat(f)
                expert_label_mask = data["expert_label_mask"].flatten().astype(bool)
                labels = data["labels"] - 1
                noisy_labels = data["noisy_labels"]

            ICLabel_labels = np.argmax(noisy_labels, axis=1)
            y_expert = labels[expert_label_mask]
            y_pred_expert = ICLabel_labels[expert_label_mask]
            brain_f1_score = f1_score(y_expert, y_pred_expert, labels=[0], average=None)

            df.loc[len(df)] = {
                "Prediction window [minutes]": test_segment_len / 60,
                "Subject ID": subj_id,
                "Brain F1 score": brain_f1_score[0],
                "Number of ICs": expert_label_mask.sum(),
            }
    std_df = df.groupby("Prediction window [minutes]")["Brain F1 score"].std()
    std_df = std_df.rename("StdDev - iclabel").reset_index()
    mean_df = (
        df.groupby("Prediction window [minutes]")["Brain F1 score"]
        .mean()
        .rename("Brain F1 score - iclabel")
        .reset_index()
    )
    mean_and_std_df = pd.merge(std_df, mean_df, on="Prediction window [minutes]")

    return mean_and_std_df
