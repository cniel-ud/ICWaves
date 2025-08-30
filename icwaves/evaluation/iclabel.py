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

    return mean_and_std_df, df


def compute_iclabel_scores_for_dataset(
    eval_dataset: str,
    validation_times: np.ndarray,
    root: Path,
) -> pd.DataFrame:
    """
    Compute ICLabel F1 scores for a specific dataset.

    Args:
        eval_dataset: Dataset name ("emotion_study" or "cue")
        validation_times: Array of validation times in seconds
        root: Root path

    Returns:
        DataFrame: ICLabel results with columns matching the main results format
    """
    # Get ICLabel data directory
    iclabel_data_dir = root / f"data/{eval_dataset}/ICLabels"

    # Define subject IDs based on dataset
    subj_ids = (
        list(range(1, 8)) if eval_dataset == "emotion_study" else list(range(1, 13))
    )

    # Calculate ICLabel F1 scores
    mean_std_f1, per_subject_f1 = calculate_iclabel_f1_scores(
        iclabel_data_dir, subj_ids, validation_times
    )

    # Convert to match the format of all_results
    iclabel_results = []
    for _, row in mean_std_f1.iterrows():
        prediction_window = row["Prediction window [minutes]"]
        mean_f1 = row["Brain F1 score - iclabel"]
        std_f1 = row["StdDev - iclabel"]

        iclabel_results.append(
            {
                "eval_dataset": eval_dataset,
                "cmmn_filter": "None",  # Special marker for ICLabel
                "feature_extractor": "iclabel",
                "classifier_type": "iclabel",
                "is_normalized": False,  # Not applicable for ICLabel
                "validation_segment_len": -1,  # Not applicable for ICLabel
                "prediction_window": prediction_window,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
            }
        )

    return pd.DataFrame(iclabel_results), per_subject_f1
