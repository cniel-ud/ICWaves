from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Sequence
from sklearn.model_selection import ParameterGrid
from icwaves.data.types import DataBundle
from sklearn.model_selection._split import BaseCrossValidator


@dataclass
class JobParameters:
    """Container for job parameters and indices."""

    parameters: Dict[str, Any]
    train_indices: Sequence[int]
    test_indices: Sequence[int]
    candidate_index: int
    split_index: int
    n_splits: int
    n_candidates: int


def get_job_parameters(
    job_id: int, data_bundle: DataBundle, cv: BaseCrossValidator, param_grid: dict
):
    """Get parameters and split for a specific job ID.

    Args:
        job_id: Index of the job to run
        data_bundle: Bundle containing data and metadata
        cv: Cross-validation splitter
        param_grid: Grid of parameters to search

    Returns:
        Tuple of (parameters, train_indices, test_indices, candidate_index, split_index, n_splits, n_candidates)
    """
    # Get all combinations of parameters and CV splits
    # data_bundle.data is used only to get the number of samples (first dimension)
    all_combinations = list(
        product(
            enumerate(param_grid),
            enumerate(
                cv.split(data_bundle.data, data_bundle.labels, data_bundle.subj_ind)
            ),
        )
    )

    # Get specific combination for this job
    (cand_idx, parameters), (split_idx, (train, test)) = all_combinations[job_id]

    n_splits = cv.get_n_splits(X=None, y=None, groups=data_bundle.subj_ind)
    n_candidates = len(param_grid)

    return JobParameters(
        parameters=parameters,
        train_indices=train,
        test_indices=test,
        candidate_index=cand_idx,
        split_index=split_idx,
        n_splits=n_splits,
        n_candidates=n_candidates,
    )
