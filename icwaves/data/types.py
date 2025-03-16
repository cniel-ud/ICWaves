# icwaves/data/types.py
from dataclasses import dataclass
from typing import Any, Optional


# TODO: improve type annotation
@dataclass
class DataBundle:
    """Container for data and associated metadata."""

    data: Any  # The main data (either centroid_assignments or ics)
    labels: Any
    expert_label_mask: Any
    subj_ind: Any
    noisy_labels: Optional[Any] = None
    n_centroids: Optional[int] = None
    srate: Optional[float] = None
