
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.validation import _num_samples, indexable


class LeaveOneSubjectOutExpertOnly(LeaveOneGroupOut):
    """
    Leave one subject out, but test only on expert-annotated ICs
    """

    def __init__(self, expert_label_mask) -> None:
        super().__init__()

        self.expert_label_mask = expert_label_mask

    def split(self, X, y=None, groups=None):
        """
        Leave one subject out, but test only on ICs with expert labels:
        * Train set has all ICs from the other subjects
        * Test set has only the ICs with expert labels from the subject that is
        left out
        """
        X, y, groups = indexable(X, y, groups)
        indices = np.arange(_num_samples(X))
        for group_mask in self._iter_test_masks(X, y, groups):
            group_expert_mask = np.logical_and(
                group_mask, self.expert_label_mask)
            train_index = indices[np.logical_not(group_mask)]
            test_index = indices[group_expert_mask]
            yield train_index, test_index
