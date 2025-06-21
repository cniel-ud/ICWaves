"""
Simplified TF-IDF scaler for count rates using TfidfTransformer.transform() directly.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import StrOptions

# Constants from sklearn
FLOAT_DTYPES = (np.float64, np.float32)


class TfidfRateScaler(BaseEstimator, TransformerMixin):
    """
    TF-IDF scaler for count rates.
    
    This scaler addresses the issue where TfidfTransformer expects integer counts
    but we have count rates (floats) due to normalization by the number of documents.
    
    The scaler:
    1. For fitting: Binarizes the count rates and converts to integers to compute IDF
    2. For transformation: Uses the actual rates with TfidfTransformer.transform()
    
    This ensures that the classifier receives IDF-scaled rates for both training
    and inference, maintaining consistency while handling the document count disparity
    between training and validation/test segments.
    
    Parameters
    ----------
    norm : {'l1', 'l2'}, default='l2'
        Norm used to normalize term vectors. 'l1' uses the Manhattan distance,
        'l2' uses the Euclidean distance.
    use_idf : bool, default=True
        Enable inverse-document-frequency reweighting. If False, idf(t) = 1.
    smooth_idf : bool, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.
    sublinear_tf : bool, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
    
    Attributes
    ----------
    idf_ : array of shape (n_features,)
        The inverse document frequency (IDF) vector; only defined
        if ``use_idf`` is True.
    """
    
    _parameter_constraints: dict = {
        "norm": [StrOptions({"l1", "l2"}), None],
        "use_idf": ["boolean"],
        "smooth_idf": ["boolean"],
        "sublinear_tf": ["boolean"],
    }
    
    def __init__(self, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        
        # Create internal TfidfTransformer once with our exact parameters
        self._tfidf_transformer = TfidfTransformer(
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf
        )
    
    def set_params(self, **params):
        """Set parameters and update internal transformer parameters."""
        super().set_params(**params)
        
        # Update the internal transformer's parameters
        self._tfidf_transformer.set_params(
            norm=self.norm,
            use_idf=self.use_idf,
            smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf
        )
        return self
        
    def fit(self, X, y=None):
        """
        Learn the IDF vector from binarized count rates.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Count rates matrix (floats).
        y : Ignored
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        self : object
            Fitted transformer.
        """
        # Validate input
        X = self._validate_data(
            X, accept_sparse=('csr', 'csc'), dtype=FLOAT_DTYPES, reset=True
        )
        
        # Convert to sparse matrix if needed
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        
        # Binarize the count rates for IDF computation
        X_binary = X.copy()
        X_binary.data = (X_binary.data > 0).astype(np.int32)
        X_binary.eliminate_zeros()
        
        # Fit the internal transformer on binarized data
        self._tfidf_transformer.fit(X_binary)
        
        # Expose IDF vector for compatibility
        if self.use_idf:
            self.idf_ = self._tfidf_transformer.idf_
        
        return self
    
    def transform(self, X, copy=True):
        """
        Transform count rates to TF-IDF representation using actual rates.
        
        This method directly uses TfidfTransformer.transform() on the count rates,
        which handles all validation, TF-IDF logic, sublinear scaling, and normalization.
        
        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            Count rates matrix (floats).
        copy : bool, default=True
            Whether to copy X and operate on the copy or perform in-place operations.
            
        Returns
        -------
        X_transformed : sparse matrix of shape (n_samples, n_features)
            TF-IDF-weighted document-term matrix.
        """
        # TfidfTransformer.transform() handles all validation, sparse conversion,
        # fitted check, sublinear TF scaling, IDF scaling, and normalization
        return self._tfidf_transformer.transform(X, copy=copy)
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        check_is_fitted(self, '_tfidf_transformer')
        # Delegate to the internal transformer which handles feature names correctly
        return self._tfidf_transformer.get_feature_names_out(input_features)
