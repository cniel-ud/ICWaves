"""
Unit tests for TfidfRateScaler using pytest.

This module contains comprehensive tests for the TfidfRateScaler class,
ensuring it works correctly in isolation and in sklearn pipelines.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from icwaves.feature_extractors.tfidf_rate_scaler import TfidfRateScaler


class TestTfidfRateScalerInit:
    """Test the __init__ method."""
    
    def test_default_parameters(self):
        """Test default parameter initialization."""
        scaler = TfidfRateScaler()
        assert scaler.norm == 'l2'
        assert scaler.use_idf == True
        assert scaler.smooth_idf == True
        assert scaler.sublinear_tf == False
        assert scaler._tfidf_transformer is not None
    
    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        scaler = TfidfRateScaler(
            norm='l1', 
            use_idf=False, 
            smooth_idf=False, 
            sublinear_tf=True
        )
        assert scaler.norm == 'l1'
        assert scaler.use_idf == False
        assert scaler.smooth_idf == False
        assert scaler.sublinear_tf == True
        
        # Check internal transformer parameters
        assert scaler._tfidf_transformer.norm == 'l1'
        assert scaler._tfidf_transformer.use_idf == False
        assert scaler._tfidf_transformer.smooth_idf == False
        assert scaler._tfidf_transformer.sublinear_tf == True
    
    def test_none_norm(self):
        """Test None norm parameter."""
        scaler = TfidfRateScaler(norm=None)
        assert scaler.norm is None
        assert scaler._tfidf_transformer.norm is None


class TestTfidfRateScalerSetParams:
    """Test the set_params method."""
    
    def test_set_params_returns_self(self):
        """Test that set_params returns self."""
        scaler = TfidfRateScaler()
        result = scaler.set_params(norm='l1')
        assert result is scaler
    
    def test_parameter_updates(self):
        """Test that parameters are updated correctly."""
        scaler = TfidfRateScaler(norm='l2', use_idf=True)
        
        scaler.set_params(norm='l1', use_idf=False, sublinear_tf=True)
        
        # Check our parameters
        assert scaler.norm == 'l1'
        assert scaler.use_idf == False
        assert scaler.sublinear_tf == True
        
        # Check internal transformer parameters
        assert scaler._tfidf_transformer.norm == 'l1'
        assert scaler._tfidf_transformer.use_idf == False
        assert scaler._tfidf_transformer.sublinear_tf == True
    
    def test_transformer_instance_preserved(self):
        """Test that the same transformer instance is preserved."""
        scaler = TfidfRateScaler()
        initial_id = id(scaler._tfidf_transformer)
        
        scaler.set_params(norm='l1', use_idf=False)
        final_id = id(scaler._tfidf_transformer)
        
        assert initial_id == final_id
    
    def test_partial_parameter_updates(self):
        """Test partial parameter updates."""
        scaler = TfidfRateScaler(norm='l2', use_idf=True, smooth_idf=True)
        
        scaler.set_params(norm='l1')  # Only update norm
        
        assert scaler.norm == 'l1'
        assert scaler.use_idf == True  # Should remain unchanged
        assert scaler.smooth_idf == True  # Should remain unchanged
    
    def test_invalid_parameter_raises_error(self):
        """Test that invalid parameters raise ValueError."""
        scaler = TfidfRateScaler()
        
        with pytest.raises(ValueError, match="Invalid parameter"):
            scaler.set_params(invalid_param=True)


class TestTfidfRateScalerFit:
    """Test the fit method."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample count rates data for testing."""
        return np.array([
            [0.5, 0.0, 0.25, 0.0, 0.25],
            [0.0, 0.33, 0.33, 0.33, 0.0],
            [0.4, 0.2, 0.0, 0.4, 0.0],
            [0.0, 0.0, 0.5, 0.25, 0.25],
        ], dtype=np.float32)
    
    def test_fit_returns_self(self, sample_data):
        """Test that fit returns self."""
        scaler = TfidfRateScaler()
        result = scaler.fit(sample_data)
        assert result is scaler
    
    def test_fit_with_idf_creates_idf_vector(self, sample_data):
        """Test that fit with use_idf=True creates IDF vector."""
        scaler = TfidfRateScaler(use_idf=True)
        scaler.fit(sample_data)
        
        assert hasattr(scaler, 'idf_')
        assert scaler.idf_ is not None
        assert len(scaler.idf_) == sample_data.shape[1]
        assert all(idf > 0 for idf in scaler.idf_)
    
    def test_fit_without_idf_no_idf_vector(self, sample_data):
        """Test that fit with use_idf=False doesn't create IDF vector."""
        scaler = TfidfRateScaler(use_idf=False)
        scaler.fit(sample_data)
        
        # Should not have idf_ attribute or it should be None
        assert not hasattr(scaler, 'idf_') or scaler.idf_ is None
    
    def test_fit_dense_vs_sparse_identical_idf(self, sample_data):
        """Test that dense and sparse inputs produce identical IDF values."""
        scaler_dense = TfidfRateScaler(use_idf=True)
        scaler_sparse = TfidfRateScaler(use_idf=True)
        
        scaler_dense.fit(sample_data)
        scaler_sparse.fit(sp.csr_matrix(sample_data))
        
        np.testing.assert_allclose(scaler_dense.idf_, scaler_sparse.idf_)
    
    def test_fit_different_dtypes(self, sample_data):
        """Test fit with different data types."""
        scaler = TfidfRateScaler()
        
        # Test float64
        scaler.fit(sample_data.astype(np.float64))
        assert hasattr(scaler, 'idf_')
        
        # Test float32
        scaler.fit(sample_data.astype(np.float32))
        assert hasattr(scaler, 'idf_')


class TestTfidfRateScalerTransform:
    """Test the transform method."""
    
    @pytest.fixture
    def fitted_scaler(self):
        """A fitted scaler for testing."""
        X = np.array([
            [0.5, 0.0, 0.25, 0.0, 0.25],
            [0.0, 0.33, 0.33, 0.33, 0.0],
            [0.4, 0.2, 0.0, 0.4, 0.0],
            [0.0, 0.0, 0.5, 0.25, 0.25],
        ], dtype=np.float32)
        
        scaler = TfidfRateScaler(use_idf=True, norm='l2')
        scaler.fit(X)
        return scaler, X
    
    def test_transform_basic_functionality(self, fitted_scaler):
        """Test basic transform functionality."""
        scaler, X = fitted_scaler
        X_transformed = scaler.transform(X)
        
        assert X_transformed.shape == X.shape
        assert sp.issparse(X_transformed)
    
    def test_transform_copy_parameter(self, fitted_scaler):
        """Test copy parameter in transform."""
        scaler, X = fitted_scaler
        
        X_copy = scaler.transform(X, copy=True)
        X_no_copy = scaler.transform(X, copy=False)
        
        # Results should be the same regardless of copy parameter
        np.testing.assert_allclose(X_copy.toarray(), X_no_copy.toarray())
    
    def test_transform_not_fitted_raises_error(self):
        """Test that transform raises error when not fitted."""
        scaler = TfidfRateScaler()
        X = np.array([[0.5, 0.3, 0.2]])
        
        with pytest.raises(NotFittedError):
            scaler.transform(X)
    
    @pytest.mark.parametrize("use_idf,norm,sublinear_tf", [
        (True, 'l2', False),
        (True, 'l1', False),
        (True, None, False),
        (False, 'l2', False),
        (True, 'l2', True),
        (False, None, True),
    ])
    def test_transform_parameter_combinations(self, use_idf, norm, sublinear_tf):
        """Test transform with different parameter combinations."""
        X = np.array([
            [0.5, 0.0, 0.25],
            [0.0, 0.33, 0.33],
            [0.4, 0.2, 0.0],
        ], dtype=np.float32)
        
        scaler = TfidfRateScaler(
            use_idf=use_idf, 
            norm=norm, 
            sublinear_tf=sublinear_tf
        )
        X_transformed = scaler.fit_transform(X)
        
        assert X_transformed.shape == X.shape
        assert sp.issparse(X_transformed)
    
    def test_transform_sparse_input(self, fitted_scaler):
        """Test transform with sparse input."""
        scaler, X = fitted_scaler
        X_sparse = sp.csr_matrix(X)
        
        X_transformed = scaler.transform(X_sparse)
        assert sp.issparse(X_transformed)
        assert X_transformed.shape == X.shape


class TestTfidfRateScalerFitTransform:
    """Test the fit_transform method."""
    
    def test_fit_transform_equivalent_to_separate_calls(self):
        """Test that fit_transform produces same result as separate fit + transform."""
        X = np.array([
            [0.5, 0.0, 0.25],
            [0.0, 0.33, 0.33],
            [0.4, 0.2, 0.0],
        ], dtype=np.float32)
        
        # fit_transform
        scaler1 = TfidfRateScaler(use_idf=True, norm='l2')
        X_fit_transform = scaler1.fit_transform(X)
        
        # Separate fit and transform
        scaler2 = TfidfRateScaler(use_idf=True, norm='l2')
        scaler2.fit(X)
        X_separate = scaler2.transform(X)
        
        np.testing.assert_allclose(
            X_fit_transform.toarray(), 
            X_separate.toarray()
        )
    
    def test_fit_transform_with_different_parameters(self):
        """Test fit_transform with different parameters."""
        X = np.array([
            [0.5, 0.0, 0.25],
            [0.0, 0.33, 0.33],
        ], dtype=np.float32)
        
        scaler = TfidfRateScaler(use_idf=False, norm='l1', sublinear_tf=True)
        X_result = scaler.fit_transform(X)
        
        assert X_result.shape == X.shape
        assert sp.issparse(X_result)


class TestTfidfRateScalerGetFeatureNamesOut:
    """Test the get_feature_names_out method."""
    
    def test_get_feature_names_out_without_input(self):
        """Test get_feature_names_out without input features."""
        X = np.array([
            [0.5, 0.0, 0.25],
            [0.0, 0.33, 0.33],
        ], dtype=np.float32)
        
        scaler = TfidfRateScaler()
        scaler.fit(X)
        
        feature_names = scaler.get_feature_names_out()
        assert len(feature_names) == X.shape[1]
    
    def test_get_feature_names_out_with_input(self):
        """Test get_feature_names_out with input features."""
        X = np.array([
            [0.5, 0.0, 0.25],
            [0.0, 0.33, 0.33],
        ], dtype=np.float32)
        
        input_features = ['feature_0', 'feature_1', 'feature_2']
        
        scaler = TfidfRateScaler()
        scaler.fit(X)
        
        feature_names = scaler.get_feature_names_out(input_features)
        assert len(feature_names) == len(input_features)
    
    def test_get_feature_names_out_not_fitted_raises_error(self):
        """Test that get_feature_names_out raises error when not fitted."""
        scaler = TfidfRateScaler()
        
        with pytest.raises(NotFittedError):
            scaler.get_feature_names_out()


class TestTfidfRateScalerComparisonWithStandard:
    """Test comparison with standard TfidfTransformer."""
    
    @pytest.mark.parametrize("use_idf,norm,sublinear_tf", [
        (True, 'l2', False),
        (True, 'l1', False),
        (True, None, False),
        (False, 'l2', False),
        (True, 'l2', True),
    ])
    def test_results_match_standard_tfidf(self, use_idf, norm, sublinear_tf):
        """Test that results match standard TfidfTransformer."""
        X_rates = np.array([
            [0.5, 0.0, 0.25, 0.0, 0.25],
            [0.0, 0.33, 0.33, 0.33, 0.0],
            [0.4, 0.2, 0.0, 0.4, 0.0],
            [0.0, 0.0, 0.5, 0.25, 0.25],
        ], dtype=np.float32)
        
        X_binary = (X_rates > 0).astype(np.int32)
        
        # Our implementation
        our_scaler = TfidfRateScaler(
            use_idf=use_idf, 
            norm=norm, 
            sublinear_tf=sublinear_tf
        )
        X_our = our_scaler.fit_transform(X_rates)
        
        # Standard TfidfTransformer (fit on binary, transform on rates)
        standard_scaler = TfidfTransformer(
            use_idf=use_idf, 
            norm=norm, 
            sublinear_tf=sublinear_tf
        )
        standard_scaler.fit(X_binary)
        X_standard = standard_scaler.transform(X_rates)
        
        # Results should be identical
        np.testing.assert_allclose(X_our.toarray(), X_standard.toarray())
        
        # IDF values should match when use_idf=True
        if use_idf:
            np.testing.assert_allclose(our_scaler.idf_, standard_scaler.idf_)


class TestTfidfRateScalerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_all_zero_column(self):
        """Test handling of all-zero column."""
        X = np.array([
            [0.5, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [0.4, 0.0, 0.0],
        ], dtype=np.float32)
        
        scaler = TfidfRateScaler(use_idf=True, smooth_idf=True)
        X_result = scaler.fit_transform(X)
        
        assert X_result.shape == X.shape
        assert not np.any(np.isnan(X_result.toarray()))
    
    def test_single_sample(self):
        """Test with single sample."""
        X = np.array([[0.5, 0.3, 0.2]], dtype=np.float32)
        
        scaler = TfidfRateScaler()
        X_result = scaler.fit_transform(X)
        
        assert X_result.shape == X.shape
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[0.5], [0.3], [0.2]], dtype=np.float32)
        
        scaler = TfidfRateScaler()
        X_result = scaler.fit_transform(X)
        
        assert X_result.shape == X.shape
    
    def test_very_small_values(self):
        """Test with very small values."""
        X = np.array([
            [1e-6, 0.0, 1e-8],
            [0.0, 1e-7, 1e-6],
        ], dtype=np.float32)
        
        scaler = TfidfRateScaler()
        X_result = scaler.fit_transform(X)
        
        assert X_result.shape == X.shape
        assert not np.any(np.isnan(X_result.toarray()))


class TestTfidfRateScalerPipelineCompatibility:
    """Test compatibility with sklearn Pipeline."""
    
    def test_basic_pipeline(self):
        """Test basic pipeline functionality."""
        X = np.random.rand(50, 20).astype(np.float32) * 0.5
        y = np.random.randint(0, 2, 50)
        
        pipeline = Pipeline([
            ("scaler", TfidfRateScaler(use_idf=True, norm='l2')),
            ("clf", LogisticRegression(random_state=42, max_iter=100)),
        ])
        
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_pipeline_parameter_setting(self):
        """Test parameter setting in pipeline."""
        X = np.random.rand(30, 10).astype(np.float32) * 0.5
        y = np.random.randint(0, 2, 30)
        
        pipeline = Pipeline([
            ("scaler", TfidfRateScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=50)),
        ])
        
        # Set parameters
        pipeline.set_params(
            scaler__use_idf=False, 
            scaler__norm='l1',
            clf__C=0.1
        )
        
        pipeline.fit(X, y)
        
        # Check parameters were set
        assert pipeline.named_steps['scaler'].use_idf == False
        assert pipeline.named_steps['scaler'].norm == 'l1'
        assert pipeline.named_steps['clf'].C == 0.1
    
    def test_pipeline_with_sample_weight(self):
        """Test pipeline with sample weights."""
        X = np.random.rand(30, 10).astype(np.float32) * 0.5
        y = np.random.randint(0, 2, 30)
        sample_weight = np.ones(30)
        sample_weight[:10] = 2.0
        
        pipeline = Pipeline([
            ("scaler", TfidfRateScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=50)),
        ])
        
        pipeline.fit(X, y, clf__sample_weight=sample_weight)
        predictions = pipeline.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_column_transformer_compatibility(self):
        """Test compatibility with ColumnTransformer."""
        n_samples = 50
        n_bowav_features = 20
        n_other_features = 10
        
        X = np.random.rand(n_samples, n_bowav_features + n_other_features).astype(np.float32)
        X[:, :n_bowav_features] *= 0.5  # bowav features (rates)
        X[:, n_bowav_features:] *= 10   # other features
        y = np.random.randint(0, 2, n_samples)
        
        column_transformer = ColumnTransformer([
            ("bowav", TfidfRateScaler(use_idf=True, norm='l2'), 
             slice(0, n_bowav_features)),
        ], remainder="passthrough")
        
        pipeline = Pipeline([
            ("scaler", column_transformer),
            ("clf", LogisticRegression(random_state=42, max_iter=50)),
        ])
        
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_grid_search_compatibility(self):
        """Test compatibility with GridSearchCV."""
        X = np.random.rand(30, 10).astype(np.float32) * 0.5
        y = np.random.randint(0, 2, 30)
        
        pipeline = Pipeline([
            ("scaler", TfidfRateScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=50)),
        ])
        
        param_grid = {
            'scaler__use_idf': [True, False],
            'scaler__norm': ['l1', 'l2'],
            'clf__C': [0.1, 1.0],
        }
        
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=3, 
            scoring='accuracy',
            n_jobs=1
        )
        
        grid_search.fit(X, y)
        
        assert hasattr(grid_search, 'best_score_')
        assert hasattr(grid_search, 'best_params_')
    
    def test_pipeline_cloning(self):
        """Test pipeline cloning."""
        X = np.random.rand(20, 5).astype(np.float32) * 0.5
        y = np.random.randint(0, 2, 20)
        
        original_pipeline = Pipeline([
            ("scaler", TfidfRateScaler(use_idf=True, norm='l2')),
            ("clf", LogisticRegression(random_state=42, max_iter=50)),
        ])
        
        cloned_pipeline = clone(original_pipeline)
        
        # Fit both pipelines
        original_pipeline.fit(X, y)
        cloned_pipeline.fit(X, y)
        
        # They should produce the same results
        original_pred = original_pipeline.predict(X)
        cloned_pred = cloned_pipeline.predict(X)
        
        np.testing.assert_array_equal(original_pred, cloned_pred)


if __name__ == "__main__":
    pytest.main([__file__])
