import unittest
import numpy as np
from sklearn.datasets import load_iris
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
from si.feature_selection.select_percentile import SelectPercentile

class TestSelectPercentile(unittest.TestCase):
    
    def setUp(self):
        # Load data once
        data = load_iris()
        self.X = data.data
        self.y = data.target
        self.features = data.feature_names
        # 4. Create a Dataset object for testing
        self.dataset = Dataset(self.X, self.y, features=self.features, label="iris_species")

    def test_init_default_parameters(self):
        sp = SelectPercentile()
        self.assertEqual(sp.percentile, 10)
        # 5. Test that the default function is your f_classification
        self.assertEqual(sp.score_func, f_classification) 
        self.assertIsNone(sp.F)
        self.assertIsNone(sp.p)

    def test_init_custom_percentile(self):
        sp = SelectPercentile(percentile=50)
        self.assertEqual(sp.percentile, 50)
        
    def test_invalid_percentile(self):
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=0)
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=-10)
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=101)

    def test_fit(self):
        sp = SelectPercentile(score_func=f_classification, percentile=50)
        # 6. Fit using the Dataset object
        result = sp.fit(self.dataset) 
        self.assertIs(result, sp)
        self.assertIsNotNone(sp.F)
        self.assertEqual(len(sp.F), self.X.shape[1])
        self.assertIsNotNone(sp.p)
        self.assertEqual(len(sp.p), self.X.shape[1])
        self.assertTrue(sp.is_fitted()) # Check the fitted method

    def test_transform(self):
        sp = SelectPercentile(score_func=f_classification, percentile=50)
        # 7. Fit and Transform using the Dataset object
        sp.fit(self.dataset)
        transformed_dataset = sp.transform(self.dataset)
        
        # Check that it returns a Dataset object
        self.assertIsInstance(transformed_dataset, Dataset)
        # Check that the shape is correct (50% of 4 features = 2)
        self.assertEqual(transformed_dataset.X.shape[0], self.X.shape[0])
        self.assertEqual(transformed_dataset.X.shape[1], 2)
        # Check that feature names are also filtered
        self.assertEqual(len(transformed_dataset.features), 2)

    def test_transform_without_fit(self):
        sp = SelectPercentile(percentile=50)
        with self.assertRaises(ValueError):
            # 8. Test transform with Dataset
            sp.transform(self.dataset) 

    def test_fit_transform(self):
        sp = SelectPercentile(score_func=f_classification, percentile=50)
        # 9. Test fit_transform with Dataset
        transformed_dataset = sp.fit_transform(self.dataset)
        
        self.assertIsInstance(transformed_dataset, Dataset)
        self.assertEqual(transformed_dataset.X.shape[1], 2)
        self.assertTrue(sp.is_fitted())

    def test_get_support(self):
        sp = SelectPercentile(score_func=f_classification, percentile=50)
        sp.fit(self.dataset)
        
        # Test boolean mask
        support_mask = sp.get_support()
        self.assertEqual(support_mask.dtype, bool)
        self.assertEqual(len(support_mask), 4)
        self.assertEqual(np.sum(support_mask), 2) # 2 features selected
        
        # Test indices
        support_indices = sp.get_support(indices=True)
        self.assertEqual(support_indices.dtype, np.int64)
        self.assertEqual(len(support_indices), 2)
        # For iris, the 2 best features are petal length and petal width (indices 2, 3)
        np.testing.assert_array_equal(support_indices, [2, 3])

    def test_get_support_without_fit(self):
        sp = SelectPercentile(percentile=50)
        with self.assertRaises(ValueError):
            sp.get_support()