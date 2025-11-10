import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import f_classif
from si.feature_selection.select_percentile import SelectPercentile

class TestSelectPercentileInitialization(unittest.TestCase):
    def test_init_default_parameters(self):
        sp = SelectPercentile()
        self.assertEqual(sp.percentile, 10)
        self.assertIsNone(sp.score_func)
        self.assertIsNone(sp.F)
        self.assertIsNone(sp.p)

    def test_init_custom_parameters(self):
        sp = SelectPercentile(score_func=f_classif, percentile=50)
        self.assertEqual(sp.percentile, 50)
        self.assertEqual(sp.score_func, f_classif)

    def test_invalid_percentile_zero(self):
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=0)

    def test_invalid_percentile_negative(self):
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=-10)

    def test_invalid_percentile_over_100(self):
        with self.assertRaises(ValueError):
            SelectPercentile(percentile=101)

    def test_valid_percentile_boundaries(self):
        sp1 = SelectPercentile(percentile=1)
        self.assertEqual(sp1.percentile, 1)
        sp2 = SelectPercentile(percentile=100)
        self.assertEqual(sp2.percentile, 100)

class TestSelectPercentileFit(unittest.TestCase):
    def setUp(self):
        data = load_iris()
        self.X = data.data
        self.y = data.target

    def test_fit_basic(self):
        sp = SelectPercentile(score_func=f_classif, percentile=50)
        result = sp.fit(self.X, self.y)
        self.assertIs(result, sp)
        self.assertIsNotNone(sp.F)
        self.assertEqual(len(sp.F), self.X.shape[1])
        self.assertIsNotNone(sp.p)
        self.assertEqual(len(sp.p), self.X.shape[1])

    def test_fit_without_score_func(self):
        sp = SelectPercentile(score_func=None)
        with self.assertRaises(ValueError):
            sp.fit(self.X, self.y)

    def test_fit_stores_f_values(self):
        sp = SelectPercentile(score_func=f_classif, percentile=75)
        sp.fit(self.X, self.y)
        self.assertIsInstance(sp.F, np.ndarray)
        self.assertEqual(len(sp.F), 4)  # Iris has 4 features

class TestSelectPercentileTransform(unittest.TestCase):
    def setUp(self):
        data = load_iris()
        self.X = data.data
        self.y = data.target

    def test_transform_basic(self):
        sp = SelectPercentile(score_func=f_classif, percentile=50)
        sp.fit(self.X, self.y)
        X_transformed = sp.transform(self.X)
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        self.assertLess(X_transformed.shape[1], self.X.shape[1])

    def test_transform_without_fit(self):
        sp = SelectPercentile(score_func=f_classif, percentile=50)
        with self.assertRaises(ValueError):
            sp.transform(self.X)

    def test_transform_percentile_100(self):
        sp = SelectPercentile(score_func=f_classif, percentile=100)
        sp.fit(self.X, self.y)
        X_transformed = sp.transform(self.X)
        self.assertEqual(X_transformed.shape, self.X.shape)

    def test_transform_percentile_25(self):
        sp = SelectPercentile(score_func=f_classif, percentile=25)
        sp.fit(self.X, self.y)
        X_transformed = sp.transform(self.X)
        self.assertEqual(X_transformed.shape[1], 1)

class TestSelectPercentileTieHandling(unittest.TestCase):
    def test_tie_handling_maintains_percentile_count(self):
        X = np.random.rand(50, 20)
        y = np.random.rand(50)
        f_values = np.array([1, 2, 2, 2, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        sp = SelectPercentile(score_func=lambda X, y: (f_values, np.zeros_like(f_values)), percentile=30)
        sp.fit(X, y)
        X_transformed = sp.transform(X)
        self.assertEqual(X_transformed.shape[1], 6)

    def test_tie_handling_all_same_values(self):
        X = np.random.rand(30, 10)
        y = np.random.rand(30)
        f_values = np.ones(10) * 5.0
        sp = SelectPercentile(score_func=lambda X, y: (f_values, np.zeros_like(f_values)), percentile=50)
        sp.fit(X, y)
        X_transformed = sp.transform(X)
        self.assertEqual(X_transformed.shape[1], 5)

class TestSelectPercentileFitTransform(unittest.TestCase):
    def setUp(self):
        data = load_iris()
        self.X = data.data
        self.y = data.target

    def test_fit_transform_basic(self):
        sp = SelectPercentile(score_func=f_classif, percentile=50)
        X_transformed = sp.fit_transform(self.X, self.y)
        self.assertEqual(X_transformed.shape[0], self.X.shape[0])
        self.assertLess(X_transformed.shape[1], self.X.shape[1])

    def test_fit_transform_equals_fit_then_transform(self):
        sp1 = SelectPercentile(score_func=f_classif, percentile=60)
        X_fit_transform = sp1.fit_transform(self.X, self.y)
        sp2 = SelectPercentile(score_func=f_classif, percentile=60)
        X_fit_then_transform = sp2.fit(self.X, self.y).transform(self.X)
        np.testing.assert_array_almost_equal(X_fit_transform, X_fit_then_transform)

class TestSelectPercentileGetSupport(unittest.TestCase):
    def setUp(self):
        data = load_iris()
        self.X = data.data
        self.y = data.target

    def test_get_support_boolean_mask(self):
        sp = SelectPercentile(score_func=f_classif, percentile=50)
        sp.fit(self.X, self.y)
        support = sp.get_support()
        self.assertIsInstance(support, np.ndarray)
        self.assertEqual(support.dtype, bool)
        self.assertEqual(len(support), self.X.shape[1])

    def test_get_support_without_fit(self):
        sp = SelectPercentile(score_func=f_classif, percentile=50)
        with self.assertRaises(ValueError):
            sp.get_support()