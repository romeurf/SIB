import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    def test_dropna(self):
        # Create data with one row containing NaN
        X = np.array([
            [1, 2, np.nan],
            [4, 5, 6],
            [7, 8, 9]
        ])
        y = np.array([1, 0, 1])
        ds = Dataset(np.copy(X), np.copy(y))
        ds.dropna()
        # Only rows without NaN remain
        assert ds.X.shape == (2, 3)
        assert ds.y.shape == (2,)
        assert np.all(~np.isnan(ds.X))

    def test_fillna_constant(self):
        X = np.array([
            [1, np.nan],
            [3, 4]
        ])
        y = np.array([0, 1])
        ds = Dataset(np.copy(X), np.copy(y))
        ds.fillna(0.0)
        # NaN replaced with 0.0
        assert ds.X[0, 1] == 0.0
        assert np.all(~np.isnan(ds.X))

    def test_fillna_mean():
        X = np.array([
            [np.nan, 2],
            [4, 6],
            [8, 10]
        ])
        y = np.array([0, 1, 1])
        ds = Dataset(np.copy(X), np.copy(y))
        ds.fillna('mean')
        # NaN replaced with mean of column
        expected_mean = np.mean([4, 8])  # mean for column 0
        assert np.isclose(ds.X[0, 0], expected_mean)
        assert np.all(~np.isnan(ds.X))

    def test_fillna_median():
        X = np.array([
            [np.nan, 2],
            [7, 6],
            [8, 10]
        ])
        y = np.array([0, 1, 1])
        ds = Dataset(np.copy(X), np.copy(y))
        ds.fillna('median')
        # NaN replaced with median of column
        expected_median = np.median([7, 8])  # column 0
        assert np.isclose(ds.X[0, 0], expected_median)
        assert np.all(~np.isnan(ds.X))

    def test_remove_by_index():
        X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        y = np.array([0, 1, 1])
        ds = Dataset(np.copy(X), np.copy(y))
        ds.remove_by_index(1)
        # Row at index 1 is gone
        assert ds.X.shape == (2, 3)
        assert ds.y.shape == (2,)
        assert np.array_equal(ds.X, [[1, 2, 3], [7, 8, 9]])
        assert np.array_equal(ds.y, [0, 1])