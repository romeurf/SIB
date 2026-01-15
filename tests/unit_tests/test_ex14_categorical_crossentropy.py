import unittest
import numpy as np

from si.neural_networks.losses import CategoricalCrossEntropy


class TestCategoricalCrossEntropy(unittest.TestCase):
    def setUp(self):
        self.loss_fn = CategoricalCrossEntropy()

    def test_loss_zero_when_prediction_is_one_hot_true(self):
        # perfect prediction: loss should be ~0
        y_true = np.array([[1, 0, 0],
                           [0, 1, 0]])
        y_pred = np.array([[1, 0, 0],
                           [0, 1, 0]])  # already clipped by implementation

        loss = self.loss_fn.loss(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0, places=7)

    def test_loss_positive_for_imperfect_prediction(self):
        y_true = np.array([[1, 0, 0]])
        y_pred = np.array([[0.7, 0.2, 0.1]])

        loss = self.loss_fn.loss(y_true, y_pred)
        # negative log of 0.7
        expected = -np.log(0.7)
        self.assertAlmostEqual(loss, expected, places=7)
        self.assertGreater(loss, 0.0)

    def test_derivative_matches_y_pred_minus_y_true(self):
        y_true = np.array([[1, 0, 0],
                           [0, 1, 0]])
        y_pred = np.array([[0.8, 0.1, 0.1],
                           [0.2, 0.7, 0.1]])

        grad = self.loss_fn.derivative(y_true, y_pred)
        expected = y_pred - y_true
        self.assertTrue(np.allclose(grad, expected, atol=1e-7))

    def test_clipping_avoids_log_zero(self):
        # y_pred contains exact 0 and 1; implementation should clip internally
        y_true = np.array([[0, 1]])
        y_pred = np.array([[0.0, 1.0]])

        # just verify it does not produce inf or nan
        loss = self.loss_fn.loss(y_true, y_pred)
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()