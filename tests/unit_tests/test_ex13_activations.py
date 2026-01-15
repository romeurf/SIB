import unittest
import numpy as np

from si.neural_networks.activation import TanhActivation, SoftmaxActivation


class TestTanhActivation(unittest.TestCase):
    def test_forward_range_and_values(self):
        act = TanhActivation()
        x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
        y = act.forward_propagation(x, training=True)

        # range must be (-1, 1)
        self.assertTrue(np.all(y > -1.0))
        self.assertTrue(np.all(y < 1.0))

        # known reference values (approx)
        self.assertTrue(np.allclose(y[0, 2], 0.0, atol=1e-7))    # tanh(0)=0
        self.assertTrue(np.allclose(y[0, 3], np.tanh(1.0), atol=1e-7))

    def test_backward_matches_analytical_derivative(self):
        act = TanhActivation()
        x = np.linspace(-2, 2, 5).reshape(1, -1)
        y = act.forward_propagation(x, training=True)

        # derivative from layer
        grad_out = np.ones_like(y)
        grad_in = act.backward_propagation(grad_out)

        # analytical derivative: 1 - tanh(x)^2
        t = np.tanh(x)
        expected = 1 - t ** 2

        self.assertTrue(np.allclose(grad_in, expected, atol=1e-7))


class TestSoftmaxActivation(unittest.TestCase):
    def test_forward_rows_are_probabilities(self):
        act = SoftmaxActivation()
        x = np.array([[1.0, 2.0, 3.0],
                      [0.0, 0.0, 0.0]])
        y = act.forward_propagation(x, training=True)

        # shape preserved
        self.assertEqual(x.shape, y.shape)

        # rows sum to 1
        row_sums = y.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-7))

        # all probabilities in [0, 1]
        self.assertTrue(np.all(y >= 0.0))
        self.assertTrue(np.all(y <= 1.0))

        # second row is uniform distribution (all logits equal)
        self.assertTrue(np.allclose(y[1], np.array([1/3, 1/3, 1/3]), atol=1e-7))

    def test_numerical_stability_invariant_to_constant_shift(self):
        act = SoftmaxActivation()

        base = np.array([[10.0, 0.0, -5.0]])
        shifted = base + 1000.0  # adding constant shouldn't change probabilities

        y1 = act.forward_propagation(base, training=True)
        y2 = act.forward_propagation(shifted, training=True)

        self.assertTrue(np.allclose(y1, y2, atol=1e-7))

    def test_backward_passthrough(self):
        act = SoftmaxActivation()
        x = np.array([[1.0, 2.0, 3.0]])
        y = act.forward_propagation(x, training=True)

        grad_out = np.array([[0.1, -0.2, 0.3]])
        grad_in = act.backward_propagation(grad_out)

        # derivative() returns 1, so layer should just pass gradient through
        self.assertTrue(np.allclose(grad_in, grad_out))


if __name__ == "__main__":
    unittest.main()