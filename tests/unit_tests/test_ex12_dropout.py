import unittest
import numpy as np

from si.neural_networks.layers import Dropout


class TestExercise12Dropout(unittest.TestCase):
    def test_training_mode_mask_and_scaling(self):
        np.random.seed(0)
        x = np.ones((3, 5))
        p = 0.5
        layer = Dropout(probability=p, input_shape=(5,))

        y_train = layer.forward_propagation(x, training=True)

        # Shape consistency
        self.assertEqual(x.shape, y_train.shape)
        self.assertEqual(x.shape, layer.mask.shape)

        # Mask only contains 0s and 1s
        self.assertTrue(np.all(np.isin(layer.mask, [0, 1])))

        # Inverted dropout scaling: non‑zero outputs should equal 1/(1-p)
        keep_prob = 1.0 - p
        scale = 1.0 / keep_prob
        self.assertTrue(np.allclose(y_train[layer.mask == 1], scale))
        self.assertTrue(np.all(y_train[layer.mask == 0] == 0))

    def test_inference_mode_pass_through(self):
        x = np.random.rand(4, 3)
        layer = Dropout(probability=0.5, input_shape=(3,))

        y_eval = layer.forward_propagation(x, training=False)

        # Inference: output must equal input exactly
        self.assertTrue(np.allclose(x, y_eval))
        self.assertEqual(x.shape, layer.mask.shape)

    def test_backward_uses_same_mask(self):
        np.random.seed(1)
        x = np.ones((2, 4))
        layer = Dropout(probability=0.5, input_shape=(4,))

        # Create mask via training forward pass
        _ = layer.forward_propagation(x, training=True)

        grad_out = np.ones_like(x)
        grad_in = layer.backward_propagation(grad_out)

        # Gradient is zero where mask==0, one where mask==1
        self.assertTrue(np.array_equal(grad_in, layer.mask))

    def test_output_shape_and_parameters(self):
        layer = Dropout(probability=0.3, input_shape=(10,))
        self.assertEqual(layer.output_shape(), (10,))
        self.assertEqual(layer.parameters(), 0)


if __name__ == "__main__":
    unittest.main()