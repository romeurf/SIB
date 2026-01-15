import unittest
import numpy as np

from si.neural_networks.optimizers import Adam


class TestAdamOptimizer(unittest.TestCase):
    def test_initial_update_matches_manual_computation(self):
        # simple 1D weight and gradient so we can compute by hand
        w = np.array([1.0])
        grad = np.array([0.1])

        lr = 0.001
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        opt = Adam(
            learning_rate=lr,
            beta_1=beta1,
            beta_2=beta2,
            epsilon=eps,
        )

        # first update call (t will become 1)
        w_new = opt.update(w.copy(), grad.copy())

        # manual Adam step for t = 1
        t = 1
        m1 = (1 - beta1) * grad         # m_1
        v1 = (1 - beta2) * (grad ** 2)  # v_1

        m_hat = m1 / (1 - beta1 ** t)
        v_hat = v1 / (1 - beta2 ** t)

        w_expected = w - lr * m_hat / (np.sqrt(v_hat) + eps)

        self.assertTrue(np.allclose(w_new, w_expected, atol=1e-10))

    def test_weights_move_in_negative_gradient_direction(self):
        w = np.array([2.0, -1.0])
        grad = np.array([0.5, -0.25])  # positive, negative

        opt = Adam(learning_rate=0.01)
        w_new = opt.update(w.copy(), grad.copy())

        # For gradient descent, weights should move opposite to gradient:
        # w_new = w - step * grad_effect
        # So if grad > 0 => w_new < w; if grad < 0 => w_new > w.
        self.assertLess(w_new[0], w[0])  # moved down for positive grad
        self.assertGreater(w_new[1], w[1])  # moved up for negative grad

    def test_state_persists_across_steps(self):
        # make sure m, v, t update over multiple steps
        w = np.array([1.0])
        grad1 = np.array([0.1])
        grad2 = np.array([-0.2])

        opt = Adam(learning_rate=0.001)

        w1 = opt.update(w.copy(), grad1.copy())
        t_after_first = opt.t
        m_after_first = opt.m.copy()
        v_after_first = opt.v.copy()

        self.assertEqual(t_after_first, 1)
        # second step should not reinitialize m, v
        w2 = opt.update(w1.copy(), grad2.copy())

        self.assertEqual(opt.t, 2)
        self.assertFalse(np.allclose(opt.m, m_after_first))
        self.assertFalse(np.allclose(opt.v, v_after_first))

        # state arrays have same shape as weights
        self.assertEqual(opt.m.shape, w.shape)
        self.assertEqual(opt.v.shape, w.shape)


if __name__ == "__main__":
    unittest.main()
