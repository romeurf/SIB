import unittest
import numpy as np

from si.model_selection.cross_validate import k_fold_cross_validation
from si.model_selection.randomized_search import randomized_search_cv


class DummyModel:
    """
    Simple model with 2 hyperparameters that affect performance in a known way.
    """

    def __init__(self):
        # hyperparameters to tune
        self.l2_penalty = 1.0
        self.alpha = 0.001
        # store last "fitted" dataset for sanity
        self.fitted_X = None
        self.fitted_y = None

    def fit(self, X, y):
        self.fitted_X = X
        self.fitted_y = y
        return self

    def predict(self, X):
        # trivial prediction rule using alpha and l2_penalty

        threshold = (self.l2_penalty + self.alpha * 1000) / 10.0
        return (X.sum(axis=1) > threshold).astype(int)


def dummy_scoring(y_true, y_pred):
    # simple accuracy
    return (y_true == y_pred).mean()

class TestRandomizedSearchCV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # small binary classification toy dataset
        rng = np.random.RandomState(0)
        X = rng.normal(size=(60, 3))
        # true label depends on sum of features
        y = (X.sum(axis=1) > 0).astype(int)
        cls.dataset = (X, y)

    def test_invalid_hyperparameter_raises(self):
        model = DummyModel()
        bad_grid = {"nonexistent_param": [1, 2, 3]}

        with self.assertRaises(ValueError):
            randomized_search_cv(
                model=model,
                dataset=self.dataset,
                hyperparameter_grid=bad_grid,
                scoring=dummy_scoring,
                cv=3,
                n_iter=2,
            )

    def test_returns_correct_structure_and_lengths(self):
        model = DummyModel()
        grid = {
            "l2_penalty": [1, 2],
            "alpha": [0.001, 0.01],
        }

        results = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_grid=grid,
            scoring=dummy_scoring,
            cv=3,
            n_iter=3,
        )

        # keys exist
        self.assertIn("hyperparameters", results)
        self.assertIn("scores", results)
        self.assertIn("best_hyperparameters", results)
        self.assertIn("best_score", results)

        # lengths match n_iter (or total combinations if smaller)
        self.assertEqual(len(results["hyperparameters"]), len(results["scores"]))
        self.assertLessEqual(len(results["hyperparameters"]), 4)  # 2x2 grid
        self.assertLessEqual(len(results["hyperparameters"]), 3)  # n_iter=3

    def test_best_score_corresponds_to_best_hyperparameters(self):
        model = DummyModel()
        # design grid so we know which combination should be best
        # suppose we expect l2_penalty=1, alpha=0.01 to work best
        grid = {
            "l2_penalty": [1, 5],
            "alpha": [0.001, 0.01],
        }

        results = randomized_search_cv(
            model=model,
            dataset=self.dataset,
            hyperparameter_grid=grid,
            scoring=dummy_scoring,
            cv=3,
            n_iter=4,  # all combinations
        )

        # best_hyperparameters is one of the tried combinations
        self.assertIn(results["best_hyperparameters"], results["hyperparameters"])

        # best_score equals the max of all scores
        self.assertAlmostEqual(
            results["best_score"], max(results["scores"]), places=10
        )


if __name__ == "__main__":
    unittest.main()