import unittest
import numpy as np

from si.ensemble.stacking_classifier import StackingClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression


class TestStackingClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a simple synthetic dataset
        # Two clusters in 2D, binary labels 0 and 1
        rng = np.random.RandomState(0)
        X0 = rng.normal(loc=-1.0, scale=0.5, size=(50, 2))
        X1 = rng.normal(loc=1.0, scale=0.5, size=(50, 2))
        cls.X = np.vstack([X0, X1])
        cls.y = np.array([0] * 50 + [1] * 50)

    def test_fit_creates_meta_model(self):
        # Two simple base models + logistic regression as final model
        base1 = KNNClassifier(k=3)
        base2 = KNNClassifier(k=5)
        final = LogisticRegression()

        stack = StackingClassifier(models=[base1, base2], final_model=final)
        stack.fit(self.X, self.y)

        # Check that base models are fitted by calling predict without error
        preds1 = base1.predict(self.X)
        preds2 = base2.predict(self.X)
        self.assertEqual(preds1.shape, self.y.shape)
        self.assertEqual(preds2.shape, self.y.shape)

        # Check that final model can predict from meta-features via ensemble
        y_pred = stack.predict(self.X)
        self.assertEqual(y_pred.shape, self.y.shape)

    def test_predict_uses_stacked_predictions(self):
        base1 = KNNClassifier(k=1)
        base2 = KNNClassifier(k=3)
        final = LogisticRegression()

        stack = StackingClassifier(models=[base1, base2], final_model=final)
        stack.fit(self.X, self.y)

        y_pred = stack.predict(self.X)

        # Accuracy
        acc = stack.score(self.X, self.y)
        self.assertGreater(acc, 0.8)

    def test_score_matches_manual_accuracy(self):
        base1 = KNNClassifier(k=3)
        base2 = KNNClassifier(k=5)
        final = LogisticRegression()

        stack = StackingClassifier(models=[base1, base2], final_model=final)
        stack.fit(self.X, self.y)

        # ensemble score
        acc_stack = stack.score(self.X, self.y)

        # manual accuracy computation
        y_pred = stack.predict(self.X)
        acc_manual = (y_pred == self.y).sum() / len(self.y)

        self.assertAlmostEqual(acc_stack, acc_manual, places=6)


if __name__ == "__main__":
    unittest.main()