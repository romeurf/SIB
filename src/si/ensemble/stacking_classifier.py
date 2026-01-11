# Exercise 10: Implement the StackingClassifier class by taking into consideration the structure of the class shown in the next slides

from si.base.model import Model


class StackingClassifier(Model):
    """
    StackingClassifier(models, final_model)

    This classifier implements a simple stacking ensemble:
    - A set of base models (level-0 models) are trained on the original data.
    - Their predictions are used as input features to a final model (level-1 model).
    - The final model makes the ultimate prediction.

    Parameters
    ----------
    models : list
        List of base model instances (must implement fit, predict).
    final_model : object
        Final model instance (must implement fit, predict).
    """

    def __init__(self, models, final_model):
        # Store base models and final model
        self.models = models
        self.final_model = final_model

    def _fit(self, X, y):
        """
        Train the stacking ensemble.

        Steps:
        1. Train the initial set of models on (X, y).
        2. Get predictions from each of these models on X.
        3. Use these predictions as new features to train the final model.
        4. Return self.
        """
        import numpy as np

        # 1) Fit each base model on the original data
        for model in self.models:
            model.fit(X, y)

        # 2) Collect predictions from each base model on X
        #    Each prediction vector has length n_samples.
        base_preds = []
        for model in self.models:
            # model.predict(X) -> shape (n_samples,)
            preds = model.predict(X)
            base_preds.append(preds)

        # 3) Stack predictions column-wise to form meta-features
        #    If there are M base models and N samples,
        #    meta_X has shape (N, M).
        meta_X = np.column_stack(base_preds)

        # 4) Train the final model on the meta-features and original targets
        self.final_model.fit(meta_X, y)

        return self

    def _predict(self, X):
        """
        Predict labels for new data using the stacking ensemble.

        Steps:
        1. Obtain predictions from each base model on X.
        2. Stack these predictions to form meta-features.
        3. Use the final model to predict from these meta-features.
        """
        import numpy as np

        # 1) Get predictions from each base model on X
        base_preds = []
        for model in self.models:
            preds = model.predict(X)
            base_preds.append(preds)

        # 2) Stack predictions into meta-features (N_samples, N_models)
        meta_X = np.column_stack(base_preds)

        # 3) Use the final model to obtain final predictions
        final_preds = self.final_model.predict(meta_X)
        return final_preds

    def _score(self, X, y):
        """
        Compute the accuracy of the stacking ensemble.

        Steps:
        1. Get predictions using the predict method.
        2. Compute accuracy between predicted and true labels.
        """
        # 1) Obtain predictions
        y_pred = self.predict(X)

        # 2) Compute accuracy
        correct = (y_pred == y).sum()
        total = len(y)
        accuracy = correct / total

        return accuracy