import numpy as np
from si.data.dataset import Dataset
from si.base.model import Model
from si.metrics.mse import mse

class RidgeRegressionLeastSquares(Model):
    """
    Ridge Regression model solved using the analytical Least Squares (Normal Equation) method.
    This model includes L2 regularization.

    Parameters
    ----------
    l2_penalty : float
        The L2 regularization parameter (lambda).
    scale : bool
        Whether to scale the data (mean normalization and standard deviation scaling)
        before fitting.

    Attributes
    ----------
    theta : np.ndarray
        The model's feature coefficients.
    theta_zero : float
        The model's intercept term.
    mean : np.ndarray
        The mean of the features from the training data (if scaled).
    std : np.ndarray
        The standard deviation of the features from the training data (if scaled).
    """
    def __init__(self, l2_penalty: float = 1.0, scale: bool = True):
        super().__init__()
        self.l2_penalty = l2_penalty
        self.scale = scale
        
        # Estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fits the Ridge Regression model using the Normal Equation.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.
        """
        X = dataset.X
        y = dataset.y
        n_samples, n_features = X.shape

        # Scale the data
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            # Avoid division by zero if std is 0
            self.std[self.std == 0] = 1 
            X = (X - self.mean) / self.std

        # Add intercept term (column of ones) to X
        # np.c_ adds the column of ones to the beginning
        X_intercept = np.c_[np.ones(n_samples), X]

        # Compute the penalty matrix (L2 * Identity)
        # Size is (n_features + 1) to account for the intercept
        identity_matrix = np.eye(n_features + 1)
        penalty_matrix = self.l2_penalty * identity_matrix

        # Change the first position to 0 (don't penalize theta_zero)
        penalty_matrix[0, 0] = 0

        # Compute model parameters using the Normal Equation
        # Formula: thetas = inv(X^T . X + penalty_matrix) . (X^T . y)
        XTX = X_intercept.T @ X_intercept
        XTy = X_intercept.T @ y
        
        thetas = np.linalg.inv(XTX + penalty_matrix) @ XTy
        
        # Store the intercept (theta_zero) and feature coefficients (theta)
        self.theta_zero = thetas[0]
        self.theta = thetas[1:]
        
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts target values for the given dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to make predictions on.
        """
        X = dataset.X
        n_samples = X.shape[0]

        # Scale the data using the *stored* mean and std
        if self.scale:
            X = (X - self.mean) / self.std

        # Add intercept term
        X_intercept = np.c_[np.ones(n_samples), X]
        
        # Concatenate thetas
        # np.r_ combines theta_zero and theta back into one vector
        thetas = np.r_[self.theta_zero, self.theta]

        # Compute predicted Y
        # Formula: y_pred = X_intercept . thetas
        predictions = X_intercept @ thetas

        return predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the MSE score for the given dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to score.
        predictions : np.ndarray
            The predictions made by the public predict() method.

        Returns
        -------
        float
            The MSE score.
        """
        y_true = dataset.y
        # Compute the mse score
        return mse(y_true, predictions)