from abc import abstractmethod

import numpy as np


class LossFunction:

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the derivative of the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean squared error loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # To avoid the additional multiplication by -1 just swap the y_pred and y_true.
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Cross entropy loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / p) + (1 - y_true) / (1 - p)

#Exercise 14: Categorical Cross Entropy Loss

class CategoricalCrossEntropy(LossFunction):
    """
    Categorical cross-entropy loss function for multi-class problems
    with one-hot encoded targets and probability predictions.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # y_true: (n_samples, n_classes) one-hot
        # y_pred: (n_samples, n_classes) probabilities
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # sum over classes, then over samples
        return -np.sum(y_true * np.log(p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Gradient w.r.t. softmax input z when using softmax + cross-entropy:
        dL/dz = y_pred - y_true
        """
        return y_pred - y_true