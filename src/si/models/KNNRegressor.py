# 7.2) Add the "KNNRegressor" class to the "models" sub-package. You should create a module named "knn_regressor.py" to implement this class. Consider the structure of the "KNNRegressor" as presented in the next slides

import numpy as np
from si.data.dataset import Dataset
from si.base.model import Model
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance

class KNNRegressor(Model):
    """
    K-Nearest Neighbors Regressor.
    A model that predicts the value of a sample based on the average value
    of its k nearest neighbors in the training data.

    Parameters
    ----------
    k : int
        The number of nearest neighbors to consider.
    distance : callable
        The function to calculate the distance between samples.
        Defaults to euclidean_distance.

    Attributes
    ----------
    dataset : Dataset
        The training dataset, stored during the fit method.
    """
    def __init__(self, k: int = 5, distance: callable = euclidean_distance):
        super().__init__()
        self.k = k
        self.distance = distance
        self.dataset = None

    def _fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Stores the training dataset.

        Parameters
        ----------
        dataset : Dataset
            The training dataset.
        """
        self.dataset = dataset
        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the target values for the given dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset (test set) to make predictions on.

        Returns
        -------
        np.ndarray
            An array of predicted values.
        """
        # Get the features from the test dataset
        X_test = dataset.X
        
        predictions = []
        # Iterate over each sample in the test set
        for sample in X_test:
            # Calculate distances from the current sample to all training samples
            distances = self.distance(sample, self.dataset.X)
            
            # Get the indices of the k nearest neighbors
            # np.argsort returns indices that would sort the array
            k_nearest_indices = np.argsort(distances)[:self.k]
            
            # Get the corresponding y values (labels) from the training set
            k_nearest_values = self.dataset.y[k_nearest_indices]
            
            # Calculate the average of the k nearest values
            prediction = np.mean(k_nearest_values)
            
            predictions.append(prediction)
        
        return np.array(predictions)

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Calculates the RMSE score for the given dataset using the provided predictions.

        Parameters
        ----------
        dataset : Dataset
            The dataset (test set) to score.
        predictions : np.ndarray
            The predictions made by the public predict() method.

        Returns
        -------
        float
            The RMSE score.
        """
        
        # Get true values
        y_true = dataset.y
        
        # Calculate RMSE
        return rmse(y_true, predictions)