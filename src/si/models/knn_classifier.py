import numpy as np
from si.base.model import Model
from si.metrics import accuracy

class KNN(Model):

    def __init__(self, k: int, distance_func: callable, **kwargs):
        self.k = k
        self.distance_func = distance_func

        self.dataset = None

    def _fit(self, dataset) -> "KNN":
        
        self.dataset = dataset

    def get_closest_neighbors(self, sample: np.ndarray):

        distance_to_all_points = self.distance_func(sample, self.dataset.X)
        indexes_of_nn = np.argsort(distance_to_all_points)[:self.k]
        nn_labels = self.dataset.y[indexes_of_nn]
        unique_labels, counts = np.unique(nn_labels, return_counts=True)
        label = unique_labels[np.argmax(counts)]
        return label
    
    def _predict(self, dataset) -> np.ndarray:

        return np.apply_along_axis(self.get_closest_neighbors, axis=1, arr=dataset.X)

    def _score(self, dataset):

        return accuracy(dataset.y, self.predict(dataset))