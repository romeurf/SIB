import numpy as np
from si.base.model import Model
from si.base.transformer import Transformer
from si.data.dataset import Dataset


class KMeans(Transformer, Model):

    def __init__(self, k, max_iter, distance):
        
        self.k = k
        self.max_iter = max_iter
        self.distance = distance

        self.centroids = None
        self.labels = None

    def _init_centroids(self, dataset: Dataset):
        
        random_indexes = np.random.permutation(dataset.shape()[0])[:self.k]
        self.centroids = dataset.X[random_indexes, :]

    def calculate_distances(self, sample):
        
        return self.distance(sample, self.centroids)
    
    def _get_closest_centroids(self, sample):
        centroid_distance = self.calculate_distances(sample)
        centroids_index = np.argmin(centroid_distance, axis=0)
        return centroids_index


    def _fit(self, dataset):
        
        self._init_centroids(dataset)

        new_labels = np.apply_along_axis(self._get_closest_centroids,
                                         arr=dataset.X,
                                         axis=1)
        self.labels = new_labels
        
        centroids = []
        for j in range (self.k):
            mask = new_labels == j
            new_centroid = np.mean(dataset.X[mask])
            centroids.append(new_centroid)

        centroids = np.array(centroids)
        convergence = not np.any(new_labels != self.labels)

        labels = new_labels
        i+=1

        self.labels = labels
        return self

    def _transform(self, dataset):
        euclidian_distances = np.apply_along_axis(self.calculate_distances,
                                                  arr=dataset.X,
                                                  axis=1)
        return euclidian_distances

    def _predict(self, dataset):
        new_labels = np.apply_along_axis(self._get_closest_centroids,
                                         arr=dataset.X,
                                         axis=1)
        return new_labels