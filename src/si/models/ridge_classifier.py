import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset

class RidgeClassifier(Model):
    
    def __init__(self, l2_term: float, alpha: float, max_iter: int,
                 patience: int, scale: bool = True, **kwargs):
        
        self.l2_term = l2_term
        self.alpha = alpha 
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = None

    def _fit(self, dataset: Dataset) -> "RidgeClassifier":
        
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = dataset.X - self.mean / self.std
        else:
            X = dataset.X

        m, n = dataset.shape()

        self.theta = np.zeros(n)
        self.theta_zero = 0

        i = 0
        early_stopping = 0
        while i < self.max_iter and early_stopping < self.patience:

            y_pred = np.dot(X, self.theta) + self.theta_zero

            gradient = (self.alpha / m) * np.dot(y_pred - dataset.y, X)
            penalization_term = self.theta * (1 - self.alpha *self.l2_term / m)

            self.theta = penalization_term - gradient
            self.theta_zero = self.theta_zero - (self.alpha /m) * np.sum(y_pred - dataset.y)

            self.cost_history[i] = self.cost(dataset.y, y_pred)

            if i > 0 and self.cost_history[i] > self.cost_history[i - 1]:
                early_stopping += 1
            
            i+=1
            
        return self
    
    def cost(self, y_true, y_pred) -> float:

        m = len(y_true)
        return (1 / 2*m) * np.sum(y_pred - y_true) ** 2 + (self.l2_term *np.sum(self.theta ** 2))