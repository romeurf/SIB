import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum((y_true - y_pred) ** 2) / len(y_true)