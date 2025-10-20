import numpy as np

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    return (y_true == y_pred).sum() / len(y_true)