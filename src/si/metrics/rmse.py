#7.1) Add the RMSE metric to the "metrics" sub-package. You must create a new module named "rmse.py". Consider the structure of the rmse function as presented in the following slide.
import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters
    ----------
    y_true : np.ndarray
        The array of true target values.
    y_pred : np.ndarray
        The array of predicted target values.

    Returns
    -------
    float
        The RMSE value.
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate the squared differences
    squared_errors = (y_true - y_pred) ** 2
    
    # Calculate the mean of the squared errors (MSE)
    mean_squared_error = np.mean(squared_errors)
    
    # Return the square root of the MSE
    return np.sqrt(mean_squared_error)