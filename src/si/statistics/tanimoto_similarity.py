# Exercise 4: Implement the Tanimoto Similarity in file tanimoto_similarity.py (in the sub-package statistics)

import numpy as np

def tanimoto_similarity(x, y):
    """
    Computes the Tanimoto similarity between a single binary sample x and multiple binary samples y.

    Args:
        x (array-like): 1D binary array or list (sample).
        y (array-like): 2D binary array or list of lists (samples, each row a sample).

    Returns:
        similarities (np.ndarray): Array of Tanimoto similarities between x and each sample in y.
    """
    x = np.asarray(x).astype(bool) # Ensure x is a boolean array
    y = np.asarray(y).astype(bool) # Ensure y is a boolean array

    # Intersection counts for each sample
    intersection = np.sum(y & x, axis=1) # axis=1 to sum over features for each sample
    # Union counts for each sample
    union = np.sum(y | x, axis=1) # axis=1 to sum over features for each sample
    # Handle potential zero union (to avoid division by zero)
    similarities = np.zeros_like(intersection, dtype=float)  #np.zeros to initialize similarities
    valid = union > 0 # Avoid division by zero
    similarities[valid] = intersection[valid] / union[valid]

    return similarities