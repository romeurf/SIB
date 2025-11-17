# 3.1) Add the SelectPercentile object to the feature_selection sub-package. You should create a module called "select_percentile.py" to implement this object. The SelectPercentile class has a similar architecture to the SelectKBest class. Consider the structure presented in the next slide.

import numpy as np
from si.data.dataset import Dataset
from si.base.transformer import Transformer
from si.statistics.f_classification import f_classification 

class SelectPercentile(Transformer):
    """
    Select features according to a percentile of the highest scores.
    Inherits from Transformer.

    Parameters
    ----------
    score_func : callable
        Function taking a Dataset and returning (scores, pvalues).
        Defaults to f_classification.
    percentile : int
        Percentile of features to select (0 < percentile <= 100).
        Defaults to 10.
    """
    def __init__(self, score_func: callable = f_classification, percentile: int = 10):
        super().__init__()  # Initialize the base Transformer class
        if not (0 < percentile <= 100):
            raise ValueError(
                f"percentile should be 0 < percentile <= 100. Got {percentile}"
            )
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None
        self._support_mask = None

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Estimates the F and p values for each feature using the score_func.
        
        Parameters
        ----------
        dataset : Dataset
            The Dataset object to fit.
        """
        # This now works: score_func (f_classification) takes the dataset directly
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Selects features based on the specified percentile of F-values.
        
        Parameters
        ----------
        dataset : Dataset
            The Dataset object to transform.
        
        Returns
        -------
        Dataset
            A new Dataset object with only the selected features.
        """
        if self.F is None:
            raise ValueError("_fit() must be called before _transform()")
            
        X = dataset.X
        n_features = X.shape[1]
        
        # Calculate the number of features to select
        n_selected = int(np.ceil(n_features * self.percentile / 100))

        # --- Your excellent logic for handling ties ---
        # Sort features by F score (descending)
        sorted_indices = np.argsort(-self.F)
        
        # Find the threshold F-value (the score of the k-th feature)
        threshold = self.F[sorted_indices[n_selected - 1]]
        
        # Start with features strictly greater than the threshold
        mask = self.F > threshold
        n_above = np.sum(mask)
        
        # Add features that are tied at the threshold, if needed
        if n_above < n_selected:
            # Find all indices with F-value equal to the threshold
            tied_indices = [i for i in sorted_indices if self.F[i] == threshold]
            
            # Add the first (n_selected - n_above) tied features
            add_indices = tied_indices[:n_selected - n_above]
            for idx in add_indices:
                mask[idx] = True

        self._support_mask = mask
        
        # Filter X and feature names
        X_transformed = X[:, self._support_mask]
        
        # Use np.array for boolean indexing
        features_array = np.array(dataset.features)
        features_transformed = list(features_array[self._support_mask])
        
        # Return a new, transformed Dataset object
        return Dataset(X_transformed, dataset.y, features=features_transformed, label=dataset.label)

    def get_support(self, indices: bool = False):
        """
        Gets a boolean mask or integer indices of the selected features.

        Parameters
        ----------
        indices : bool
            If True, return the indices of the selected features.
            If False, return a boolean mask.

        Returns
        -------
        np.ndarray
            Boolean mask or array of indices.
        """
        if self._support_mask is None:
            raise ValueError("fit() must be called before get_support()")
        
        if indices:
            return np.where(self._support_mask)[0]
        
        return self._support_mask