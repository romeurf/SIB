# 3.1) Add the SelectPercentile object to the feature_selection sub-package. You should create a module called "select_percentile.py" to implement this object. The SelectPercentile class has a similar architecture to the SelectKBest class. Consider the structure presented in the next slide.

import numpy as np

class SelectPercentile:
    """
    Select features according to a percentile of the highest scores.
    """
    def __init__(self, score_func=None, percentile=10):
        if not (0 < percentile <= 100):
            raise ValueError(
                f"percentile should be 0 < percentile <= 100. Got {percentile}"
            )
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None
        self._support_mask = None

    def _fit(self, X, y):
        if self.score_func is None:
            raise ValueError("score_func cannot be None")
        X = np.asarray(X)
        y = np.asarray(y)
        score_result = self.score_func(X, y)
        # Handle both (scores, pvalues) and just scores return formats
        if isinstance(score_result, tuple) and len(score_result) == 2:
            self.F = np.asarray(score_result[0])
            self.p = np.asarray(score_result[1])
        else:
            self.F = np.asarray(score_result)
            self.p = None
        return self

    def _transform(self, X):
        if self.F is None:
            raise ValueError("_fit() must be called before _transform()")
        X = np.asarray(X)
        n_features = X.shape[1]
        n_selected = int(np.ceil(n_features * self.percentile / 100))
        # Sort features by F score (descending)
        sorted_indices = np.argsort(-self.F)
        threshold = self.F[sorted_indices[n_selected - 1]]
        # Features greater than threshold
        mask = self.F > threshold
        n_above = np.sum(mask)
        if n_above < n_selected:
            # Find tied threshold indices in sorting order
            tied_indices = [i for i in sorted_indices if self.F[i] == threshold]
            # Add first required number of tied features
            add_indices = tied_indices[:n_selected - n_above]
            for idx in add_indices:
                mask[idx] = True
        self._support_mask = mask
        return X[:, mask]

    def fit(self, X, y):
        return self._fit(X, y)

    def transform(self, X):
        return self._transform(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        if self._support_mask is None:
            raise ValueError("fit() must be called before get_support()")
        if indices:
            return np.where(self._support_mask)[0]
        return self._support_mask