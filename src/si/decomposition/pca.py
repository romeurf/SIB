# 5.1) Add the "PCA" class in the "pca.py" module on the "decomposition" sub-package. Consider the PCA class structure presented in the next slide.

import numpy as np
from si.base.Transformer import Transformer  # Update if necessary

class PCA(Transformer):
    def __init__(self, n_components=None):
        super().__init__()
        self.n_components = n_components
        self.mean = None
        self.components = None  # shape: (n_components, n_features)
        self.explained_variance = None  # shape: (n_components,)

    def _fit(self, X):
        # Step 1: Center data
        X = np.asarray(X)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Step 2: Covariance and eigenvalue decomposition
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)  # Note np.linalg.eig (not eigh)

        # Step 3: Sort by eigenvalues descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Step 4: Select top n_components
        if self.n_components is not None:
            eigenvalues = eigenvalues[:self.n_components]
            eigenvectors = eigenvectors[:, :self.n_components]

        # Principal components: eigenvectors
        self.components = eigenvectors.T  # shape (n_components, n_features)

        # Explained variance
        self.explained_variance = eigenvalues / np.sum(eigenvalues)
        return self

    def _transform(self, X):
        # Step 1: Center data
        X = np.asarray(X)
        X_centered = X - self.mean
        # Step 2: Project onto components ("X_reduced = X * V")
        # Each component is a row in self.components
        return np.dot(X_centered, self.components.T)  # shape (n_samples, n_components)