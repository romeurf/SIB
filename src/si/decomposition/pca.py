import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class PCA(Transformer):
    """
    Performs Principal Component Analysis (PCA) on a dataset.
    It is a Transformer, so it fits and transforms a Dataset object.

    Parameters
    ----------
    n_components : int
        The number of principal components to keep.

    Attributes
    ----------
    mean : np.ndarray
        The mean of the features from the training data.
    components : np.ndarray
        The principal components (eigenvectors). Shape (n_components, n_features)
    explained_variance : np.ndarray
        The variance explained by each principal component. Shape (n_components,)
    """
    def __init__(self, n_components: int = None):
        super().__init__()
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset: Dataset) -> 'PCA':
        """
        Fits PCA to the dataset.
        Estimates the mean, components, and explained variance.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object to fit.
        """
        X = dataset.X
        
        # Step 1: Center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Step 2: Covariance and eigenvalue decomposition
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Get the sum of all eigenvalues before slicing
        total_variance = np.sum(eigenvalues)

        # Step 3: Sort by eigenvalues descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]  # Sort columns

        # Step 4: Select top n_components
        if self.n_components is not None:
            eigenvalues = eigenvalues[:self.n_components]
            eigenvectors = eigenvectors[:, :self.n_components]
        
        # Store components (eigenvectors) as rows
        self.components = eigenvectors.T  # Shape (n_components, n_features)

        # Store explained variance (using the correct total)
        self.explained_variance = eigenvalues / total_variance
        
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset by projecting it onto the principal components.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object to transform.

        Returns
        -------
        Dataset
            A new Dataset object with the reduced features.
        """
        X = dataset.X
        
        # Step 1: Center data using the stored mean
        X_centered = X - self.mean
        
        # Step 2: Project onto components
        # (n_samples, n_features) @ (n_features, n_components) -> (n_samples, n_components)
        X_reduced = np.dot(X_centered, self.components.T)
        
        # Create new feature names
        new_features = [f"PC{i+1}" for i in range(self.components.shape[0])]

        # Return a new, transformed Dataset object
        return Dataset(X=X_reduced, 
                       y=dataset.y, 
                       features=new_features, 
                       label=dataset.label)