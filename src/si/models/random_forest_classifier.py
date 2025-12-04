# 9.1) In the "models" sub-package add the "random_forest_classifier.py" module. Implement the RandomForestClassifier class by taking into consideration the structure of the class shown in the next slides.

import numpy as np
from scipy import stats
from si.data.dataset import Dataset
from si.base.model import Model
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier(Model):
    """
    Random Forest Classifier.
    An ensemble model that fits multiple Decision Trees on bootstrapped samples
    and feature subsets.
    """
    def __init__(self, 
                 n_estimators: int = 100, 
                 max_features: int = None, 
                 min_samples_split: int = 2, 
                 max_depth: int = 10, 
                 mode: str = 'gini', # gini is the default impurity
                 seed: int = None):
        """
        Parameters
        ----------
        n_estimators : int
            The number of trees in the forest.
        max_features : int
            The number of features to consider when looking for the best split.
            If None, then max_features = sqrt(n_features).
        min_samples_split : int
            The minimum number of samples required to split an internal node.
        max_depth : int
            The maximum depth of the tree.
        mode : str
            The impurity function to use ('gini' or 'entropy').
        seed : int
            Random seed for reproducibility.
        """
        super().__init__()
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        
        # Estimated parameters
        # This will store tuples of (tree, feature_indices)
        self.trees = []

    def _fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Trains all the decision trees in the forest.
        """
        # Set random seed
        if self.seed is not None:
            np.random.seed(self.seed)
            
        n_samples, n_features = dataset.shape()
        
        # Set max_features
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        
        self.trees = []
        for _ in range(self.n_estimators):
            
            # Create bootstrap dataset
            # Bootstrap samples (with replacement)
            sample_indices = np.random.choice(n_samples, n_samples, replace=True) #replace=True for bootstrap
            
            # Bootstrap features (without replacement)
            feature_indices = np.random.choice(n_features, self.max_features, replace=False) #replace=False to avoid duplicate features
            
            # Create the bootstrapped dataset
            X_boot = dataset.X[sample_indices][:, feature_indices]
            y_boot = dataset.y[sample_indices]
            # Get feature names, as required by DecisionTreeClassifier
            boot_features = [dataset.features[i] for i in feature_indices] 
            
            boot_dataset = Dataset(X_boot, y_boot, features=boot_features, label=dataset.label)
            
            # Create and train a decision tree
            # Pass the parameters to the tree
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split,
                                          max_depth=self.max_depth,
                                          mode=self.mode)
            tree.fit(boot_dataset)
            
            # Append the tree and the features it used
            self.trees.append((tree, feature_indices))

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class for each sample using majority voting.
        """
        X_test = dataset.X
        
        # Store predictions from all trees: (n_samples, n_estimators)
        all_predictions = []

        # Get predictions for each tree
        for tree, feature_indices in self.trees:
            # Get the test data only for the features this tree was trained on
            X_test_subset = X_test[:, feature_indices]
            
            # Create a temporary dataset to pass to the tree's predict method
            sub_features = [dataset.features[i] for i in feature_indices]
            sub_dataset = Dataset(X_test_subset, 
                                  y=dataset.y, # y is not used in predict, but required
                                  features=sub_features, 
                                  label=dataset.label)

            tree_preds = tree.predict(sub_dataset)
            all_predictions.append(tree_preds)
        
        # Transpose to: (n_samples, n_estimators)
        stacked_preds = np.array(all_predictions).T
        
        # Get the most common predicted class (majority vote)
        # stats.mode finds the mode along axis=1 (for each sample)
        final_predictions = stats.mode(stacked_preds, axis=1, keepdims=False)[0] # keepdims=False to return 1D array; [0] to get the mode values
        
        return final_predictions

    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Computes accuracy for the model.
        This matches the signature from your model.py base class.
        """
        y_true = dataset.y
        return accuracy(y_true, predictions)