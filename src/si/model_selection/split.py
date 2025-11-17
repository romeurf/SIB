from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

# 6.1) In the "split.py" module of the "model_selection" subpackage add the "stratified_train_test_split" function (Consider the structure of the function presented in the next slide).

def stratified_train_test_split(dataset: Dataset, 
                                test_size: float = 0.2, 
                                random_state: int = None) -> Tuple[Dataset, Dataset]:
    """
    Splits a Dataset into stratified train and test sets.
    This ensures that the class proportions are preserved in both sets.

    Parameters
    ----------
    dataset : Dataset
        The Dataset object to split.
    test_size : float, optional
        The proportion of the dataset to include in the test split, by default 0.2
    random_state : int, optional
        Controls the shuffling applied to the data before splitting, by default None

    Returns
    -------
    Tuple[Dataset, Dataset]
        A tuple containing the (train_dataset, test_dataset).
    """
    
    # Set the random seed
    if random_state is not None:
        np.random.seed(random_state)
        
    # Get unique class labels and their counts
    classes, class_counts = np.unique(dataset.y, return_counts=True)
    
    # Initialize empty lists for train and test indices
    train_indices = []
    test_indices = []

    # Loop through unique labels
    for cls, count in zip(classes, class_counts):
        
        # Calculate the number of test samples for the current class
        # Use np.ceil to ensure at least 1 test sample if test_size is small
        n_test_samples = int(np.ceil(count * test_size))
        
        # Get all indices for the current class
        class_indices = np.where(dataset.y == cls)[0]
        
        # Shuffle and select indices for the test set
        np.random.shuffle(class_indices)
        current_test_indices = class_indices[:n_test_samples]
        
        # Add the remaining indices to the train set
        current_train_indices = class_indices[n_test_samples:]
        
        test_indices.extend(current_test_indices)
        train_indices.extend(current_train_indices)

    # After the loop, create training and testing datasets
    
    # Create training dataset
    train_X = dataset.X[train_indices]
    train_y = dataset.y[train_indices]
    train_dataset = Dataset(train_X, train_y, features=list(dataset.features), label=dataset.label)
    
    # Create testing dataset
    test_X = dataset.X[test_indices]
    test_y = dataset.y[test_indices]
    test_dataset = Dataset(test_X, test_y, features=list(dataset.features), label=dataset.label)

    # Return the training and testing datasets
    return train_dataset, test_dataset