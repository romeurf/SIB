import numpy as np
from si.data.dataset import Dataset


def train_test__split (dataset: Dataset, test_size: float, random_state: int = 42) -> tuple[Dataset, Dataset]:

    np.random.seed(random_state)
    n_samples = dataset.X.shape()[0]
    test_samples = int(n_samples * test_size)
    permutations = np.random.permutation(n_samples)
    test_indexes = permutations[:test_samples]
    train_indexes = permutations[test_samples:]

    train_dataset = Dataset(X=dataset.X[train_indexes], y=dataset.y[train_indexes],
                            features=dataset.features, label=dataset.label)
    
    test_dataset = Dataset(X=dataset.X[test_indexes], y=dataset.y[test_indexes],
                           features=dataset.features, label=dataset.label)
    
    return train_dataset, test_dataset