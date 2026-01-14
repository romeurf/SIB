# Exercise 11: Implement a function for randomized hyperparameter search with cross-validation.

import numpy as np
from sklearn.linear_model import LogisticRegression

from si.model_selection.cross_validate import k_fold_cross_validation


def randomized_search_cv(model,
                         dataset,
                         hyperparameter_grid,
                         scoring,
                         cv,
                         n_iter):
    """
    Randomized search cross-validation.

    Parameters
    ----------
    model : object
        Model to validate. Must support setting hyperparameters as attributes
        and be compatible with k_fold_cross_validation.
    dataset : tuple or custom object
        Validation dataset. Typically (X, y), but adapt to your k_fold_cross_validation.
    hyperparameter_grid : dict
        Dictionary mapping hyperparameter names (str) to an iterable of
        candidate values. Example:
            {
                "l2_penalty": [1, 2, 3],
                "alpha": [0.001, 0.0005],
                "max_iter": [1000, 1500]
            }
    scoring : callable
        Function that receives (y_true, y_pred) and returns a scalar score.
        This is used internally by k_fold_cross_validation.
    cv : int
        Number of folds for cross-validation.
    n_iter : int
        Number of random hyperparameter combinations to test.

    Returns
    -------
    results : dict
        Dictionary with the following keys:
            - 'hyperparameters': list of dicts with the hyperparameters used.
            - 'scores': list of mean scores obtained for each combination.
            - 'best_hyperparameters': dict with the best hyperparameter combo.
            - 'best_score': best mean score.
    """

    # CHECK THAT ALL PROVIDED HYPERPARAMETERS EXIST ON THE MODEL

    for param_name in hyperparameter_grid.keys():
        if not hasattr(model, param_name):
            raise ValueError(
                f"Hyperparameter '{param_name}' does not exist in the model."
            )

    # PREPARE STORAGE FOR RESULTS

    all_hyperparams = []  # list of dicts with hyperparameters
    all_scores = []       # list of mean scores

    best_score = -np.inf
    best_hyperparams = None

    # SAMPLE n_iter HYPERPARAMETER COMBINATIONS
    #    USING np.random.choice OVER ALL POSSIBLE COMBINATIONS

    # Build a list of all combinations explicitly.
    
    param_names = list(hyperparameter_grid.keys())
    param_values = [hyperparameter_grid[name] for name in param_names]

    # Build cartesian product of all values
    all_combinations = []
    def build_combinations(idx, current):
        if idx == len(param_names):
            all_combinations.append(dict(current))
            return
        name = param_names[idx]
        for v in param_values[idx]:
            current[name] = v
            build_combinations(idx + 1, current)
    build_combinations(0, {})

    n_combinations = len(all_combinations)
    if n_iter > n_combinations:

        n_iter = n_combinations

    # Randomly choose n_iter unique indices from all combinations
    chosen_indices = np.random.choice(
        np.arange(n_combinations),
        size=n_iter,
        replace=False
    )

    # FOR EACH RANDOMLY CHOSEN COMBINATION:
    #       - SET PARAMS
    #       - CROSS-VALIDATE
    #       - STORE MEAN SCORE AND PARAMS

    for idx in chosen_indices:
        current_params = all_combinations[idx]

        # Set model hyperparameters with the current combination
        for name, value in current_params.items():
            setattr(model, name, value)

        # Cross validate the model using k_fold_cross_validation
        #    k_fold_cross_validation(model, dataset, scoring, cv) that returns a list of scores (one per fold).
        
        scores = k_fold_cross_validation(
            model=model,
            dataset=dataset,
            scoring=scoring,
            cv=cv
        )

        # Save the mean of the scores and respective hyperparameters
        mean_score = float(np.mean(scores))
        all_hyperparams.append(current_params.copy())
        all_scores.append(mean_score)

        # Update best score and best hyperparameters if needed
        if mean_score > best_score:
            best_score = mean_score
            best_hyperparams = current_params.copy()

    # BUILD FINAL OUTPUT DICTIONARY

    results = {
        "hyperparameters": all_hyperparams,
        "scores": all_scores,
        "best_hyperparameters": best_hyperparams,
        "best_score": best_score,
    }

    return results

# Exercise 11.1: Test the randomized_search_cv function

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV

data = pd.read_csv(r"datasets\breast_bin\breast-bin.csv", header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

model = SGDClassifier(loss="log_loss", penalty="l2")

param_dist = {
    "alpha": np.linspace(0.001, 0.0001, 100),
    "max_iter": np.linspace(1000, 2000, 200, dtype=int)
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring="accuracy",
    random_state=0
)

search.fit(X, y)

print("All mean CV scores:", search.cv_results_["mean_test_score"])
print("Best score:", search.best_score_)
print("Best params:", search.best_params_)