from typing import Callable, Optional, Union

import numpy as np
from classification_tree import Tree, PrunedTree
from impurity_functions import gini_index


def build_tree(
    X: np.ndarray,
    y: np.ndarray,
    impurity_func: Optional[Callable] = gini_index,
    min_samples_per_node: Optional[int] = 5,
    alpha: Optional[float] = None,
    random_seed: Optional[int] = 99
) -> Tree:
    """
    Returns a trained classification tree. Tree is pruned if alpha is specified.
    """
    tree = Tree(X, y, min_samples_per_node, impurity_func, random_seed=random_seed).train()
    if alpha:
        tree = PrunedTree(tree, alpha).train()
    return tree


def build_cv_tree(
    X: np.ndarray,
    y: np.ndarray,
    impurity_func: Optional[Callable] = gini_index,
    min_samples_per_node: Optional[int] = 5,
    alpha_vals: Optional[list] = list(np.geomspace(0.01, 10, 10)),
    n_folds: Optional[int] = 5,
    random_seed: Optional[int] = 99
) -> Tree:
    """
    Returns a trained classification tree where the alpha parameter is determined by 
    kfold cross validation over n_folds.
    """
    np.random.seed(random_seed)
    n_samples, _ = X.shape
    indices = np.arange(0, n_samples)
    np.random.shuffle(indices)
    fold_indices = np.split(indices, n_folds)

    alpha_errors = [[] for _ in range(len(alpha_vals))]

    for fi in fold_indices:
        X_train, y_train = X[fi, :], y[fi]
        X_test, y_test = np.delete(X, fi, axis=0), np.delete(y, fi)
        tree = Tree(X_train, y_train, min_samples_per_node, impurity_func).train()

        for i, alpha in enumerate(alpha_vals):
            pruned_tree = PrunedTree(tree, alpha).train()
            test_error = sum(pruned_tree.predict(X_test) != y_test) / len(y_test)
            alpha_errors[i].append(test_error)
    
    opt_alpha = alpha_vals[np.argmax(np.mean(alpha_errors, axis=1))]
    full_tree = Tree(X, y, min_samples_per_node, impurity_func).train()
    pruned_tree = PrunedTree(full_tree, opt_alpha).train()
    return pruned_tree
