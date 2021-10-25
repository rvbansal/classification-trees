from typing import Callable, Optional, Union
import math

import numpy as np

from classification_forest import Forest
from impurity_functions import gini_index


def build_forest(
    X: np.ndarray,
    y: np.ndarray,
    impurity_func: Optional[Callable] = gini_index,
    min_samples_per_node: Optional[int] = 5,
    num_trees: Optional[int] = 500,
    num_rand_vars: Optional[int] = None,
    random_seed: Optional[int] = 99
) -> Forest:
    """
    Returns a trained random forest.
    """
    if not num_rand_vars:
        _, n_predictors = X.shape
        num_rand_vars = int(np.sqrt(n_predictors))
    forest = Forest(X, y, min_samples_per_node, impurity_func, num_trees, num_rand_vars, random_seed=random_seed).train()
    return forest