from typing import Callable, Optional, Tuple, List

import numpy as np
from scipy.stats import mode

from classification_tree import Tree
from impurity_functions import gini_index


class Forest:
    """
    A classification forest. This can be trained with input data and then used to make predictions.

    Attributes
    ----------
    X : np.ndarray
        training data array of shape (N, k)
    y : np.ndarray
        training labels array of shape (N,)
    min_samples_per_node : Optional[int], optional
        each node's region in all trees must have at least this many samples
    impurity_func : Optional[Callable[np.ndarray]], optional
        a func which takes np.array([label_count_0, label_count_1]) and returns a float
    num_trees : Optional[int], optional
        num of trees which comprise the forest
    num_rand_vars : Optional[int], optional
        num of random vars to select in each node split
    trees : List[Tree]
        a list of the trees which comprise the forest
    random_seed : Optional[int], optional
        random seed
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        min_samples_per_node: Optional[int] = 10,
        impurity_func: Optional[Callable[[np.ndarray], float]] = gini_index,
        num_trees: Optional[int] = 500,
        num_rand_vars: Optional[int] = 5,
        random_seed: Optional[int] = 99
    ):
        self.X = X
        self.y = y
        self.min_samples_per_node = min_samples_per_node
        self.impurity_func = impurity_func
        self.num_trees = num_trees
        self.num_rand_vars = num_rand_vars
        self.trees = []
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
    
    def _train_tree(self) -> Tree:
        """
        Gerneates a bootstrap sample, trains an tree on that sample and returns it.
        """
        num_samples, _ = self.X.shape
        rand_indices = np.random.choice(np.arange(0, num_samples), num_samples, replace=True)
        X_i = self.X[rand_indices]
        y_i = self.y[rand_indices]
        tree = Tree(
            X=X_i, 
            y=y_i, 
            min_samples_per_node=self.min_samples_per_node, 
            impurity_func=self.impurity_func, 
            var_split_type='random', 
            num_rand_vars=self.num_rand_vars,
            random_seed=self.random_seed
        ).train()
        return tree
    
    def train(self):
        """
        Generates a list of trained trees of size num_trees.
        """
        trees = [self._train_tree() for _ in np.arange(0, self.num_trees)]
        self.trees = trees
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Takes a data matrix of shape (N, k) and returns predictions for each of the N points
        by taking a majority vote of the trees.
        """
        all_predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions = mode(all_predictions, axis=0)
        return predictions[0].reshape(-1)
