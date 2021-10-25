import pytest

import numpy as np

from classification_forest import Forest
from classification_forest_builder import build_forest
from validator import is_valid


def test_forest_perfect_classifier():
    N = 100
    X0 = np.full(shape=(N, 1), fill_value=5)
    X1 = np.full(shape=(N, 1), fill_value=-10)
    X = np.hstack([X0, X1])
    y = np.full(shape=(N,), fill_value=1)
    forest = Forest(X=X, y=y, min_samples_per_node=1, num_trees=10).train()
    predictions = forest.predict(X)
    assert sum(predictions != y) == 0
    is_valid(forest)


def test_forest_random_data():
    N = 5000
    X = np.random.uniform(-20, 20, size=(N, 20))
    y = np.random.choice([0, 1], size=(N,))
    forest = Forest(X=X, y=y, min_samples_per_node=100, num_trees=3).train()
    is_valid(forest)


def test_wrapper_functions():
    N = 100
    X = np.random.uniform(-20, 20, size=(N, 5))
    y = np.random.choice([0, 1], size=(N,))
    trained_forest = build_forest(X=X, y=y, num_trees=5, num_rand_vars=4, min_samples_per_node=20)
    assert isinstance(trained_forest, Forest)
    is_valid(trained_forest)
