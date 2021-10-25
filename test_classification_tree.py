import pytest

import numpy as np
import psycopg2


from classification_tree import Tree, PrunedTree, MatrixNode
from classification_tree_builder import build_tree, build_cv_tree
import impurity_functions
from validator import is_valid


def test_tree_no_splits():
    N = 1000
    X0 = np.full(shape=(N, 1), fill_value=5)
    X1 = np.full(shape=(N, 1), fill_value=-10)
    X = np.hstack([X0, X1])
    y = np.full(shape=(N,), fill_value=1)
    tree = Tree(X=X, y=y).train()

    assert isinstance(tree, Tree)
    assert tree.root is not None
    assert tree.root.left is None
    assert tree.root.right is None
    assert tree.root.split_point is None
    assert tree.root.split_var is None
    assert np.all(tree.root.remaining_vars == [0, 1])
    assert tree.root.leaf == True
    assert tree.root.classification == 1
    assert tree.root.node_idx == 1
    assert tree.root.depth == 1
    assert np.all(tree.root.region == list(np.arange(0, N)))
    is_valid(tree)


def test_tree_perfect_classifier():
    N = 1000
    X0 = np.random.uniform(-1, 1, size=(N, 1))
    X1 = np.random.uniform(-1, 1, size=(N, 1))
    X = np.hstack([X0, X1])
    y = ((X0 >= 0) & (X1 < -0.5)).astype(int).reshape(-1).T
    tree = Tree(X=X, y=y, min_samples_per_node=1).train()
    prediction = tree._predict_point(tree.root, np.array([0.1, -0.6]))
    predictions = tree.predict(X)
    assert prediction == 1
    assert sum(predictions != y) == 0
    is_valid(tree)


def test_tree_random_data():
    N = 5000
    X = np.random.uniform(-20, 20, size=(N, 20))
    y = np.random.choice([0, 1], size=(N,))
    tree = Tree(X=X, y=y, min_samples_per_node=100).train()
    is_valid(tree)


def test_matrixnode_util_functions():
    N = 1000
    X = np.random.uniform(-1, 1, size=(N, 20))
    y = np.random.choice([0, 1], size=(N,))
    test_node = MatrixNode(X=X, y=y, depth=10, remaining_vars=[], region=[], node_idx=0)

    # test label count
    arr = np.array([0, 0, 1, 0, 1, 0, 1, 0])
    label_counts = test_node._label_count(arr)
    assert len(label_counts) == 2
    assert label_counts[0] == 5
    assert label_counts[1] == 3
    
    arr = np.array([0, 0, 0])
    label_counts = test_node._label_count(arr)
    assert len(label_counts) == 2
    assert label_counts[0] == 3
    assert label_counts[1] == 0

    # test select vars
    all_remaining_vars = [0, 20, 9, 88, 1000, 3, 10, 65, 3, 2, 67, 90]
    rand_vars = test_node._select_vars(all_remaining_vars, 'random', 5)
    best_vars = test_node._select_vars(all_remaining_vars, 'best', 5)
    assert len(rand_vars) == 5
    assert np.all([rv in set(all_remaining_vars) for rv in rand_vars])
    assert np.all(best_vars == all_remaining_vars)


def test_pruned_tree():
    N = 500
    X = np.random.uniform(-20, 20, size=(N, 20))
    y = np.random.choice([0, 1], size=(N,))
    tree = Tree(X=X, y=y, min_samples_per_node=10).train()
    pruned_small = PrunedTree(orig_tree = tree, alpha=0)
    pruned_large = PrunedTree(orig_tree = tree, alpha=1000000000000)
    pruned_opt_small = pruned_small.train()
    pruned_opt_large = pruned_large.train()
    assert isinstance(pruned_opt_small, Tree)
    assert isinstance(pruned_opt_large, Tree)
    assert pruned_opt_large.root.num_leaves <= pruned_opt_small.root.num_leaves
    assert pruned_opt_large.root.num_leaves <= pruned_large.pruned_trees[0].root.num_leaves
    assert pruned_opt_small.root.num_leaves <= pruned_small.pruned_trees[0].root.num_leaves
    is_valid(pruned_opt_small)
    is_valid(pruned_opt_large)


def test_wrapper_functions():
    N = 5000
    X = np.random.uniform(-20, 20, size=(N, 5))
    y = np.random.choice([0, 1], size=(N,))
    trained_tree = build_tree(X=X, y=y)
    pruned_tree = build_tree(X=X, y=y, alpha=0.0000001)
    cv_trained_tree = build_cv_tree(X=X, y=y)
    assert isinstance(trained_tree, Tree)
    assert isinstance(pruned_tree, Tree)
    assert isinstance(cv_trained_tree, Tree)
    is_valid(trained_tree)
    is_valid(pruned_tree)       
    is_valid(cv_trained_tree)


def test_impurity_functions():
    perfect_label_counts = np.array([100000, 0])
    assert impurity_functions.bayes_error(perfect_label_counts) == 0
    assert impurity_functions.cross_entropy(perfect_label_counts) == 0
    assert impurity_functions.gini_index(perfect_label_counts) == 0
    
    label_counts = np.array([100, 400])
    assert impurity_functions.bayes_error(label_counts) == pytest.approx(0.2, 0.00001)
    assert impurity_functions.cross_entropy(label_counts) == pytest.approx(0.5004, 0.00001)
    assert impurity_functions.gini_index(label_counts) == pytest.approx(0.16, 0.00001)
