from collections import deque
from typing import Callable, Union, List

import numpy as np

from classification_tree import Tree, Node
from classification_forest import Forest


def is_valid(classifier: Union[Tree, Forest]):
    """
    Runs a series of checks on a Tree or Forest.
    """
    if isinstance(classifier, Tree):
        assert(obeys_min_samples_per_node(classifier) == True), 'there are non-empty nodes!'
        assert(leaves_equal_full_sample(classifier) == True), 'the samples in your leaves dont add up to the full sample!'
        assert(splits_points_vs_children(classifier) == True), 'some child node in your tree is not obeying the split point of the parent!'
        assert(classification_is_0_or_1(classifier) == True), 'tree generates classifications other than 0 or 1!'
        assert(leaves_have_no_children(classifier) == True), 'tree has leaves with child nodes!'
    elif isinstance(classifier, Forest):
        for tree in classifier.trees:
            is_valid(tree)


def tree_traversal_for_node_check(tree: Tree, node_check: Callable) -> List:
    """
    A utility function to traverse a tree and call a function node_check on each node.
    Returns a list of the outputs returned by node_check.
    """
    results = []
    queue = deque()
    queue.append(tree.root)
    while queue:
        curr = queue.popleft()
        results.append(node_check(curr))
        if curr.left:
            queue.append(curr.left)
        if curr.right:
            queue.append(curr.right)
    return results


def obeys_min_samples_per_node(tree: Tree) -> bool:
    """
    Checks that each node has more than min_samples_per_node.
    """
    node_check = lambda node: len(node.region) >= tree.min_samples_per_node
    results = tree_traversal_for_node_check(tree, node_check)
    return np.all(results)


def classification_is_0_or_1(tree: Tree)  -> bool:
    """
    Checks that each node's classification is either 0 or 1.
    """
    node_check = lambda node: node.classification in [0, 1]
    results = tree_traversal_for_node_check(tree, node_check)
    return np.all(results)


def leaves_have_no_children(tree: Tree)  -> bool:
    """
    Checks that leaf nodes have no children.
    """
    def _node_check(node):
        if node.leaf:
            return node.right is None and node.left is None
        else:
            return True
    results = tree_traversal_for_node_check(tree, _node_check)
    return np.all(results)


def leaves_equal_full_sample(tree: Tree) -> bool:
    """
    Checks that the region samples from the leaf nodes combined equals the full sample.
    """
    def _node_output(node):
        if node.leaf:
            return node.region

    leave_samples = tree_traversal_for_node_check(tree, _node_output)
    leave_samples = np.hstack(leave_samples)
    leave_samples = leave_samples[leave_samples != np.array(None)]
    leave_samples = np.sort(leave_samples)
    full_sample = np.sort(tree.root.region)
    return np.all(leave_samples == full_sample)


def splits_points_vs_children(tree: Tree) -> bool:
    """
    Checks that the samples in the children of a node obey the split point.
    """
    def _node_output(node):
        if node.leaf:
            return True
        split_point = node.split_point
        left_samples = tree.X[node.left.region, node.split_var]
        right_samples = tree.X[node.right.region, node.split_var]
        return np.all(left_samples <= split_point) and np.all(right_samples > split_point)
    results = tree_traversal_for_node_check(tree, _node_output)
    return np.all(results)
