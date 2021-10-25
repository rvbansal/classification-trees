import copy
from collections import deque
import random
from typing import Callable, Optional, Tuple, List, Union
import warnings

import numpy as np
import psycopg2
from psycopg2.extensions import connection, AsIs

from impurity_functions import gini_index


class Node:
    """
    A node in the tree.

    Attributes
    ----------
    depth : int
        depth of the node in the tree (root is at depth 1)
    remaining_vars : Union[List[int], List[str]]
        variables over which the node optimizes the split; int for MatrixNode and str for SqlNode
    region : Union[List[int], List[str]]
        idenfitiers for the samples in the node's region; indices for MatrixNode and a list of 
        conditions of the form 'split_var < split_point' for SqlNode
    node_idx : int
        a unique integer identifier for the node        
    leaf : bool
        true if the node is a leaf, false otherwise
    split_var : Union[int, str]
        the variable over which the node splits; int for MatrixNode and str for SqlNode
    split_point : float
        the threshold for the split
    classification : int
        the majority vote of the region_samples (either 0 or 1)
    left : Node
        the left child
    right : Node
        the right child
    """
    def __init__(
        self,
        depth: int,
        remaining_vars: Union[List[int], List[str]], 
        region: Union[List[int], List[str]],
        node_idx: int,
    ):
        self.depth = depth
        self.remaining_vars = remaining_vars
        self.region = region
        self.node_idx = node_idx
        self.leaf = False
        self.split_var = None
        self.split_point = None
        self.classification = None
        self.left = None
        self.right = None

    def _split_node(
        self,
        var_split_type: str,
        num_rand_vars: int,
        min_samples_per_node: int,
        impurity_func: Union[Callable[[np.ndarray], float], str]
    ):
        """
        Splits a node and computes its classification. Then, determines the split_var and split_point
        by iterating over all remaining vars, searching the min to max range of each var and finding 
        the minimum impurity split. 
        
        A node is identified as a leaf if:
            - it has less than min_samples_per_node samples
            - is pure; all samples in the region are either 1 or 0
            - there are no remaining vars on which a split can be made
        
        Variables which are constant in the region are removed from the set of remaining vars, so child 
        nodes do not optimize over them.
        """
        frac_ones_y = self._get_frac_ones_y()
        n_samples = self._get_n_samples()
        self.classification = 1 if frac_ones_y > 0.5 else 0

        is_pure = True if (frac_ones_y == 1 or frac_ones_y == 0) else False
        is_too_small = True if n_samples <= 2*min_samples_per_node else False
        if is_pure or is_too_small:
            self.leaf = True
            return
        
        min_impurity = float('inf')
        split_vars = self._select_vars(self.remaining_vars, var_split_type, num_rand_vars)
        for var in split_vars:
            if self._is_var_constant(var):
                self.remaining_vars.remove(var)
                continue

            var_optimal_split = self._get_optimal_split(var, min_samples_per_node, impurity_func)
            if var_optimal_split is not None:
                var_impurity, var_split_point = var_optimal_split
                if var_impurity < min_impurity:
                    min_impurity = var_impurity
                    split_var = var
                    split_point = var_split_point
        
        if min_impurity == float('inf'):
            self.leaf = True
            return
        self.split_var = split_var
        self.split_point = split_point
    
    def _select_vars(
        self, 
        all_remaining_vars: Union[List[int], List[str]],
        var_split_type: str,
        num_rand_vars: int
    ) -> Union[List[str], List[int]]:
        """
        A utility function which takes a set of vars and returns the set of vars to optimize over. 
        If var_split_type = 'best', it returns all remaining vars. If var_split_type = 'random', it 
        returns a random subset of size num_rand_vars.
        """
        if var_split_type == 'best':
            return all_remaining_vars
        elif var_split_type == 'random':
            if len(all_remaining_vars) <= num_rand_vars:
                return all_remaining_vars
            rand_vars = random.sample(all_remaining_vars, num_rand_vars)
            return rand_vars
    
    def _get_optimal_split(
        self, 
        var: Union[str, int], 
        min_samples_per_node: int, 
        impurity_func: Union[Callable[[np.ndarray], float], str]
    ) -> Tuple[float, float]:
        """
        Returns a tuple of (impurity, split_point) for the optimal split at a node.
        Implemented via array computations for MatrixNode and sql queries for SqlNode.
        """
        raise NotImplementedError('Each subclass must implement this method!')

    def _get_frac_ones_y(self) -> float:
        """
        Returns the fraction of ones in the node's region.
        """
        raise NotImplementedError('Each subclass must implement this method!')

    def _get_n_samples(self) -> int:
        """
        Returns the number of samples in the node's region.
        """
        raise NotImplementedError('Each subclass must implement this method!')

    def _is_var_constant(self, var: Union[str, int]) -> bool:
        """
        Returns True if a var is constant within the node's region.
        """
        raise NotImplementedError('Each subclass must implement this method!')

    def _get_child_region(self, side: str ='left') -> Union[List[str], List[int]]:
        """
        Returns the appropriate region for the left or right child.
        """
        raise NotImplementedError('Each subclass must implement this method!')

    def _get_misclassification_cost(self) -> float:
        """
        Computes the node's misclassification cost to be used by pruning procedure.
        """
        raise NotImplementedError('Each subclass must implement this method!')


class MatrixNode(Node):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
        remaining_vars: List[int],
        region: List[int],
        node_idx: int,
    ):
        self.X = X
        self.y = y
        super().__init__(depth, remaining_vars, region, node_idx)
    
    @staticmethod
    def _label_count(arr) -> np.ndarray:
        """
        A utility function which takes an array of labels and returns np.array([label_count_0, label_count_1]).
        """
        label_counts = np.bincount(arr)
        if len(label_counts) == 1:
            label_counts = np.append(label_counts, 0)
        return label_counts

    def _get_frac_ones_y(self) -> float:
        labels = self.y[self.region]
        label_counts = self._label_count(labels)
        return label_counts[1] / sum(label_counts)

    def _get_n_samples(self) -> int:
        return len(self.region)

    def _is_var_constant(self, var: int) -> bool:
        var_values = self.X[self.region, var]
        sort_idxs = np.argsort(var_values)
        var_values_sorted = var_values[sort_idxs]
        return True if var_values_sorted[0] == var_values_sorted[-1] else False
    
    def _get_optimal_split(
        self,
        var: str, 
        min_samples_per_node: int, 
        impurity_func: Callable[[np.ndarray], float]
    ) -> Tuple[float, float]:
        n_samples = self._get_n_samples()
        min_impurity = float('inf')

        var_values = self.X[self.region, var]
        labels = self.y[self.region]
        sort_idxs = np.argsort(var_values)
        var_values_sorted = var_values[sort_idxs]
        labels_sorted = labels[sort_idxs]

        left_label_counts = np.zeros(shape=(2,))
        right_label_counts = self._label_count(labels)

        prev_split_idx = 0
        split_idx = min_samples_per_node - 1
        while split_idx < n_samples - min_samples_per_node:
            if var_values_sorted[split_idx] == var_values_sorted[split_idx + 1]:
                repeat_split_idx = np.searchsorted(var_values_sorted, var_values_sorted[split_idx], side='right') - 1
                if repeat_split_idx >= n_samples - min_samples_per_node:
                    break
                else:
                    split_idx = repeat_split_idx
            
            left_label_counts = self._label_count(labels_sorted[:(split_idx + 1)])
            right_label_counts = self._label_count(labels_sorted[(split_idx + 1):])
            left_impurity = impurity_func(left_label_counts)
            right_impurity = impurity_func(right_label_counts)
            left_prob = sum(left_label_counts) / n_samples
            right_prob = sum(right_label_counts) / n_samples
            impurity = left_prob*left_impurity + right_prob*right_impurity

            if impurity < min_impurity:
                min_impurity = impurity
                optimal_split_point = var_values_sorted[split_idx]
            
            split_idx += 1

        if min_impurity == float('inf'):
            return
        return min_impurity, optimal_split_point
    
    def _get_child_region(self, side: str = 'left') -> np.ndarray:
        split_var_vals = self.X[self.region, self.split_var]
        region_arr = np.asarray(self.region)
        if side == 'left':
            child_region = list(region_arr[split_var_vals <= self.split_point])
        else:
            child_region = list(region_arr[split_var_vals > self.split_point])
        return child_region
    
    def _get_misclassification_cost(self) -> float:
        p_of_total = len(self.region) / len(self.y)
        p_misclassified = sum(self.classification != self.y[self.region]) / len(self.region)
        return p_misclassified*p_of_total


class SqlNode(Node):
    def __init__(
        self,
        db_connection: connection,
        db_table_name: str,
        db_y_col: str,
        depth: int,
        remaining_vars: List[str],
        region: List[str],
        node_idx: int,
        precision: int = 5
    ):
        self.db_connection = db_connection
        self.db_table_name = db_table_name
        self.db_y_col = db_y_col
        super().__init__(depth, remaining_vars, region, node_idx)
        self.query_params = {
            'y_col': AsIs(self.db_y_col),
            'table': AsIs(self.db_table_name),
            'where_cond': AsIs(' AND '.join(self.region))
        }
        self.precision = precision

    def _fetchone_query(self, query_str: str, query_params: dict = None) -> float:
        """
        A utility function to run a sql query which returns a single float value.
        """
        if query_params is None:
            query_params = self.query_params
        cur = self.db_connection.cursor()
        cur.execute(query_str, query_params)
        val = cur.fetchone()[0]
        cur.close()
        return val

    def _get_frac_ones_y(self) -> float:
        return self._fetchone_query(
            """SELECT COALESCE(fraction_ones(%(y_col)s), 0) FROM %(table)s WHERE %(where_cond)s;"""
        )
    
    def _get_n_samples(self) -> int:
        return self._fetchone_query(
            'SELECT COUNT(*) FROM %(table)s WHERE %(where_cond)s;'
        )
    
    def _is_var_constant(self, var: str) -> bool:
        query_params = self.query_params.copy()
        query_params.update({'var': AsIs(var)})
        query_str = 'SELECT COUNT(DISTINCT %(var)s) FROM %(table)s WHERE %(where_cond)s;'
        num_distinct_vals = self._fetchone_query(query_str, query_params)
        return True if num_distinct_vals == 1 else False
    
    def _get_optimal_split(
        self,
        var: str, 
        min_samples_per_node: int, 
        impurity_func: str,
    ) -> Tuple[float, float]:
        query_params = self.query_params.copy()
        query_params.update({'var': AsIs(var), 'impurity_func': AsIs(impurity_func)})
        cur = self.db_connection.cursor('split_options')
        cur.execute('SELECT %(var)s FROM %(table)s WHERE %(where_cond)s;', query_params)
        
        n_samples = self._get_n_samples()
        min_impurity = float('inf')

        for split_point in cur:
            eval_split_point = split_point[0]
            query_params.update({
                'split_where_cond_left': AsIs(' AND '.join(self.region + ['round(%s::numeric, %d) <= round(%f::numeric, %d)' \
                    % (var, self.precision, eval_split_point, self.precision)])),
                'split_where_cond_right': AsIs(' AND '.join(self.region + ['round(%s::numeric, %d) > round(%f::numeric, %d)' \
                    % (var, self.precision, eval_split_point, self.precision)])),
            })

            n_samples_to_left_query = 'SELECT COUNT(*) FROM %(table)s WHERE %(split_where_cond_left)s;'
            n_samples_to_right_query = 'SELECT COUNT(*) FROM %(table)s WHERE %(split_where_cond_right)s;'
            n_samples_to_left = self._fetchone_query(n_samples_to_left_query, query_params)
            n_samples_to_right = self._fetchone_query(n_samples_to_right_query, query_params)
            
            if n_samples_to_left < min_samples_per_node or n_samples_to_right < min_samples_per_node:
                continue

            impurity_left = 0
            impurity_right = 0
            if n_samples_to_left > 0:
                impurity_left_query = """
                    SELECT COALESCE(%(impurity_func)s(fraction_ones(%(y_col)s)), 0)
                    FROM %(table)s WHERE %(split_where_cond_left)s;
                """
                impurity_left = self._fetchone_query(impurity_left_query, query_params)
            if n_samples_to_right > 0:
                impurity_right_query = """
                    SELECT COALESCE(%(impurity_func)s(fraction_ones(%(y_col)s)), 0)
                    FROM %(table)s WHERE %(split_where_cond_right)s;
                """
                impurity_right = self._fetchone_query(impurity_right_query, query_params)

            p_left = n_samples_to_left / n_samples
            p_right = n_samples_to_right / n_samples
            impurity = p_left*impurity_left + p_right*impurity_right

            if impurity < min_impurity:
                min_impurity = impurity
                optimal_split_point = eval_split_point
        cur.close()

        if min_impurity == float('inf'):
            return
        return min_impurity, optimal_split_point
    
    def _get_child_region(self, side='left') -> List[str]:
        if side == 'left':
            split_condition = '%s <= %s' % (self.split_var, self.split_point)
        else:
            split_condition = '%s > %s' % (self.split_var, self.split_point)
        child_region = self.region + [split_condition]
        return child_region
    
    def _get_misclassification_cost(self) -> float:
        query_params = self.query_params.copy()
        query_params.update({'classification': self.classification})
        n_samples = self._get_n_samples()
        n_total_samples_query = 'SELECT COUNT(*) FROM %(table)s;'
        n_total_samples = self._fetchone_query(n_total_samples_query, query_params)
        n_misclassified_query = """
            SELECT COUNT(*) FROM %(table)s WHERE %(where_cond)s AND %(y_col)s != %(classification)s
        """
        n_misclassified = self._fetchone_query(n_misclassified_query, query_params)
        p_of_total = n_samples / n_total_samples
        p_misclassified = n_misclassified / n_samples
        return p_of_total*p_misclassified


class Tree:
    """
    A classification tree. This can be trained with input data and then used to make predictions.

    Attributes
    ----------
    X : np.ndarray
        training data array of shape (N, k)
    y : np.ndarray
        training label array of shape (N,)
    min_samples_per_node : Optional[int], optional
        each node's region must have at least this many samples
    impurity_func : Optional[Union[Callable[[np.ndarray], float], str]], optional
        a function which takes np.array([label_count_0, label_count_1]) and returns a float or a 
        string which is one of ('gini_index', 'bayes_error', 'cross_entropy')
    root : Node
        the root node of the tree
    var_split_type : Optional[str], optional
        if best, optimize splits over all variables; if random, optimize splits over random subset of variables
    num_rand_vars : Optional[int], optional
        if var_split_type is random, the number of random variables to select
    random_seed : Optional[int], optional
        random seed; only matters if var_split_type = 'random'
    node_idx_counter : int
        an integer counter to assign unique node_idx to nodes
    """
    def __init__(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        min_samples_per_node: Optional[int] = 10,
        impurity_func: Optional[Union[Callable[[np.ndarray], float], str]] = gini_index,
        var_split_type: Optional[str] = 'best',
        num_rand_vars: Optional[int] = 5,
        random_seed: Optional[int] = 99
    ):
        self.X = X
        self.y = y
        self.min_samples_per_node = min_samples_per_node
        self.impurity_func = impurity_func
        self.var_split_type = var_split_type
        self.num_rand_vars = num_rand_vars
        np.random.seed(random_seed)
        self.root = None
        self.node_idx_counter = 0

    def train(self):
        """
        Trains the tree on the input data. It creates a root node and then recursively splits 
        nodes until leaves are generated.
        """
        self.root = self._get_root_node()
        node_stack = [self.root]
        while len(node_stack) > 0:
            node = node_stack.pop()
            node._split_node(self.var_split_type, self.num_rand_vars, self.min_samples_per_node, self.impurity_func)
            if not node.leaf:
                left_node, right_node = self._get_child_nodes(node)
                node.left = left_node
                node.right = right_node
                node_stack.append(left_node)
                node_stack.append(right_node)
        return self
    
    def _get_root_node(self) -> MatrixNode:
        """
        Builds the root node of the tree.
        """
        self.node_idx_counter += 1
        n_samples, n_vars = self.X.shape
        region = np.arange(0, n_samples)
        var_indices = [i for i in range(n_vars)]
        root = MatrixNode(self.X, self.y, 1, var_indices, region, self.node_idx_counter)
        return root
    
    def _get_child_nodes(self, node: MatrixNode) -> Tuple[MatrixNode, MatrixNode]:
        """
        Builds the child nodes of a given node.
        """
        left_region = node._get_child_region(side='left')
        right_region = node._get_child_region(side='right')
        remaining_vars = node.remaining_vars
        depth = node.depth + 1
        self.node_idx_counter += 1
        left_node = MatrixNode(self.X, self.y, depth, remaining_vars, left_region, self.node_idx_counter)
        self.node_idx_counter += 1
        right_node = MatrixNode(self.X, self.y, depth, remaining_vars, right_region, self.node_idx_counter)
        return left_node, right_node

    def _predict_point(self, root_node: Node, x_i: np.ndarray) -> float:
        """
        Takes a root node and prediction point in the form of np.ndarray of shape (k, 1). Recursively goes 
        through tree until the appropriate leaf is reached and outputs the leaf's classification.
        """
        if root_node.leaf:
            return root_node.classification
        split_var = root_node.split_var
        split_point = root_node.split_point
        if x_i[split_var] <= split_point:
            return self._predict_point(root_node.left, x_i)
        else:
            return self._predict_point(root_node.right, x_i)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Takes a data matrix of shape (N, k) and returns predictions for each of the N points.
        """
        n_samples, _ = X.shape
        predictions = np.array([self._predict_point(self.root, X[i]) for i in range(n_samples)])
        return predictions
        

class SqlTree(Tree):
    def __init__(
        self, 
        db_connection: connection,
        db_table_name: str,
        db_X_cols: List[str], 
        db_y_col: str,
        min_samples_per_node: Optional[int] = 10,
        impurity_func: Optional[str] = 'gini_index',
        var_split_type: Optional[str] = 'best',
        num_rand_vars: Optional[int] = 5,
        random_seed: Optional[int] = 99,
        precision: int = 5
    ):
        self.db_connection = db_connection
        self.db_table_name = db_table_name
        self.db_X_cols = db_X_cols
        self.db_y_col = db_y_col
        super().__init__(
            min_samples_per_node = min_samples_per_node, var_split_type = var_split_type,
            num_rand_vars = num_rand_vars, random_seed = random_seed
        )
        self.impurity_func = impurity_func
        self.precision = precision
    
    def _get_root_node(self) -> SqlNode:
        self.node_idx_counter += 1
        region = ['%s is NOT NULL' % self.db_y_col]
        root = SqlNode(
            self.db_connection, self.db_table_name, self.db_y_col, 
            1, self.db_X_cols, region, self.node_idx_counter, self.precision
        )
        return root
    
    def _get_child_nodes(self, node: SqlNode) -> Tuple[SqlNode, SqlNode]:
        left_region = node._get_child_region(side='left')
        right_region = node._get_child_region(side='right')
        remaining_vars = node.remaining_vars
        depth = node.depth + 1
        self.node_idx_counter += 1
        left_node = SqlNode(
            self.db_connection, self.db_table_name, self.db_y_col, depth, 
            remaining_vars, left_region, self.node_idx_counter, self.precision
        )
        self.node_idx_counter += 1
        right_node = SqlNode(
            self.db_connection, self.db_table_name, self.db_y_col, depth, 
            remaining_vars, right_region, self.node_idx_counter, self.precision
        )
        return left_node, right_node
    
    def _predict_point(self, root_node: Node, x_i: dict) -> float:
        """
        Takes a root node and prediction point in the form of a dictionary where the keys are variable names. 
        """
        if root_node.leaf:
            return root_node.classification
        split_var = root_node.split_var
        split_point = root_node.split_point
        if round(x_i[split_var], self.precision)  <= round(split_point, self.precision):
            return self._predict_point(root_node.left, x_i)
        else:
            return self._predict_point(root_node.right, x_i)
    
    def predict(self, X: List[dict]) -> np.ndarray:
        """
        Takes a data matrix in the form of a list of dicts, where each dict corresponds to a single
        prediction point. The keys are variable names.
        """
        n_samples = len(X)
        predictions = np.array([self._predict_point(self.root, X[i]) for i in range(n_samples)])
        return predictions


class PrunedTree:
    """
    A pruned classification tree. This takes a full size tree, alpha param and generates the optimal 
    pruned tree per the cost complexity measure. Details on algorithm are here:
        http://www.ams.org/publicoutreach/feature-column/fc-2014-12

    Attributes
    ----------
    orig_tree : Tree
        the original full size tree
    alpha : float
        the cost complexity parameter; a higher alpha means a smaller tree
    pruned_trees : list[Tree]
        the sequence of trees generated en route to the optimal tree
    opt_pruned_tree : Tree
        the optimal tree
    """
    def __init__(self, orig_tree: Tree, alpha: float):
        self.orig_tree = orig_tree
        self.alpha = alpha
        self.pruned_trees = None
        self.opt_pruned_tree = None
    
    def train(self) -> Tree:
        """
        Trains and returns the pruned tree. 
        """
        copy_tree = copy.copy(self.orig_tree)
        pruned_trees = []
        self._build_pruned_trees(self.alpha, copy_tree, pruned_trees)
        self.pruned_trees = [copy_tree] + pruned_trees
        self.opt_pruned_tree = self.pruned_trees[-1]
        return self.opt_pruned_tree
    
    def _build_pruned_trees(self, alpha: float, tree: Tree, pruned_trees: list = []) -> List[Tree]:
        """
        Recursively prunes a tree until it is just the root or the min_cost_complexity_imp over 
        all nodes in the tree except the leaves is greater than alpha.
        """
        self._preprocess(tree)
        if tree.root.num_leaves == 1:
            return pruned_trees
        if tree.root.min_cost_complexity_imp > alpha:
            return pruned_trees
        else:
            copy_tree = copy.deepcopy(tree)
            self._prune_nodes(copy_tree.root, copy_tree.root.min_cost_complexity_imp)
            pruned_trees.append(copy_tree)
            self._build_pruned_trees(alpha, copy_tree, pruned_trees)
        return pruned_trees
    
    def _prune_nodes(self, node: Node, min_cost_complexity_imp: float):
        """
        Finds the nodes in a tree with node.cost_complexity_imp == min_cost_complexity_imp
        and converts them to a leaf.
        """
        if node:
            if node.cost_complexity_imp == min_cost_complexity_imp:
                node.right = None
                node.left = None
                node.leaf = True
            self._prune_nodes(node.left, min_cost_complexity_imp)
            self._prune_nodes(node.right, min_cost_complexity_imp)
    
    def _preprocess(self, tree: Tree):
        """
        Populates the nodes of a tree with the following data about its subtree:
            - num of leaves
            - misclassification cost
            - cost complexity improvement
            - min cost complexity improvement over all nodes in the subtree
        """
        queue = deque()
        queue.append(tree.root)
        stack = deque()
        
        while queue:
            curr = queue.popleft()
            stack.append(curr)
            if curr.right:
                queue.append(curr.right)
            if curr.left:
                queue.append(curr.left)
        
        while stack:
            node = stack.pop()
            if node.leaf:
                node.num_leaves = 1
                node.cost = node._get_misclassification_cost()
                node.cost_complexity_imp = float('inf')
                node.min_cost_complexity_imp = float('inf')
            else:
                node.num_leaves = node.left.num_leaves + node.right.num_leaves
                node.cost = node.right.cost + node.left.cost
                cost_complexity_imp_num = node._get_misclassification_cost() - node.cost
                cost_complexity_imp_den = node.num_leaves - 1
                node.cost_complexity_imp = cost_complexity_imp_num / cost_complexity_imp_den
                node.min_cost_complexity_imp = min(
                    node.cost_complexity_imp, 
                    node.right.cost_complexity_imp, 
                    node.left.cost_complexity_imp
                )