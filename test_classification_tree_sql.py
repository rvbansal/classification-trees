import pytest

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from classification_tree import SqlTree, PrunedTree, SqlNode, Tree
import impurity_functions
from validator import is_valid
credentials = pytest.importorskip("credentials")


def test_sql_node_functions():
    table_query = """
        CREATE TABLE test_table (
            outcome int,
            var1 float,
            var2 float,
            var3 float
        );

        INSERT INTO test_table (outcome, var1, var2, var3)
        VALUES 
            (1, 10, 200, 30),
            (0, 10, 1, 8),
            (1, 10, 200, 8),
            (0, 10, 89, 9),
            (1, 10, 200, 9000),
            (0, 10, 6, 3),
            (1, 10, 200, 900),
            (0, 10, 3, 2)
    """
    conn = psycopg2.connect(
        host = 'sculptor.stat.cmu.edu', 
        database = credentials.DB_NAME,
        user = credentials.DB_USER, 
        password = credentials.DB_PASSWORD
    )
    cur = conn.cursor()
    cur.execute(table_query)

    sql_node = SqlNode(
        db_connection = conn,
        db_table_name = 'test_table',
        db_y_col = 'outcome',
        depth = 10,
        remaining_vars = ['var1', 'var2', 'var3'],
        region = ['%s is NOT NULL' % 'outcome'],
        node_idx = 20
    )

    assert sql_node._get_frac_ones_y() == 0.5
    assert sql_node._get_n_samples() == 8
    assert sql_node._is_var_constant(var='var1') == True
    assert sql_node._is_var_constant(var='var2') == False
    assert sql_node._is_var_constant(var='var3') == False
    assert sql_node._get_optimal_split('var2', 2, 'gini_index') == (0, 89)
    assert sql_node._get_optimal_split('var2', 2, 'bayes_error') == (0, 89)
    assert sql_node._get_optimal_split('var2', 8, 'bayes_error') == None
    assert sql_node._get_optimal_split('var1', 2, 'gini_index') == None
    assert sql_node._get_optimal_split('var1', 8, 'bayes_error') == None
    assert sql_node._get_optimal_split('var3', 5, 'gini_index') == None
    assert sql_node._get_optimal_split('var3', 1, 'gini_index') == (0.1, 9)


def test_sql_tree_no_splits():
    N = 1000
    X0 = np.full(shape=(N,), fill_value=5)
    X1 = np.full(shape=(N,), fill_value=-10)
    y = np.full(shape=(N,), fill_value=1)
    stacked = np.dstack([X0, X1, y])
    insert_values = [tuple(i.astype('float')) for i in stacked[0]]

    create_table_query = """
        CREATE TABLE test_table (
            var1 int,
            var2 int,
            outcome int
        );
    """
    conn = psycopg2.connect(
        host = 'sculptor.stat.cmu.edu', 
        database = credentials.DB_NAME,
        user = credentials.DB_USER, 
        password = credentials.DB_PASSWORD
    )    
    cur = conn.cursor()
    cur.execute(create_table_query)
    execute_values(cur, 'INSERT INTO test_table(var1, var2, outcome) VALUES %s', insert_values)

    tree = SqlTree(
        db_connection = conn, 
        db_table_name = 'test_table', 
        db_X_cols = ['var1', 'var2'], 
        db_y_col = 'outcome'
    ).train()

    assert isinstance(tree, SqlTree)
    assert tree.root is not None
    assert tree.root.left is None
    assert tree.root.right is None
    assert tree.root.split_point is None
    assert tree.root.split_var is None
    assert np.all(tree.root.remaining_vars == ['var1', 'var2'])
    assert tree.root.leaf == True
    assert tree.root.classification == 1
    assert tree.root.node_idx == 1
    assert tree.root.depth == 1
    assert tree.root.region == ['outcome is NOT NULL']


def test_sql_tree_perfect_classifier():
    N = 100
    X0 = np.random.uniform(-1, 1, size=(N,))
    X1 = np.random.uniform(-1, 1, size=(N,))
    y = ((X0 >= 0) & (X1 < -0.5)).astype(int).reshape(-1).T
    stacked = np.dstack([X0, X1, y])
    insert_values = [tuple(i.astype('float')) for i in stacked[0]]
    
    create_table_query = """
        CREATE TABLE test_table (
            var1 float,
            var2 float,
            outcome int
        );
    """
    conn = psycopg2.connect(
        host = 'sculptor.stat.cmu.edu', 
        database = credentials.DB_NAME,
        user = credentials.DB_USER, 
        password = credentials.DB_PASSWORD
    )
    cur = conn.cursor()
    cur.execute(create_table_query)
    execute_values(cur, 'INSERT INTO test_table(var1, var2, outcome) VALUES %s', insert_values)

    tree = SqlTree(
        db_connection = conn, 
        db_table_name = 'test_table', 
        db_X_cols = ['var1', 'var2'], 
        db_y_col = 'outcome', 
        min_samples_per_node = 1
    ).train()

    test_pt = {'var1': 0.1, 'var2': -0.6}
    test_pts = [{'var1': x0, 'var2': x1} for x0, x1 in zip(X0, X1)]
    prediction = tree._predict_point(tree.root, test_pt)
    predictions = tree.predict(test_pts)
    assert prediction == 1
    assert sum(predictions != y) == 0


def test_pruned_sql_tree():
    N = 100
    X0 = np.random.uniform(-1, 1, size=(N,))
    X1 = np.random.uniform(-1, 1, size=(N,))
    y = np.full(shape=(N,), fill_value=1)
    stacked = np.dstack([X0, X1, y])
    insert_values = [tuple(i.astype('float')) for i in stacked[0]]

    create_table_query = """
        CREATE TABLE test_table (
            var1 float,
            var2 float,
            outcome int
        );
    """
    conn = psycopg2.connect(
        host = 'sculptor.stat.cmu.edu', 
        database = credentials.DB_NAME,
        user = credentials.DB_USER, 
        password = credentials.DB_PASSWORD
    )    
    cur = conn.cursor()
    cur.execute(create_table_query)
    execute_values(cur, 'INSERT INTO test_table(var1, var2, outcome) VALUES %s', insert_values)

    tree = SqlTree(
        db_connection = conn, 
        db_table_name = 'test_table', 
        db_X_cols = ['var1', 'var2'], 
        db_y_col = 'outcome', 
        min_samples_per_node = 1
    ).train()

    pruned_small = PrunedTree(orig_tree = tree, alpha=0)
    pruned_large = PrunedTree(orig_tree = tree, alpha=1000000000000)
    pruned_opt_small = pruned_small.train()
    pruned_opt_large = pruned_large.train()
    assert isinstance(pruned_opt_small, Tree)
    assert isinstance(pruned_opt_large, Tree)
    assert pruned_opt_large.root.num_leaves <= pruned_opt_small.root.num_leaves
    assert pruned_opt_large.root.num_leaves <= pruned_large.pruned_trees[0].root.num_leaves
    assert pruned_opt_small.root.num_leaves <= pruned_small.pruned_trees[0].root.num_leaves


def test_matrix_vs_sql_tree():
    N = 100
    X0 = np.random.uniform(-500, 500, size=(N,))
    X1 = np.random.uniform(-500, 500, size=(N,))
    X = np.hstack([X0.reshape(-1, 1), X1.reshape(-1, 1)])
    y = np.random.choice([0, 1], size=(N, ))
    
    stacked = np.dstack([X0, X1, y])
    insert_values = [tuple(i.astype('float')) for i in stacked[0]]

    conn = psycopg2.connect(
        host = 'sculptor.stat.cmu.edu', 
        database = credentials.DB_NAME,
        user = credentials.DB_USER, 
        password = credentials.DB_PASSWORD
    )
    cur = conn.cursor()
    create_table_query = """
        CREATE TABLE test_table (
            var1 numeric,
            var2 numeric,
            outcome int
        );
    """
    cur.execute(create_table_query)
    execute_values(cur, 'INSERT INTO test_table(var1, var2, outcome) VALUES %s', insert_values)

    m_tree = Tree(X=X, y=y, min_samples_per_node=5).train()
    s_tree = SqlTree(
        db_connection = conn, 
        db_table_name = 'test_table', 
        db_X_cols = ['var1', 'var2'], 
        db_y_col = 'outcome',
        min_samples_per_node=5
    ).train()

    m_test_pt1 = np.array([0.5, 0.8])
    s_test_pt1 = {'var1': 0.5, 'var2': 0.8}
    assert m_tree._predict_point(m_tree.root, m_test_pt1) == \
        s_tree._predict_point(s_tree.root, s_test_pt1)
    
    m_test_pt2 = np.array([-1000, 0.8])
    s_test_pt2 = {'var1': -1000, 'var2': 0.8}
    assert m_tree._predict_point(m_tree.root, m_test_pt2) == \
        s_tree._predict_point(s_tree.root, s_test_pt2)
    
    if m_tree.root.split_point:
        assert pytest.approx(m_tree.root.split_point, s_tree.root.split_point)
    if m_tree.root.left.split_point:
        assert pytest.approx(m_tree.root.left.split_point, s_tree.root.left.split_point)
    if m_tree.root.right.split_point:
        assert pytest.approx(m_tree.root.right.split_point, s_tree.root.right.split_point)
