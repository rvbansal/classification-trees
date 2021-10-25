#!/usr/bin/env python3

import timeit
from typing import Callable, Optional, Union, List, Tuple, Dict

import numpy as np

from classification_tree import Tree


def benchmark_data(
    N: int,
    K: int,
    random_seed: int = 99
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates random data for benchmarking purposes. 
    Returns an X array of shape (N, K) and y array of shape (N,).
    """
    np.random.seed(random_seed)
    X = np.random.uniform(-10, 10, size=(N, K))
    y = np.random.choice([0, 1], size=(N,))
    return X, y


def benchmark(
    sample_sizes: Optional[List] = list(np.geomspace(1, 5000, num=5, dtype='int')),
    num_predictors: Optional[List] = list(np.geomspace(1, 100, num=5, dtype='int')),
    repeat: int = 10,
    random_seed: int = 99
):
    """
    Prints training and query times for each pair (N, K) = (sample_size, num_predictors).
    """
    def _print_exec_time_matrix(
        times: Dict[Tuple, float],
        title: str = 'Training Time (ms)',
        print_buffer: int = 15,
    ):
        print(title + '\n')
        print('(N, K)', end='')
        print(' '*(print_buffer - len('(N, K)')), end='')
        print(''.join(str(K).rjust(print_buffer) for K in num_predictors))
        print('-'*(print_buffer*(len(num_predictors) + 1)))
        
        for N in sample_sizes:
            print(str(N).ljust(print_buffer), end='')
            for K in num_predictors:
                formatted_time = '{:.3f}'.format(times[(N, K)]*1e3).rjust(print_buffer)
                print(formatted_time, end='')
            print()
        print('\n'*2)
    
    print('Running benchmarks ... ')
    print('\n')
    train_times, query_times = {}, {}
    for N in sample_sizes:
        for K in num_predictors:
            X, y = benchmark_data(N=N, K=K, random_seed=random_seed)
            tree = Tree(X=X, y=y)
            X_test, _ = benchmark_data(N=1000, K=K, random_seed=random_seed)
            repeat_train_times = timeit.Timer(lambda: tree.train()).repeat(repeat=repeat, number=1)
            repeat_query_times = timeit.Timer(lambda: tree.predict(X_test)).repeat(repeat=repeat, number=1)
            train_times[(N, K)] = min(repeat_train_times)
            query_times[(N, K)] = min(repeat_query_times)
    
    _print_exec_time_matrix(train_times, 'Training Time (ms)')
    _print_exec_time_matrix(query_times, 'Query Time (ms)')


if __name__ == '__main__': 
    benchmark()
