from typing import Union

import numpy as np
import matplotlib.pyplot as plt 

from classification_tree import Tree
from classification_forest import Forest


def plot_2d_classifier(
    X: np.ndarray, 
    y: np.ndarray, 
    classifier: Union[Tree, Forest]
):
    """
    Helps visualize predictions from a tree or forest by plotting it over a 2d grid.
    This ONLY works when X is of shape (N, 2).

    NOTE: I used the following resource in order to put together this code:
    https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    """
    X1_min, X1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    X2_min, X2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    X1_grid, X2_grid = np.meshgrid(np.arange(X1_min, X1_max, 0.1), np.arange(X2_min, X2_max, 0.5))
    X_grid = np.c_[X1_grid.ravel(), X2_grid.ravel()]
    predictions = classifier.predict(X_grid)
    predictions = predictions.reshape(X1_grid.shape)
    plt.contourf(X1_grid, X2_grid, predictions, alpha=0.5, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.xlim(X1_grid.min(), X1_grid.max())
    plt.ylim(X2_grid.min(), X2_grid.max())

    
