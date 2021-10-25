import numpy as np


def compute_p(label_counts: np.ndarray) -> float:
    """
    Computes fraction of points in class 0. If there is a perfect classification, returns p = 1.
    
    Parameters
    ----------
    label_counts : np.ndarray
        First element is counts of class 0. Second element is counts of class 1.
    """
    if 0 in label_counts:
        return 1
    return label_counts[0] / sum(label_counts)


def bayes_error(label_counts: np.ndarray) -> float:
    """
    f(p) = min(p, 1-p)
    """
    p = compute_p(label_counts)
    return min(p, 1 - p)


def cross_entropy(label_counts: np.ndarray) -> float:
    """
    f(p) = -p*log(p) - (1 - p)*log(1 - p)
    """
    p = compute_p(label_counts)
    return 0 if p == 1 else -p*np.log(p) - (1 - p)*np.log(1 - p)


def gini_index(label_counts: np.ndarray) -> float:
    """
    f(p) = p*(1-p)
    """
    p = compute_p(label_counts)
    return p*(1 - p)
