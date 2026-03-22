#!/usr/bin/env python3
"""
L2 Regularization Cost
"""

import numpy as np

def l2_reg_cost(lambtha, weights, L, m):
    """
    Calculates the L2 regularization cost for a neural network.

    Args:
        lambtha (float): The regularization hyperparameter.
        weights (dict): Dictionary containing the weights of the network.
        L (int): The number of layers in the network.
        m (int): The number of examples in the training set.

    Returns:
        float: The L2 regularization cost.
    """
    l2_cost = 0
    for l in range(1, L + 1):
        l2_cost += np.sum(np.square(weights['W' + str(l)]))
    
    l2_cost = (lambtha / (2 * m)) * l2_cost
    
    return l2_cost
