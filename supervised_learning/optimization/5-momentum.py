#!/usr/bin/env python3
"""
    Momentum Optimization
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Update variables using the momentum optimization algorithm.

    :param alpha: float, learning rate
    :param beta1: float, momentum hyperparameter
    :param var: ndarray, current variable values
    :param grad: ndarray, gradient of the cost function with respect to var
    :param v: ndarray, current velocity (momentum term)
    :return: tuple containing updated variables (var, v)
    """
    # Update velocity
    v = beta1 * v + (1 - beta1) * grad

    # Update variables
    var = var - alpha * v

    return var, v
