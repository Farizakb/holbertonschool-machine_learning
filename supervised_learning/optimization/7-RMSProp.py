#!/usr/bin/env python3
"""
    RMSProp Optimization
"""

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Update variables using the RMSProp optimization algorithm.

    :param alpha: float, learning rate
    :param beta2: float, decay rate for the moving average of squared gradients
    :param epsilon: float, small constant to prevent division by zero
    :param var: ndarray, current variable values
    :param grad: ndarray, gradient of the cost function with respect to var
    :param s: ndarray, moving average of squared gradients
    :return: tuple containing updated variables (var, s)
    """
    # Update moving average of squared gradients
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Update variables
    var = var - alpha * grad / (np.sqrt(s) + epsilon)

    return var, s
