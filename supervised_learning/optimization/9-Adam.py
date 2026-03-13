#!/usr/bin/env python3
"""
    Adam Optimization
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Update variables using the Adam optimization algorithm.

    :param alpha: float, learning rate
    :param beta1: float, decay rate for the moving average of first moments
    :param beta2: float, decay rate for the moving average of second moments
    :param epsilon: float, small constant to prevent division by zero
    :param var: ndarray, current variable values
    :param grad: ndarray, gradient of the cost function with respect to var
    :param v: ndarray, moving average of first moments
    :param s: ndarray, moving average of second moments
    :param t: int, iteration number
    :return: tuple containing updated variables (var, v, s)
    """
    # Update moving average of first moments
    v = beta1 * v + (1 - beta1) * grad

    # Update moving average of second moments
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Bias correction
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)

    # Update variables
    var = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var, v, s
