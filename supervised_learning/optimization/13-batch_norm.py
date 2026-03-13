#!/usr/bin/env python3
"""
    Batch Normalization
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Apply batch normalization to a matrix.

    :param Z: ndarray, shape(m, n) to normalize
    :param gamma: ndarray, shape(n,) containing the scaling factors
    :param beta: ndarray, shape(n,) containing the shifting factors
    :param epsilon: float, small constant to prevent division by zero
    :return: tuple containing normalized data (Z_norm, mean, variance)
    """
    # Calculate mean and variance
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    # Normalize
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    # Scale and shift
    Z_norm = gamma * Z_norm + beta
    return Z_norm
