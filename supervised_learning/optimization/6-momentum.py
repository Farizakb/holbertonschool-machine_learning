#!/usr/bin/env python3
"""
    Momentum Optimization TensorFlow
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Creates a TensorFlow optimization operation for momentum.

    :param alpha: float, learning rate
    :param beta1: float, momentum hyperparameter
    Returns:
        optimizer: Optimizer object for gradient descent with momentum.
    """
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=alpha,
        momentum=beta1)
    return optimizer
