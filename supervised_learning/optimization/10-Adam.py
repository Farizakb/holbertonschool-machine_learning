#!/usr/bin/env python3
"""
    Adam Optimization TensorFlow
"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Creates a TensorFlow optimization operation for Adam.

    :param alpha: float, learning rate
    :param beta1: float, decay rate for the moving average of first moments
    :param beta2: float, decay rate for the moving average of second moments
    :param epsilon: float, small constant to prevent division by zero
    :return: tf.train.Optimizer instance
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
