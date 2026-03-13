#!/usr/bin/env python3
"""
    RMSProp Optimization TensorFlow
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates a TensorFlow optimization operation for RMSProp.

    :param alpha: float, learning rate
    :param beta2: float, decay rate for the moving average of squared gradients
    :param epsilon: float, small constant to prevent division by zero
    :return: tf.train.Optimizer instance
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon
    )
