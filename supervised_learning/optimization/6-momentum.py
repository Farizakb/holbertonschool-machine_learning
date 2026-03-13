#!/usr/bin/env python3
"""
    Momentum Optimization TensorFlow
"""

import numpy as np
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Creates a TensorFlow optimization operation for momentum.

    :param alpha: float, learning rate
    :param beta1: float, momentum hyperparameter
    :return: tf.train.Optimizer instance
    """
    return tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
