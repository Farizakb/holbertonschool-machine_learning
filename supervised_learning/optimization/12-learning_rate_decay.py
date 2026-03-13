#!/usr/bin/env python3
"""
    Learning Rate Decay TensorFlow
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a TensorFlow optimization operation for learning rate decay.

    :param alpha: float, initial learning rate
    :param decay_rate: float, decay rate
    :param decay_step: int, number of iterations after which to decay the 
    learning rate
    :return: tf.keras.optimizers.schedules.ExponentialDecay instance
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
