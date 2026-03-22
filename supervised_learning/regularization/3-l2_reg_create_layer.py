#!/usr/bin/env python3
"""
    Create layer with L2 regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Function that creates a tensorflow layer includes L2 regularization:
    - prev is a tensor containing the output of the previous layer
    - n is the number of nodes the new layer should contain
    - activation is the activation function that should be used on the layer
    - lambtha is the L2 regularization parameter
    Returns: the output of the new layer
    """
    # set initialization to He et. al (using fan_avg as seen in other tasks)
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode='fan_avg')

    # Create dense layer with L2 regularization
    regularizer = tf.keras.regularizers.l2(lambtha)
    new_layer = tf.keras.layers.Dense(units=n,
                                      activation=activation,
                                      kernel_initializer=initializer,
                                      kernel_regularizer=regularizer)

    # apply layer to input
    output = new_layer(prev)

    return output
