#!/usr/bin/env python3
"""
    Batch Normalization TensorFlow
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.
    
    Args:
        prev: The activated output of the previous layer.
        n: The number of nodes in the layer to be created.
        activation: The activation function to be used on the output.
        
    Returns:
        A tensor of the activated output for the layer.
    """
    # 1. Define the kernel initializer
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    # 2. Create the base Dense layer 
    # Note: We disable the internal bias because Batch Norm's 'beta' handles the offset
    dense_layer = tf.keras.layers.Dense(
        units=n, 
        kernel_initializer=initializer,
        use_bias=False 
    )

    # Apply the linear transformation (Wx)
    z = dense_layer(prev)

    # 3. Calculate mean and variance for the current batch
    # axes=[0] calculates moments across the batch dimension
    mean, variance = tf.nn.moments(z, axes=[0])

    # 4. Initialize trainable parameters gamma and beta
    gamma = tf.Variable(tf.ones([n]), name="gamma", trainable=True)
    beta = tf.Variable(tf.zeros([n]), name="beta", trainable=True)

    # 5. Apply Batch Normalization
    epsilon = 1e-7
    # Normalized = gamma * (z - mean) / sqrt(variance + epsilon) + beta
    normalized = tf.nn.batch_normalization(
        z,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon
    )

    # 6. Apply the activation function and return
    return activation(normalized)
