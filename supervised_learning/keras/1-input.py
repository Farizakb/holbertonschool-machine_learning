#!/usr/bin/env python3
"""
    Input
"""

import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx: number of input features
    layers: list of number of nodes in each layer
    activations: list of activation functions for each layer
    lambtha: L2 regularization parameter
    keep_prob: dropout rate
    """
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=K.regularizers.l2(lambtha))(x)
        if i != len(layers) - 1 and keep_prob is not None:
            x = K.layers.Dropout(1-keep_prob)(x)
    return K.Model(inputs, x)
