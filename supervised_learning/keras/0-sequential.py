#!/usr/bin/env python3
"""
    Sequential
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
    model = K.Sequential()
    model.add(K.layers.Dense(layers[0], input_dim=nx, activation=activations[0], kernel_regularizer=K.regularizers.l2(lambtha)))
    for i in range(1, len(layers)):
        model.add(K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=K.regularizers.l2(lambtha)))
        if i != len(layers) - 1 and keep_prob is not None:
            model.add(K.layers.Dropout(1-keep_prob))
    return model
