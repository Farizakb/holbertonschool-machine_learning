#!/usr/bin/env python3
"""
    Optimize
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    network: model to optimize
    alpha: learning rate
    beta1: Adam optimizer parameter
    beta2: Adam optimizer parameter
    """
    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
