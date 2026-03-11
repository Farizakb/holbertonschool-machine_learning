#!/usr/bin/env python3
"""
    One hot
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    labels: numpy.ndarray with shape (m,)
    classes: number of classes
    """
    return K.utils.to_categorical(labels, classes)
