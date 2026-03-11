#!/usr/bin/env python3
"""
    Predict function
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
        function that predicts with a model

        :param network: model to predict with
        :param data: ndarray, shape(m, nx), input data
        :param verbose: boolean, print or not during prediction

        :return: ndarray, shape(m, classes), predictions
    """
    return network.predict(x=data, verbose=verbose)