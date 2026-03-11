#!/usr/bin/env python3
"""
    Test model function
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
        function that tests a model

        :param network: model to test
        :param data: ndarray, shape(m, nx), input data
        :param labels: ndarray, shape(m,classes), labels
        :param verbose: boolean, print or not during testing

        :return: tuple, (loss, accuracy)
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)