#!/usr/bin/env python3
"""
    Save and load weights function
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
        function that saves the weights of a neural network

        :param network: model to save weights from
        :param filename: path where weights saved to
        :param save_format: format to save weights in

        :return: None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
        function that loads the weights of a neural network

        :param network: model to load weights into
        :param filename: path where weights loaded from

        :return: None
    """
    network.load_weights(filename)
