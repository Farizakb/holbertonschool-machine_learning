#!/usr/bin/env python3
"""
    Save and load model configuration function
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
        function that saves the configuration of a neural network

        :param network: model to save configuration from
        :param filename: path where configuration saved to

        :return: None
    """
    network.save_config(filename)


def load_config(filename):
    """
        function that loads the configuration of a neural network

        :param filename: path where configuration loaded from

        :return: None
    """
    network.load_config(filename)
