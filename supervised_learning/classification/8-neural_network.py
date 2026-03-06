#!/usr/bin/env python3
"""
    Class NeuralNetwork
"""

import numpy as np


class NeuralNetwork:
    """
        Class NeuralNetwork : define neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of integers")
        if len(layers) == 0:
            raise ValueError("layers must not be empty")
        for i in layers:
            if type(i) is not int:
                raise TypeError("layers must be a list of integers")
            if i < 1:
                raise ValueError("layers must be a list of positive integers")
        self.__W1 = np.random.randn(layers[0], nx) * 0.01
        self.__b1 = np.zeros((layers[0], 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(layers[1], layers[0]) * 0.01
        self.__b2 = np.zeros((layers[1], 1))
        self.__A2 = 0
