#!/usr/bin/env python3
"""
DeepNeuralNetwork performing binary classification
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Class that represents a deep neural network for binary classification
    """

    def __init__(self, nx, layers):
        """
        Class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        weights = {}
        previous = nx

        for index, layer in enumerate(layers, 1):
            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")

            weights["b{}".format(index)] = np.zeros((layer, 1))
            weights["W{}".format(index)] = (
                np.random.randn(layer, previous) * np.sqrt(2 / previous))
            previous = layer

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = weights

    @property
    def L(self):
        """
        gets the private instance attribute __L
        """
        return self.__L

    @property
    def cache(self):
        """
        gets the private instance attribute __cache
        """
        return self.__cache

    @property
    def weights(self):
        """
        gets the private instance attribute __weights
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Performs the forward propagation for the deep neural network
        """
        self.__cache["A0"] = X
        for i in range(1, self.L + 1):
            A_prev = self.__cache["A{}".format(i - 1)]
            W = self.weights["W{}".format(i)]
            b = self.weights["b{}".format(i)]
            Z = np.dot(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A{}".format(i)] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the neural network
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return np.squeeze(cost)

    def evaluate(self, X, Y):
        """
        Evaluates the neural network
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        pred = np.where(A >= 0.5, 1, 0)
        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        back = {}

        for i in range(self.L, 0, -1):
            A = cache["A{}".format(i)]
            A_prev = cache["A{}".format(i - 1)]

            if i == self.L:
                dz = A - Y
            else:
                W_next = self.weights["W{}".format(i + 1)]
                dz = np.dot(W_next.T, dz) * A * (1 - A)

            dw = (1 / m) * np.dot(dz, A_prev.T)
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            back["dw{}".format(i)] = dw
            back["db{}".format(i)] = db

        for i in range(self.L, 0, -1):
            self.__weights["W{}".format(i)] -= alpha * back["dw{}".format(i)]
            self.__weights["b{}".format(i)] -= alpha * back["db{}".format(i)]
