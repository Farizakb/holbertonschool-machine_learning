#!/usr/bin/env python3
"""
    Function to shuffle the data
"""

import numpy as np


def shuffle_data(X, Y):
    """
        Method to shuffle the data

        :param X: ndarray, shape(m,nx) to shuffle
                m : number of data points
                nx: number of features
        :param Y: ndarray, shape(m,) to shuffle

        :return: shuffled data
    """
    return np.random.permutation(X), np.random.permutation(Y)
