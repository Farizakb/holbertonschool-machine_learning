#!/usr/bin/env python3
"""
    Function to normalize the data
"""

import numpy as np


def normalize(X, m, s):
    """
        Method to normalize the data

        :param X: ndarray, shape(m,nx) to normalize
                m : number of data points
                nx: number of features
        :param m: ndarray, shape(nx,) containing the mean of each feature
        :param s: ndarray, shape(nx,) containing the standard deviation of each feature

        :return: normalized data
    """
    return (X - m) / s
