#!/usr/bin/env python3
"""
   Moving Average
"""

import numpy as np


def moving_average(data, beta):
    """
        Method that calculates the weighted moving average of
        a data set

        Formul:
        MA = (val1 + val2 +val3 + ... + valN) / N

        :param data: ndarray, shape (t,) to calculate moving average
        :param beta: float, weight used for moving average

        :return: ndarray, shape (t,) containing the moving averages of data
    """
    v = 0
    v_history = []
    for t, x in enumerate(data):
        v = beta * v + (1 - beta) * x
        v_corrected = v / (1 - beta ** (t + 1))
        v_history.append(v_corrected)
    return v_history
