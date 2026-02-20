#!/usr/bin/env python3
"""
    Precision
"""

import numpy as np


def precision(confusion):
    """
        calculates the precision for each class in a confusion matrix

        :param confusion: ndarray, shape(classes,classes), matrix confusion

        :return: ndarray, shape(classes,) containing precision of each class
    """
    # number of classes
    classes = confusion.shape[0]
    # initialize precision
    precision_matrix = np.zeros((classes,))

    for i in range(classes):
        true_positive = confusion[i, i]
        # sum along the column
        total_positives = np.sum(confusion[:, i])

        precision_matrix[i] = true_positive / total_positives

    return precision_matrix
