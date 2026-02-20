#!/usr/bin/env python3
"""
    Specificity
"""

import numpy as np


def specificity(confusion):
    """
        calculates the specificity for each class in a confusion matrix

        :param confusion: ndarray, shape(classes,classes), matrix confusion

        :return: ndarray, shape(classes,) containing specificity of each class
    """
    # number of classes
    classes = confusion.shape[0]
    # initialize specificity
    specificity_matrix = np.zeros((classes,))

    for i in range(classes):
        true_positive = confusion[i, i]
        # sum along the column
        total_positives = np.sum(confusion[:, i])

        specificity_matrix[i] = true_positive / total_positives

    return specificity_matrix
