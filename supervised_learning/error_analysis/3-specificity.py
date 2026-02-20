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
    # total sum of all elements in confusion matrix
    total = np.sum(confusion)

    for i in range(classes):
        # elements correctly predicted as i
        tp = confusion[i, i]
        # elements incorrectly predicted as i (actual is not i)
        fp = np.sum(confusion[:, i]) - tp
        # elements incorrectly predicted as not i (actual is i)
        fn = np.sum(confusion[i, :]) - tp
        # elements correctly predicted as not i (actual is not i)
        tn = total - (tp + fp + fn)

        # specificity = TN / (TN + FP)
        specificity_matrix[i] = tn / (tn + fp)

    return specificity_matrix
