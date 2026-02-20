#!/usr/bin/env python3
"""
    F1 Score
"""

import numpy as np


def f1_score(confusion):
    """
        calculates the f1 score for each class in a confusion matrix

        :param confusion: ndarray, shape(classes,classes), matrix confusion

        :return: ndarray, shape(classes,) containing f1 score of each class
    """
    # number of classes
    classes = confusion.shape[0]
    # initialize f1 score
    f1_score_matrix = np.zeros((classes,))

    for i in range(classes):
        true_positive = confusion[i, i]
        # sum along the column
        total_positives = np.sum(confusion[:, i])
        false_negatives = np.sum(confusion[i, :]) - true_positive
        total_negatives = np.sum(confusion) - total_positives - false_negatives
        f1_score_matrix[i] = 2 * (true_positive / (true_positive + false_negatives)) * (true_positive / (true_positive + false_positives)) / (true_positive / (true_positive + false_negatives) + true_positive / (true_positive + false_positives))

    return f1_score_matrix
