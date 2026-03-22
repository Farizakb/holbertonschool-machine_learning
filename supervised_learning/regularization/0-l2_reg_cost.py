#!/usr/bin/env python3
"""
    L2 regularization
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
        Function that calculates the cost of a network with L2 Regularization

        FORMULA = loss + lamda/2m * sum||w||**2
    """
    reg_term = 0

    for i in range(1, L + 1):
        reg_term += np.sum(np.square(weights['W' + str(i)]))
    cost_L2 = cost + (lambtha / (2 * m)) * reg_term

    return cost_L2
