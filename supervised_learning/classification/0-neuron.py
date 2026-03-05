#!/usr/bin/env python3
"""
0. Neuron
"""

import numpy as np

class Neuron:
    """
        Class Neuron : define single neuron performing binary classification
    """

    def __init__(self, nx):

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx <=0:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(nx, 1) * 0.01
        self.b = 0
        self.A = 0
