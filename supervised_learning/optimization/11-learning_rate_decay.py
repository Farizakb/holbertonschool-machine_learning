#!/usr/bin/env python3
"""
    Learning Rate Decay
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Calculates the learning rate after a decay schedule.

    :param alpha: float, initial learning rate
    :param decay_rate: float, decay rate
    :param global_step: int, current iteration number
    :param decay_step: int, number of iterations after which to decay the
    learning rate
    :return: float, new learning rate
    """
    return alpha * (1 / (1 + decay_rate * (global_step // decay_step)))
