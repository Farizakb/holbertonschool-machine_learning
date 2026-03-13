#!/usr/bin/env python3
"""
    Function to create mini batches
"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches for training.

    :param X: np.ndarray, shape (m, nx)
    :param Y: np.ndarray, shape (m,)
    :param batch_size: int, size of each mini-batch
    :return: list of tuples (X_batch, Y_batch)
    """
    m = X.shape[0]
    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    mini_batches = []
    num_batches = m // batch_size

    for i in range(num_batches):
        X_batch = X_shuffled[i * batch_size:(i + 1) * batch_size]
        Y_batch = Y_shuffled[i * batch_size:(i + 1) * batch_size]
        mini_batches.append((X_batch, Y_batch))

    # Handle the last batch if m is not divisible by batch_size
    if m % batch_size != 0:
        X_batch = X_shuffled[num_batches * batch_size:]
        Y_batch = Y_shuffled[num_batches * batch_size:]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
