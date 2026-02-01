#!/usr/bin/env python3
"""
Module to calculate the definiteness of a matrix
"""
import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix:
    - matrix is a numpy.ndarray whose definiteness should be calculated
    - If matrix is not a numpy.ndarray, raise a TypeError with the message
      matrix must be a numpy.ndarray
    - If matrix is not symmetric, return None
    - Returns: Positive definite, Positive semi-definite, Negative definite,
      Negative semi-definite, or Indefinite
    - If matrix does not fit any of the above categories, return None
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if the matrix is square and symmetric
    if (len(matrix.shape) != 2 or
            matrix.shape[0] != matrix.shape[1] or
            matrix.size == 0):

        return None
    if not np.allclose(matrix, matrix.T):
        return None

    try:
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(matrix)

        pos_count = np.sum(eigenvalues > 1e-10)
        neg_count = np.sum(eigenvalues < -1e-10)
        zero_count = np.sum(np.abs(eigenvalues) <= 1e-10)
        n = len(eigenvalues)

        if pos_count == n:
            return "Positive definite"
        if neg_count == n:
            return "Negative definite"
        if pos_count > 0 and neg_count > 0:
            return "Indefinite"
        if pos_count > 0 and zero_count > 0:
            return "Positive semi-definite"
        if neg_count > 0 and zero_count > 0:
            return "Negative semi-definite"

        return None
    except Exception:
        return None
