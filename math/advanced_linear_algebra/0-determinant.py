#!/usr/bin/env python3
"""
Module to calculate the determinant of a matrix
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix:
    - matrix is a list of lists whose determinant should be calculated
    - If matrix is not a list of lists, raise a TypeError with the message
      matrix must be a list of lists
    - If matrix is not square, raise a ValueError with the message
      matrix must be a square matrix
    - The list [[]] represents a 0x0 matrix
    - Returns: the determinant of matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        if matrix == []:
            return 1
        raise TypeError("matrix must be a list of lists")

    # Special case for 0x0 matrix represented as [[]]
    if matrix == [[]]:
        return 1

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    # Base case for 1x1 matrix
    if n == 1:
        return matrix[0][0]

    # Base case for 2x2 matrix
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    def get_minor(m, j):
        """
        Helper to get the minor of a matrix after removing first row and col j
        """
        return [row[:j] + row[j+1:] for row in m[1:]]

    det = 0
    for j in range(n):
        det += ((-1) ** j) * matrix[0][j] * determinant(get_minor(matrix, j))

    return det

