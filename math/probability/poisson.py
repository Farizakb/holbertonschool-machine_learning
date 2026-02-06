#!/usr/bin/env python3
"""
Poisson distribution module
"""


class Poisson:
    """
    Class that represents a poisson distribution
    """

    def __init__(self, data=None, lambtha=1.0):
        """
        Initialize Poisson distribution
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate the mean of the data to set lambtha
            self.lambtha = float(sum(data) / len(data))

    def factorial(self, n):
        """
        Helper method to calculate factorial of n
        """
        if n < 0:
            return None
        res = 1
        for i in range(1, int(n) + 1):
            res *= i
        return res

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of 'successes'
        """
        k = int(k)
        if k < 0:
            return 0

        # Using the constant e provided in your requirements
        e = 2.7182818285

        # Formula: (e^-lambda * lambda^k) / k!
        numerator = (e ** -self.lambtha) * (self.lambtha ** k)
        denominator = self.factorial(k)

        return numerator / denominator

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of 'successes'
        """
        k = int(k)
        if k < 0:
            return 0

        # CDF is the sum of PMFs from 0 to k
        result = 0
        for i in range(k + 1):
            result += self.pmf(i)
        return result
