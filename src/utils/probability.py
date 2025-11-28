import numpy as np
from itertools import combinations
from typing import List


def poisson_binomial_pmf(probabilities: List[float], k: int) -> float:
    """
    Compute the probability mass function for exactly k successes
    in a Poisson Binomial distribution.

    >>> abs(poisson_binomial_pmf([0.1, 0.2], 2) - 0.02) < 1e-6
    True
    >>> abs(poisson_binomial_pmf([0.1, 0.2, 0.9], 2) - 0.236) < 1e-6
    True
    >>> abs(poisson_binomial_pmf([0.1, 0.2, 0.3], 2) - 0.092) < 1e-6
    True
    """
    n = len(probabilities)
    if k > n:
        return 0.0
    dp = np.zeros(k+1)
    dp[0] = 1.0
    for p in probabilities:
        dp[1:] = dp[1:] * (1 - p) + dp[:-1] * p
        dp[0] *= (1 - p)
    return dp[k]


def poisson_binomial_cdf(probabilities: List[float], start: int) -> float:
    """
    Compute the cumulative distribution function for `at least` k successes
    in a Poisson Binomial distribution.

    >>> abs(poisson_binomial_cdf([0.1, 0.2], 2) - 0.02) < 1e-6
    True
    >>> abs(poisson_binomial_cdf([0.1, 0.2, 0.9], 2) - 0.236) < 1e-6
    False
    >>> abs(poisson_binomial_cdf([0.1, 0.2, 0.3], 2) - 0.092) < 1e-6
    False
    >>> abs(poisson_binomial_cdf([0.1, 0.2, 0.9], 2) - 0.254) < 1e-6
    True
    >>> abs(poisson_binomial_cdf([0.1, 0.2, 0.3], 2) - 0.098) < 1e-6
    True
    >>> abs(poisson_binomial_cdf([0.1, 0.2, 0.9, 0.3], 2) - 0.4562) < 1e-6
    True
    """
    end = len(probabilities)
    return np.sum([poisson_binomial_pmf(probabilities, i) for i in range(start, end+1)])


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
