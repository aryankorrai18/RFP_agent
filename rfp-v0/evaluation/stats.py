"""Small, dependency-free statistics: every number in the report comes with an interval, and every comparison is paired."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(rate, low, high): a 95% Wilson interval for a proportion. Sound at small n and near 0 or 1."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean(values: Sequence[float], seed: int = 0, iters: int = 4000) -> tuple[float, float, float]:
    """(mean, low, high): a 95% percentile bootstrap interval of the mean."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(values) / n, means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def paired_difference(a: Sequence[float], b: Sequence[float], seed: int = 0, iters: int = 4000) -> tuple[float, float, float]:
    """(mean(a-b), low, high) over the same cases, a 95% bootstrap interval. a and b are matched element by element."""
    if len(a) != len(b):
        raise ValueError("paired samples must be the same length")
    return bootstrap_mean([x - y for x, y in zip(a, b, strict=True)], seed, iters)


def sign_test(a: Sequence[bool], b: Sequence[bool]) -> tuple[int, int, float]:
    """(a-only, b-only, two-sided exact p): McNemar's exact test on the cases where the two systems disagree."""
    a_only = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    b_only = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = a_only + b_only
    if n == 0:
        return 0, 0, 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return a_only, b_only, min(1.0, 2 * tail)


def percent(x: float) -> str:
    return f"{100 * x:.0f}%"


def interval(rate: float, low: float, high: float) -> str:
    return f"{percent(rate)} ({percent(low)} to {percent(high)})"
