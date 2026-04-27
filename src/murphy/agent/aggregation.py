from __future__ import annotations

import math
from collections.abc import Sequence

from murphy.priors import clamp_probability


def logit(probability: float) -> float:
    p = clamp_probability(probability)
    return math.log(p / (1.0 - p))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def arithmetic_mean(probabilities: Sequence[float]) -> float:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    return sum(probabilities) / len(probabilities)


def logit_mean(probabilities: Sequence[float], shrinkage: float = 1.0) -> float:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    if shrinkage < 0 or shrinkage > 1:
        raise ValueError("shrinkage must be in [0, 1]")
    mean_logit = sum(logit(p) for p in probabilities) / len(probabilities)
    return clamp_probability(sigmoid(shrinkage * mean_logit))


def variance_adaptive_shrinkage(
    probabilities: Sequence[float],
    floor: float = 0.25,
    scale: float = 0.1,
) -> float:
    """Simple paper-inspired shrinkage coefficient based on trial logit disagreement."""
    if len(probabilities) < 2:
        return 1.0
    logits = [logit(p) for p in probabilities]
    mean = sum(logits) / len(logits)
    variance = sum((value - mean) ** 2 for value in logits) / (len(logits) - 1)
    std = math.sqrt(variance)
    return min(max(1.0 - scale * std, floor), 1.0)

