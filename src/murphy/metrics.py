from __future__ import annotations

import math
from collections.abc import Sequence

from .priors import clamp_probability


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    _validate_lengths(probabilities, labels)
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels, strict=True)) / len(labels)


def log_loss(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    _validate_lengths(probabilities, labels)
    total = 0.0
    for probability, label in zip(probabilities, labels, strict=True):
        p = clamp_probability(probability)
        total += -(label * math.log(p) + (1 - label) * math.log(1 - p))
    return total / len(labels)


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int],
    bins: int = 10,
) -> float:
    _validate_lengths(probabilities, labels)
    if bins <= 0:
        raise ValueError("bins must be positive")

    total = len(labels)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        in_bin = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= probability < upper or (bin_index == bins - 1 and probability == 1.0)
        ]
        if not in_bin:
            continue
        confidence = sum(probabilities[index] for index in in_bin) / len(in_bin)
        accuracy = sum(labels[index] for index in in_bin) / len(in_bin)
        ece += (len(in_bin) / total) * abs(accuracy - confidence)
    return ece


def _validate_lengths(probabilities: Sequence[float], labels: Sequence[int]) -> None:
    if not probabilities:
        raise ValueError("probabilities must not be empty")
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have same length")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be binary")

