from __future__ import annotations

from dataclasses import dataclass

from murphy.agent.aggregation import logit, sigmoid
from murphy.priors import clamp_probability


@dataclass(frozen=True)
class PlattCalibration:
    """A minimal Platt transform container.

    Fit this later with scikit-learn or scipy. Keeping the transform separate makes
    calibration easy to serialize and test.
    """

    slope: float = 1.0
    intercept: float = 0.0
    group_offsets: dict[str, float] | None = None

    def transform_one(self, probability: float, group: str | None = None) -> float:
        offset = 0.0
        if group is not None and self.group_offsets is not None:
            offset = self.group_offsets.get(group, 0.0)
        return clamp_probability(sigmoid(self.slope * logit(probability) + self.intercept + offset))

    def transform(self, probabilities: list[float], groups: list[str] | None = None) -> list[float]:
        if groups is not None and len(groups) != len(probabilities):
            raise ValueError("groups must match probabilities length")
        return [
            self.transform_one(probability, None if groups is None else groups[index])
            for index, probability in enumerate(probabilities)
        ]

