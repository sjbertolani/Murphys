from __future__ import annotations

import numpy as np

from murphy.baselines import chronological_split, empirical_base_rate_predictions


def test_chronological_split_keeps_ordered_train_before_test() -> None:
    timestamps = np.array([3, 1, 2, 4, 5])
    train_idx, test_idx = chronological_split(timestamps, test_fraction=0.4)

    assert timestamps[train_idx].max() <= timestamps[test_idx].min()
    assert len(train_idx) == 3
    assert len(test_idx) == 2


def test_empirical_base_rate_predictions_laplace_smoothing() -> None:
    y_train = np.array([1, 1, 0, 0])
    predictions = empirical_base_rate_predictions(y_train, n_test=3)

    assert predictions.tolist() == [0.5, 0.5, 0.5]

