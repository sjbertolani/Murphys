from __future__ import annotations

import math
import unittest

from murphy.agent.aggregation import arithmetic_mean, logit_mean, variance_adaptive_shrinkage
from murphy.features import log_returns, realized_volatility, simple_returns
from murphy.metrics import brier_score, expected_calibration_error, log_loss
from murphy.priors import empirical_bucket_prior, normal_cdf, risk_neutral_call_itm_probability


class MathCoreTests(unittest.TestCase):
    def test_returns(self) -> None:
        closes = [100.0, 110.0, 99.0]
        returns = simple_returns(closes)
        self.assertAlmostEqual(returns[0], 0.1)
        self.assertAlmostEqual(returns[1], -0.1)
        self.assertAlmostEqual(log_returns(closes)[0], math.log(1.1))

    def test_realized_volatility_none_for_short_history(self) -> None:
        self.assertIsNone(realized_volatility([]))

    def test_normal_cdf(self) -> None:
        self.assertAlmostEqual(normal_cdf(0.0), 0.5)
        self.assertGreater(normal_cdf(1.0), 0.8)

    def test_risk_neutral_probability_near_atm(self) -> None:
        p = risk_neutral_call_itm_probability(
            spot=100.0,
            strike=100.0,
            dte=30.0,
            volatility=0.3,
        )
        self.assertGreater(p, 0.4)
        self.assertLess(p, 0.6)

    def test_empirical_prior(self) -> None:
        self.assertAlmostEqual(empirical_bucket_prior(2, 4), 0.5)

    def test_aggregation(self) -> None:
        probabilities = [0.4, 0.6]
        self.assertAlmostEqual(arithmetic_mean(probabilities), 0.5)
        self.assertAlmostEqual(logit_mean(probabilities), 0.5)
        self.assertGreaterEqual(variance_adaptive_shrinkage(probabilities), 0.0)

    def test_metrics(self) -> None:
        probabilities = [0.1, 0.9]
        labels = [0, 1]
        self.assertAlmostEqual(brier_score(probabilities, labels), 0.01)
        self.assertLess(log_loss(probabilities, labels), 0.2)
        self.assertLess(expected_calibration_error(probabilities, labels, bins=2), 0.2)


if __name__ == "__main__":
    unittest.main()
