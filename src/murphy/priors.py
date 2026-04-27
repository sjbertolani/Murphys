from __future__ import annotations

import math

SQRT_2 = math.sqrt(2.0)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def clamp_probability(probability: float, eps: float = 1e-6) -> float:
    return min(max(probability, eps), 1.0 - eps)


def logit(probability: float) -> float:
    probability = clamp_probability(probability)
    return math.log(probability / (1.0 - probability))


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def bayesian_binary_update(
    prior_probability: float,
    signal_probability: float,
    signal_weight: float = 1.0,
) -> float:
    """Update a binary prior using a probability signal as log-odds evidence.

    The signal is interpreted relative to a neutral 0.5 baseline. A signal of
    0.5 leaves the prior unchanged; a signal above/below 0.5 moves the posterior
    by `signal_weight` times the signal log-odds.
    """
    if signal_weight < 0:
        raise ValueError("signal_weight must be non-negative")
    posterior_log_odds = logit(prior_probability) + signal_weight * logit(signal_probability)
    return clamp_probability(logistic(posterior_log_odds))


def black_scholes_d2(
    spot: float,
    strike: float,
    time_to_expiration_years: float,
    volatility: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
) -> float:
    if spot <= 0:
        raise ValueError("spot must be positive")
    if strike <= 0:
        raise ValueError("strike must be positive")
    if time_to_expiration_years <= 0:
        return math.inf if spot > strike else -math.inf
    if volatility <= 0:
        return math.inf if spot > strike else -math.inf

    numerator = math.log(spot / strike) + (
        risk_free_rate - dividend_yield - 0.5 * volatility**2
    ) * time_to_expiration_years
    denominator = volatility * math.sqrt(time_to_expiration_years)
    return numerator / denominator


def risk_neutral_call_itm_probability(
    spot: float,
    strike: float,
    dte: float,
    volatility: float,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
    trading_days_per_year: float = 252.0,
) -> float:
    """Black-Scholes risk-neutral P(S_T > K), equal to N(d2)."""
    t = max(dte / trading_days_per_year, 0.0)
    d2 = black_scholes_d2(
        spot=spot,
        strike=strike,
        time_to_expiration_years=t,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    return clamp_probability(normal_cdf(d2))


def delta_as_probability(delta: float | None) -> float | None:
    """Use call delta as a rough, biased probability prior when no IV model is available."""
    if delta is None:
        return None
    return clamp_probability(delta)


def empirical_bucket_prior(successes: int, total: int, alpha: float = 1.0, beta: float = 1.0) -> float:
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("invalid successes/total")
    return (successes + alpha) / (total + alpha + beta)
