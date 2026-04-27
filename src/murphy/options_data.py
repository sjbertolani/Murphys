from __future__ import annotations

from datetime import datetime, timezone

from .schemas import OptionExample, OptionSnapshot, UnderlyingBar

SECONDS_PER_DAY = 24 * 60 * 60


def days_to_expiration(forecast_timestamp: datetime, expiration: datetime) -> float:
    """Return non-negative days from forecast time to expiration."""
    delta_seconds = (expiration - forecast_timestamp).total_seconds()
    return max(delta_seconds / SECONDS_PER_DAY, 0.0)


def moneyness(spot: float, strike: float) -> float:
    """Return S/K - 1, so ATM is near 0."""
    if strike <= 0:
        raise ValueError("strike must be positive")
    return (spot / strike) - 1.0


def is_near_money(spot: float, strike: float, tolerance: float = 0.03) -> bool:
    return abs(moneyness(spot, strike)) <= tolerance


def call_finished_itm(close_at_expiration: float, strike: float) -> int:
    return int(close_at_expiration > strike)


def build_option_example(
    snapshot: OptionSnapshot,
    close_at_expiration: float | None = None,
    resolution_timestamp: datetime | None = None,
) -> OptionExample:
    """Build a labeled or unlabeled call-option binary example from one snapshot."""
    if snapshot.spot is None:
        raise ValueError("snapshot.spot is required to build an example")

    label = None
    if close_at_expiration is not None:
        label = call_finished_itm(close_at_expiration, snapshot.strike)

    return OptionExample(
        example_id=make_example_id(
            snapshot.symbol,
            snapshot.option_symbol,
            snapshot.quote_timestamp,
        ),
        symbol=snapshot.symbol,
        option_symbol=snapshot.option_symbol,
        forecast_timestamp=snapshot.quote_timestamp,
        expiration=snapshot.expiration,
        strike=snapshot.strike,
        spot=snapshot.spot,
        dte=days_to_expiration(snapshot.quote_timestamp, snapshot.expiration),
        moneyness=moneyness(snapshot.spot, snapshot.strike),
        label=label,
        resolution_timestamp=resolution_timestamp,
    )


def make_example_id(symbol: str, option_symbol: str, forecast_timestamp: datetime) -> str:
    ts = forecast_timestamp.astimezone(timezone.utc).isoformat()
    safe_ts = ts.replace(":", "").replace("+", "p")
    return f"{symbol}:{option_symbol}:{safe_ts}"


def latest_bar_at_or_before(
    bars: list[UnderlyingBar],
    timestamp: datetime,
) -> UnderlyingBar | None:
    eligible = [bar for bar in bars if bar.timestamp <= timestamp]
    if not eligible:
        return None
    return max(eligible, key=lambda bar: bar.timestamp)

