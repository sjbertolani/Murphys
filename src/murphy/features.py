from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import UnderlyingBar


def simple_returns(closes: Sequence[float]) -> list[float]:
    returns: list[float] = []
    for previous, current in zip(closes, closes[1:], strict=False):
        if previous <= 0:
            raise ValueError("close prices must be positive")
        returns.append((current / previous) - 1.0)
    return returns


def log_returns(closes: Sequence[float]) -> list[float]:
    returns: list[float] = []
    for previous, current in zip(closes, closes[1:], strict=False):
        if previous <= 0 or current <= 0:
            raise ValueError("close prices must be positive")
        returns.append(math.log(current / previous))
    return returns


def realized_volatility(
    bars: Sequence["UnderlyingBar"],
    annualization: float = 252.0,
) -> float | None:
    """Annualized realized volatility from close-to-close log returns."""
    if len(bars) < 3:
        return None
    returns = log_returns([bar.close for bar in bars])
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return math.sqrt(variance * annualization)


def trailing_return(bars: Sequence["UnderlyingBar"]) -> float | None:
    if len(bars) < 2:
        return None
    start = bars[0].close
    end = bars[-1].close
    if start <= 0:
        raise ValueError("close prices must be positive")
    return (end / start) - 1.0


def bid_ask_spread_fraction(bid: float | None, ask: float | None, mid: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    reference = mid if mid and mid > 0 else (bid + ask) / 2.0
    if reference <= 0:
        return None
    return (ask - bid) / reference


def zscore(value: float, mean: float, std: float) -> float:
    if std <= 0:
        raise ValueError("std must be positive")
    return (value - mean) / std
