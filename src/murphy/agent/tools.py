from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from murphy.features import bid_ask_spread_fraction, realized_volatility, trailing_return
from murphy.priors import delta_as_probability, risk_neutral_call_itm_probability
from murphy.schemas import OptionExample, OptionSnapshot, UnderlyingBar


@dataclass(frozen=True)
class ToolObservation:
    name: str
    payload: dict[str, Any]


class OptionToolStore(Protocol):
    def get_option_snapshot(self, example_id: str) -> OptionSnapshot: ...

    def get_underlying_history(self, symbol: str, end_timestamp: object, lookback_days: int) -> list[UnderlyingBar]: ...


class InMemoryToolStore:
    def __init__(
        self,
        snapshots: dict[str, OptionSnapshot],
        histories: dict[str, list[UnderlyingBar]],
    ) -> None:
        self.snapshots = snapshots
        self.histories = histories

    def get_option_snapshot(self, example_id: str) -> OptionSnapshot:
        return self.snapshots[example_id]

    def get_underlying_history(
        self,
        symbol: str,
        end_timestamp: object,
        lookback_days: int,
    ) -> list[UnderlyingBar]:
        del lookback_days
        return [bar for bar in self.histories.get(symbol, []) if bar.timestamp <= end_timestamp]


class OptionTools:
    def __init__(self, store: OptionToolStore) -> None:
        self.store = store

    def fetch_option_snapshot(self, example: OptionExample) -> ToolObservation:
        snapshot = self.store.get_option_snapshot(example.example_id)
        spread = bid_ask_spread_fraction(snapshot.bid, snapshot.ask, snapshot.mid)
        return ToolObservation(
            name="fetch_option_snapshot",
            payload={
                "symbol": snapshot.symbol,
                "option_symbol": snapshot.option_symbol,
                "quote_timestamp": snapshot.quote_timestamp.isoformat(),
                "expiration": snapshot.expiration.isoformat(),
                "strike": snapshot.strike,
                "spot": snapshot.spot,
                "dte": example.dte,
                "moneyness": example.moneyness,
                "bid": snapshot.bid,
                "ask": snapshot.ask,
                "mid": snapshot.mid,
                "implied_volatility": snapshot.implied_volatility,
                "delta": snapshot.delta,
                "bid_ask_spread_fraction": spread,
                "volume": snapshot.volume,
                "open_interest": snapshot.open_interest,
            },
        )

    def compute_underlying_features(
        self,
        example: OptionExample,
        lookback_days: int = 30,
    ) -> ToolObservation:
        bars = self.store.get_underlying_history(
            example.symbol,
            example.forecast_timestamp,
            lookback_days,
        )
        return ToolObservation(
            name="compute_underlying_features",
            payload={
                "symbol": example.symbol,
                "lookback_days": lookback_days,
                "bar_count": len(bars),
                "realized_volatility": realized_volatility(bars),
                "trailing_return": trailing_return(bars),
                "last_close": None if not bars else bars[-1].close,
            },
        )

    def compute_market_prior(self, example: OptionExample) -> ToolObservation:
        snapshot = self.store.get_option_snapshot(example.example_id)
        probability = None
        source = None

        if snapshot.implied_volatility is not None:
            probability = risk_neutral_call_itm_probability(
                spot=example.spot,
                strike=example.strike,
                dte=example.dte,
                volatility=snapshot.implied_volatility,
            )
            source = "black_scholes_d2"
        elif snapshot.delta is not None:
            probability = delta_as_probability(snapshot.delta)
            source = "call_delta"

        return ToolObservation(
            name="compute_market_prior",
            payload={
                "probability": probability,
                "source": source,
                "note": "Risk-neutral or delta-derived prior; not a physical probability.",
            },
        )

