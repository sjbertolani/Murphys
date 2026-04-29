from __future__ import annotations

from datetime import datetime
from typing import Protocol

from murphy.schemas import OptionSnapshot, UnderlyingBar


class MarketDataProvider(Protocol):
    name: str

    def fetch_underlying_bars(
        self,
        tickers: list[str],
        lookback_days: int,
        as_of: datetime | None = None,
    ) -> list[UnderlyingBar]: ...

    def fetch_option_chain_snapshots(
        self,
        tickers: list[str],
        min_dte: float,
        max_dte: float,
        as_of: datetime | None = None,
    ) -> list[OptionSnapshot]: ...
