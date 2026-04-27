from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from murphy.market_data.provider import MarketDataProvider
from murphy.market_data.yahoo import YahooFinanceProvider
from murphy.schemas import OptionSnapshot, UnderlyingBar


@dataclass(frozen=True)
class CollectionResult:
    provider: str
    collected_at: datetime
    underlying_bars: int
    option_snapshots: int


def provider_from_name(name: str) -> MarketDataProvider:
    normalized = name.lower()
    if normalized in {"yahoo", "yfinance"}:
        return YahooFinanceProvider()
    raise ValueError(f"Unsupported market data provider: {name}")


def collect_market_snapshots(
    provider: MarketDataProvider,
    tickers: list[str],
    min_dte: int,
    max_dte: int,
    lookback_days: int,
    as_of: datetime | None = None,
) -> tuple[list[UnderlyingBar], list[OptionSnapshot], CollectionResult]:
    as_of = _utc_now_if_none(as_of)
    normalized_tickers = sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})
    if not normalized_tickers:
        raise ValueError("At least one ticker is required")
    bars = provider.fetch_underlying_bars(normalized_tickers, lookback_days=lookback_days, as_of=as_of)
    snapshots = provider.fetch_option_chain_snapshots(
        normalized_tickers,
        min_dte=min_dte,
        max_dte=max_dte,
        as_of=as_of,
    )
    return (
        bars,
        snapshots,
        CollectionResult(
            provider=provider.name,
            collected_at=as_of,
            underlying_bars=len(bars),
            option_snapshots=len(snapshots),
        ),
    )


def _utc_now_if_none(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)

