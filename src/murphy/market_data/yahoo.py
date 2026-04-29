from __future__ import annotations

import math
from datetime import UTC, datetime, time
from typing import Any

from murphy.options_data import days_to_expiration
from murphy.schemas import OptionRight, OptionSnapshot, UnderlyingBar


class YahooFinanceProvider:
    """Yahoo/yfinance market data provider.

    Yahoo is useful as a low-cost live collector. It is not an official SLA-backed
    data feed, so we persist every snapshot we use for forecasting.
    """

    name = "yahoo"

    def fetch_underlying_bars(
        self,
        tickers: list[str],
        lookback_days: int,
        as_of: datetime | None = None,
    ) -> list[UnderlyingBar]:
        import yfinance as yf

        as_of = _utc_now_if_none(as_of)
        period = f"{max(lookback_days + 5, 7)}d"
        bars: list[UnderlyingBar] = []
        for ticker in tickers:
            frame = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
            if frame is None or frame.empty:
                continue
            for index, row in frame.iterrows():
                timestamp = _to_utc_datetime(index)
                if timestamp > as_of:
                    continue
                close = _clean_float(row.get("Close"))
                if close is None:
                    continue
                bars.append(
                    UnderlyingBar(
                        symbol=ticker.upper(),
                        timestamp=timestamp,
                        open=_clean_float(row.get("Open")) or close,
                        high=_clean_float(row.get("High")) or close,
                        low=_clean_float(row.get("Low")) or close,
                        close=close,
                        volume=_clean_float(row.get("Volume")),
                        adjusted_close=_clean_float(row.get("Adj Close")),
                    )
                )
        return bars

    def fetch_option_chain_snapshots(
        self,
        tickers: list[str],
        min_dte: float,
        max_dte: float,
        as_of: datetime | None = None,
    ) -> list[OptionSnapshot]:
        import yfinance as yf

        as_of = _utc_now_if_none(as_of)
        snapshots: list[OptionSnapshot] = []
        for ticker in tickers:
            yf_ticker = yf.Ticker(ticker)
            spot = _fetch_spot(yf_ticker)
            if spot is None:
                continue
            expirations = list(getattr(yf_ticker, "options", []) or [])
            for expiration_text in expirations:
                expiration = _expiration_to_utc(expiration_text)
                dte = days_to_expiration(as_of, expiration)
                if dte < min_dte or dte > max_dte:
                    continue
                try:
                    chain = yf_ticker.option_chain(expiration_text)
                except Exception:
                    continue
                snapshots.extend(
                    _rows_to_snapshots(
                        ticker=ticker.upper(),
                        quote_timestamp=as_of,
                        expiration=expiration,
                        frame=getattr(chain, "calls", None),
                        right=OptionRight.CALL,
                        spot=spot,
                    )
                )
                snapshots.extend(
                    _rows_to_snapshots(
                        ticker=ticker.upper(),
                        quote_timestamp=as_of,
                        expiration=expiration,
                        frame=getattr(chain, "puts", None),
                        right=OptionRight.PUT,
                        spot=spot,
                    )
                )
        return snapshots


def _rows_to_snapshots(
    ticker: str,
    quote_timestamp: datetime,
    expiration: datetime,
    frame: Any,
    right: OptionRight,
    spot: float,
) -> list[OptionSnapshot]:
    if frame is None or frame.empty:
        return []
    snapshots: list[OptionSnapshot] = []
    for row in frame.to_dict(orient="records"):
        strike = _clean_float(row.get("strike"))
        if strike is None or strike <= 0:
            continue
        bid = _clean_float(row.get("bid"))
        ask = _clean_float(row.get("ask"))
        last_price = _clean_float(row.get("lastPrice"))
        mid = _mid_price(bid, ask, last_price)
        contract_symbol = str(row.get("contractSymbol") or _option_symbol(ticker, expiration, right, strike))
        snapshots.append(
            OptionSnapshot(
                symbol=ticker,
                option_symbol=contract_symbol,
                quote_timestamp=quote_timestamp,
                expiration=expiration,
                strike=strike,
                right=right,
                bid=bid,
                ask=ask,
                mid=mid,
                implied_volatility=_clean_float(row.get("impliedVolatility")),
                volume=_clean_float(row.get("volume")),
                open_interest=_clean_float(row.get("openInterest")),
                spot=spot,
            )
        )
    return snapshots


def _fetch_spot(yf_ticker: Any) -> float | None:
    try:
        info = getattr(yf_ticker, "fast_info", None)
        if info is not None:
            for key in ("last_price", "lastPrice", "regular_market_price"):
                value = _clean_float(_safe_get(info, key))
                if value is not None:
                    return value
    except Exception:
        pass

    try:
        frame = yf_ticker.history(period="5d", interval="1d", auto_adjust=False)
        if frame is not None and not frame.empty:
            return _clean_float(frame["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return None


def _safe_get(container: Any, key: str) -> Any:
    try:
        return container[key]
    except Exception:
        return getattr(container, key, None)


def _mid_price(bid: float | None, ask: float | None, fallback: float | None) -> float | None:
    if bid is not None and ask is not None and bid >= 0 and ask >= 0:
        return (bid + ask) / 2.0
    return fallback


def _clean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _to_utc_datetime(value: Any) -> datetime:
    timestamp = value.to_pydatetime() if hasattr(value, "to_pydatetime") else value
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _expiration_to_utc(value: str) -> datetime:
    date = datetime.strptime(value, "%Y-%m-%d").date()
    # Treat expiration resolution as the regular market close timestamp.
    return datetime.combine(date, time(hour=21), tzinfo=UTC)


def _option_symbol(ticker: str, expiration: datetime, right: OptionRight, strike: float) -> str:
    return f"{ticker}-{expiration.date().isoformat()}-{right.value}-{strike:g}"


def _utc_now_if_none(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
