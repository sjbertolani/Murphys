from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb

from murphy.market_data.collect import collect_market_snapshots
from murphy.repository import DuckDbRepository
from murphy.schemas import OptionRight, OptionSnapshot, UnderlyingBar


class FakeProvider:
    name = "fake"

    def fetch_underlying_bars(self, tickers, lookback_days, as_of=None):
        del lookback_days
        as_of = as_of or datetime.now(timezone.utc)
        return [
            UnderlyingBar(
                symbol=ticker,
                timestamp=as_of - timedelta(days=1),
                open=99,
                high=101,
                low=98,
                close=100,
                volume=1000,
            )
            for ticker in tickers
        ]

    def fetch_option_chain_snapshots(self, tickers, min_dte, max_dte, as_of=None):
        del min_dte, max_dte
        as_of = as_of or datetime.now(timezone.utc)
        return [
            OptionSnapshot(
                symbol=ticker,
                option_symbol=f"{ticker}260501C00100000",
                quote_timestamp=as_of,
                expiration=as_of + timedelta(days=7),
                strike=100,
                right=OptionRight.CALL,
                bid=4.9,
                ask=5.1,
                mid=5.0,
                implied_volatility=0.3,
                volume=10,
                open_interest=100,
                spot=100,
            )
            for ticker in tickers
        ]


def test_collect_snapshots_to_duckdb(tmp_path) -> None:
    db_path = tmp_path / "murphy.duckdb"
    as_of = datetime(2026, 4, 26, 18, 0, tzinfo=timezone.utc)
    bars, snapshots, result = collect_market_snapshots(
        provider=FakeProvider(),
        tickers=["aapl"],
        min_dte=5,
        max_dte=10,
        lookback_days=10,
        as_of=as_of,
    )

    repo = DuckDbRepository(db_path)
    try:
        assert repo.insert_underlying_bars(bars) == 1
        assert repo.insert_option_snapshots(snapshots) == 1
        repo.record_collection_result(result)
    finally:
        repo.close()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        assert con.execute("SELECT count(*) FROM underlying_bars").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM option_chain_snapshots").fetchone()[0] == 1
        assert con.execute("SELECT count(*) FROM daily_runs").fetchone()[0] == 1
    finally:
        con.close()

