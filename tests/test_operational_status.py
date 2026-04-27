from __future__ import annotations

from datetime import UTC, datetime, timedelta

from murphy.operational_status import build_operational_status_report, operational_warnings
from murphy.repository import DuckDbRepository
from murphy.schemas import OptionRight, OptionSnapshot, UnderlyingBar


def test_duckdb_repository_operational_status(tmp_path) -> None:
    db_path = tmp_path / "murphy.duckdb"
    now = datetime.now(UTC)
    repo = DuckDbRepository(db_path)
    try:
        repo.insert_underlying_bars(
            [
                UnderlyingBar(
                    symbol="AAPL",
                    timestamp=now,
                    open=100,
                    high=101,
                    low=99,
                    close=100,
                    volume=1000,
                )
            ]
        )
        repo.insert_option_snapshots(
            [
                OptionSnapshot(
                    symbol="AAPL",
                    option_symbol="AAPL260504C00100000",
                    quote_timestamp=now,
                    expiration=now + timedelta(days=7),
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
            ]
        )
        assert len(repo.generate_live_questions(min_dte=5, max_dte=10, max_questions=5)) == 1

        status = repo.operational_status()

        assert status["option_snapshots"]["count"] == 1
        assert status["underlying_bars"]["count"] == 1
        assert status["live_questions"]["total"] == 1
        assert status["live_questions"]["status_counts"]["pending_prediction"] == 1
    finally:
        repo.close()


def test_operational_warnings_for_stale_snapshots_and_pending_predictions() -> None:
    now = datetime(2026, 4, 27, 18, 0, tzinfo=UTC)
    status = {
        "option_snapshots": {"count": 1, "latest": now - timedelta(hours=7)},
        "live_questions": {
            "total": 2,
            "status_counts": {"pending_prediction": 1},
            "due_unresolved": 1,
        },
    }

    warnings = operational_warnings(status, now=now, max_snapshot_age_hours=6)
    codes = {warning["code"] for warning in warnings}

    assert "stale_option_snapshots" in codes
    assert "pending_predictions" in codes
    assert "due_unresolved_questions" in codes


def test_operational_status_report_includes_bigquery_warnings() -> None:
    now = datetime.now(UTC)
    report = build_operational_status_report(
        {
            "option_snapshots": {"count": 1, "latest": now},
            "live_questions": {"total": 1, "status_counts": {}, "due_unresolved": 0},
        },
        bigquery_status={"live_questions": {"num_rows": 1, "modified": now - timedelta(days=2)}},
        max_bigquery_age_hours=24,
    )

    assert any(warning["code"] == "stale_bigquery_mirror" for warning in report["warnings"])
