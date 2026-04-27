from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb

from murphy.predict import DeterministicPredictor, predict_pending
from murphy.repository import DuckDbRepository
from murphy.schemas import OptionRight, OptionSnapshot, UnderlyingBar


def test_duckdb_repository_live_question_and_prediction(tmp_path) -> None:
    db_path = tmp_path / "murphy.duckdb"
    quote_time = datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc)
    repo = DuckDbRepository(db_path)
    try:
        repo.insert_underlying_bars(
            [
                UnderlyingBar(
                    symbol="AAPL",
                    timestamp=quote_time - timedelta(days=1),
                    open=199,
                    high=202,
                    low=198,
                    close=201,
                    volume=1000,
                )
            ]
        )
        repo.insert_option_snapshots(
            [
                OptionSnapshot(
                    symbol="AAPL",
                    option_symbol="AAPL260501C00200000",
                    quote_timestamp=quote_time,
                    expiration=quote_time + timedelta(days=7),
                    strike=200,
                    right=OptionRight.CALL,
                    bid=4.9,
                    ask=5.1,
                    mid=5.0,
                    implied_volatility=0.25,
                    volume=10,
                    open_interest=100,
                    spot=201,
                )
            ]
        )
        questions = repo.generate_live_questions(min_dte=5, max_dte=10, max_questions=5)
        assert len(questions) == 1

        prompts = repo.pending_prediction_prompts()
        assert len(prompts) == 1
        assert "Will the price of $AAPL" in prompts[0]["prompt"]

        predictions = predict_pending(
            repo,
            DeterministicPredictor(probability=0.61),
            model="dry-run",
        )
        assert len(predictions) == 1
        report = repo.evaluation_report(ticker="AAPL")
        assert report["summary"]["n_predictions"] == 1
        assert report["summary"]["n_resolved"] == 0
        item = report["predictions"][0]
        assert item["external_cache"]["llm_prediction_records"] == 1
        assert item["leakage_checks"]["prediction_not_after_resolution_due"] is True

        repo.db.conn.execute(
            """
            INSERT INTO live_resolutions
              (question_id, resolved_at, underlying_close, label, source)
            VALUES (?, ?, ?, ?, 'test')
            """,
            [questions[0].question_id, quote_time + timedelta(days=7), 205.0, 1],
        )
        repo.db.conn.execute(
            "UPDATE option_examples SET label = 1 WHERE example_id = ?",
            [questions[0].example_id],
        )
        resolved_report = repo.evaluation_report(ticker="AAPL", include_unresolved=False)
        assert resolved_report["summary"]["n_resolved"] == 1
        assert resolved_report["summary"]["brier_score"] == (0.61 - 1.0) ** 2
        assert resolved_report["predictions"][0]["correct_at_0_5"] is True
    finally:
        repo.close()

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        assert con.execute("SELECT count(*) FROM live_questions").fetchone()[0] == 1
        assert con.execute("SELECT status FROM live_questions").fetchone()[0] == "predicted"
        assert con.execute("SELECT probability FROM llm_responses").fetchone()[0] == 0.61
        cache = con.execute(
            """
            SELECT provider, call_type, response_sha256
            FROM external_call_cache
            """
        ).fetchone()
        assert cache[0] == "deterministic"
        assert cache[1] == "llm_prediction"
        assert len(cache[2]) == 64
    finally:
        con.close()


def test_duckdb_repository_limits_one_question_per_strike_expiry_hour(tmp_path) -> None:
    db_path = tmp_path / "murphy.duckdb"
    quote_time = datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc)
    expiration = quote_time + timedelta(days=7)
    repo = DuckDbRepository(db_path)
    try:
        repo.insert_option_snapshots(
            [
                OptionSnapshot(
                    symbol="AAPL",
                    option_symbol="AAPL260504C00200000",
                    quote_timestamp=quote_time,
                    expiration=expiration,
                    strike=200,
                    right=OptionRight.CALL,
                    bid=4.9,
                    ask=5.1,
                    mid=5.0,
                    implied_volatility=0.25,
                    volume=10,
                    open_interest=100,
                    spot=201,
                )
            ]
        )
        assert len(repo.generate_live_questions(min_dte=5, max_dte=10, max_questions=5)) == 1

        repo.insert_option_snapshots(
            [
                OptionSnapshot(
                    symbol="AAPL",
                    option_symbol="AAPL260504C00200000",
                    quote_timestamp=quote_time + timedelta(minutes=30),
                    expiration=expiration,
                    strike=200,
                    right=OptionRight.CALL,
                    bid=5.2,
                    ask=5.4,
                    mid=5.3,
                    implied_volatility=0.26,
                    volume=12,
                    open_interest=100,
                    spot=202,
                )
            ]
        )
        assert len(repo.generate_live_questions(min_dte=5, max_dte=10, max_questions=5)) == 0

        repo.insert_option_snapshots(
            [
                OptionSnapshot(
                    symbol="AAPL",
                    option_symbol="AAPL260504C00200000",
                    quote_timestamp=quote_time + timedelta(hours=1),
                    expiration=expiration,
                    strike=200,
                    right=OptionRight.CALL,
                    bid=6.0,
                    ask=6.2,
                    mid=6.1,
                    implied_volatility=0.27,
                    volume=15,
                    open_interest=100,
                    spot=203,
                )
            ]
        )
        next_hour_questions = repo.generate_live_questions(min_dte=5, max_dte=10, max_questions=5)
        assert len(next_hour_questions) == 1
    finally:
        repo.close()


def test_duckdb_resolver_requires_expiration_date_bar(tmp_path) -> None:
    db_path = tmp_path / "murphy.duckdb"
    quote_time = datetime(2026, 4, 18, 16, 0, tzinfo=timezone.utc)
    expiration = datetime(2026, 4, 25, 21, 0, tzinfo=timezone.utc)
    repo = DuckDbRepository(db_path)
    try:
        repo.insert_option_snapshots(
            [
                OptionSnapshot(
                    symbol="AAPL",
                    option_symbol="AAPL260425C00200000",
                    quote_timestamp=quote_time,
                    expiration=expiration,
                    strike=200,
                    right=OptionRight.CALL,
                    bid=4.9,
                    ask=5.1,
                    mid=5.0,
                    implied_volatility=0.25,
                    volume=10,
                    open_interest=100,
                    spot=201,
                )
            ]
        )
        questions = repo.generate_live_questions(min_dte=5, max_dte=10, max_questions=5)
        assert len(questions) == 1

        repo.insert_underlying_bars(
            [
                UnderlyingBar(
                    symbol="AAPL",
                    timestamp=expiration - timedelta(days=1),
                    open=204,
                    high=206,
                    low=203,
                    close=205,
                    volume=1000,
                )
            ]
        )
        assert repo.resolve_due_live_questions() == 0

        repo.insert_underlying_bars(
            [
                UnderlyingBar(
                    symbol="AAPL",
                    timestamp=expiration - timedelta(minutes=1),
                    open=204,
                    high=206,
                    low=203,
                    close=205,
                    volume=1000,
                )
            ]
        )
        assert repo.resolve_due_live_questions() == 1

        report = repo.evaluation_report(ticker="AAPL", include_unresolved=False)
        assert report["summary"]["n_resolved"] == 1
        assert report["predictions"][0]["label"] == 1
        assert report["predictions"][0]["underlying_close"] == 205
    finally:
        repo.close()
