from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from murphy.cloud_config import CloudSqlConfig
from murphy.cloud_sql import close_cloud_sql_engine, create_cloud_sql_engine
from murphy.db import MurphyDb
from murphy.live import LiveQuestion, call_option_question_text, live_llm_prompt
from murphy.market_data.collect import CollectionResult
from murphy.options_data import make_example_id
from murphy.priors import (
    bayesian_binary_update,
    delta_as_probability,
    risk_neutral_call_itm_probability,
)
from murphy.schemas import OptionSnapshot, UnderlyingBar


class DuckDbRepository:
    def __init__(self, db_path: str | Path = "data/murphy.duckdb") -> None:
        self.db = MurphyDb(db_path)
        self.db.initialize()

    def close(self) -> None:
        self.db.close()

    def insert_underlying_bars(self, bars: list[UnderlyingBar]) -> int:
        rows = [_bar_row(bar) for bar in bars]
        if not rows:
            return 0
        self.db.conn.executemany(
            """
            INSERT OR REPLACE INTO underlying_bars
              (symbol, timestamp, open, high, low, close, volume, adjusted_close)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return len(rows)

    def insert_option_snapshots(self, snapshots: list[OptionSnapshot]) -> int:
        rows = [_snapshot_row(snapshot) for snapshot in snapshots]
        if not rows:
            return 0
        self.db.conn.executemany(
            """
            INSERT OR REPLACE INTO option_chain_snapshots
              (symbol, option_symbol, quote_timestamp, expiration, strike, option_right,
               bid, ask, mid, implied_volatility, delta, gamma, theta, vega,
               volume, open_interest, spot)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return len(rows)

    def record_collection_result(self, result: CollectionResult) -> None:
        self.db.conn.execute(
            """
            INSERT OR REPLACE INTO daily_runs
              (run_id, run_type, started_at, completed_at, status, details_json)
            VALUES (?, 'collect_snapshots', ?, ?, 'completed', ?)
            """,
            [
                f"collect:{result.provider}:{result.collected_at.isoformat()}",
                result.collected_at,
                datetime.now(result.collected_at.tzinfo),
                json.dumps(result.__dict__, default=str, sort_keys=True),
            ],
        )

    def record_external_call(
        self,
        provider: str,
        call_type: str,
        request_payload: dict,
        response_payload: dict | list | None = None,
        response_text: str | None = None,
        captured_at: datetime | None = None,
        information_cutoff: datetime | None = None,
        source_timestamp: datetime | None = None,
    ) -> str:
        call = _external_call_record(
            provider=provider,
            call_type=call_type,
            request_payload=request_payload,
            response_payload=response_payload,
            response_text=response_text,
            captured_at=captured_at,
            information_cutoff=information_cutoff,
            source_timestamp=source_timestamp,
        )
        self.db.conn.execute(
            """
            INSERT INTO external_call_cache
              (call_id, provider, call_type, request_json, response_json, response_text,
               captured_at, information_cutoff, source_timestamp, response_sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                call["call_id"],
                call["provider"],
                call["call_type"],
                call["request_json"],
                call["response_json"],
                call["response_text"],
                call["captured_at"],
                call["information_cutoff"],
                call["source_timestamp"],
                call["response_sha256"],
            ],
        )
        return str(call["call_id"])

    def generate_live_questions(
        self,
        max_abs_moneyness: float = 0.03,
        min_dte: float = 0.0,
        max_dte: float = 14.0,
        max_questions: int = 50,
        min_open_interest: float = 1.0,
        min_volume: float = 0.0,
        strike_window_size: int = 5,
        max_questions_per_ticker: int | None = None,
        source: str = "live_option_chain",
    ) -> list[LiveQuestion]:
        return _generate_live_questions_duckdb(
            self.db.conn,
            max_abs_moneyness,
            min_dte,
            max_dte,
            max_questions,
            min_open_interest,
            min_volume,
            strike_window_size,
            max_questions_per_ticker,
            source,
        )

    def pending_prediction_prompts(self, limit: int | None = None) -> list[dict]:
        return _pending_prediction_prompts_duckdb(self.db.conn, limit)

    def record_llm_response(
        self,
        question_id: str,
        model: str,
        prompt: str,
        probability: float,
        reasoning: str | None,
        raw_response: str | None,
    ) -> str:
        response_id = str(uuid4())
        created_at = datetime.now().astimezone()
        info_cutoff = self.db.conn.execute(
            "SELECT information_cutoff, example_id FROM live_questions WHERE question_id = ?",
            [question_id],
        ).fetchone()
        if info_cutoff is None:
            raise ValueError(f"unknown question_id {question_id}")
        prior_components = _prior_components_for_example_duckdb(self.db.conn, info_cutoff[1])
        posterior_probability = bayesian_binary_update(
            prior_components["combined_prior_probability"],
            probability,
            signal_weight=0.75,
        )
        self.db.conn.execute(
            """
            INSERT INTO llm_responses
              (response_id, question_id, model, prompt, probability, reasoning,
               raw_response, created_at, information_cutoff)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                response_id,
                question_id,
                model,
                prompt,
                probability,
                reasoning,
                raw_response,
                created_at,
                info_cutoff[0],
            ],
        )
        self.record_external_call(
            provider="openai" if model != "dry-run" else "deterministic",
            call_type="llm_prediction",
            request_payload={
                "model": model,
                "question_id": question_id,
                "prompt": prompt,
            },
            response_payload={
                "probability": probability,
                "reasoning": reasoning,
            },
            response_text=raw_response,
            captured_at=created_at,
            information_cutoff=info_cutoff[0],
            source_timestamp=created_at,
        )
        _record_prediction_trace_duckdb(
            self.db.conn,
            response_id=response_id,
            example_id=info_cutoff[1],
            model=model,
            created_at=created_at,
            probability=probability,
            reasoning=reasoning,
            raw_response=raw_response,
            prior_components=prior_components,
            posterior_probability=posterior_probability,
        )
        self.db.conn.execute(
            "UPDATE live_questions SET status = 'predicted' WHERE question_id = ?",
            [question_id],
        )
        return response_id

    def resolve_due_live_questions(self, require_expiration_date: bool = True) -> int:
        return _resolve_due_live_questions_duckdb(self.db.conn, require_expiration_date)

    def live_ticker_summary(self, ticker: str, limit: int = 5) -> dict:
        return _live_ticker_summary_duckdb(self.db.conn, ticker, limit)

    def evaluation_report(
        self,
        ticker: str | None = None,
        limit: int = 50,
        include_unresolved: bool = True,
    ) -> dict:
        return _evaluation_report_duckdb(self.db.conn, ticker, limit, include_unresolved)

    def operational_status(self) -> dict:
        return _operational_status_duckdb(self.db.conn)


class CloudSqlRepository:
    def __init__(self, config: CloudSqlConfig) -> None:
        self.engine = create_cloud_sql_engine(config)

    def close(self) -> None:
        close_cloud_sql_engine(self.engine)

    def insert_underlying_bars(self, bars: list[UnderlyingBar]) -> int:
        import sqlalchemy

        rows = [_bar_dict(bar) for bar in bars]
        if not rows:
            return 0
        with self.engine.begin() as conn:
            conn.execute(
                sqlalchemy.text(
                    """
                INSERT INTO underlying_bars
                  (symbol, timestamp, open, high, low, close, volume, adjusted_close)
                VALUES
                  (:symbol, :timestamp, :open, :high, :low, :close, :volume, :adjusted_close)
                ON CONFLICT (symbol, timestamp) DO UPDATE SET
                  open = EXCLUDED.open,
                  high = EXCLUDED.high,
                  low = EXCLUDED.low,
                  close = EXCLUDED.close,
                  volume = EXCLUDED.volume,
                  adjusted_close = EXCLUDED.adjusted_close
                """,
                ),
                rows,
            )
        return len(rows)

    def insert_option_snapshots(self, snapshots: list[OptionSnapshot]) -> int:
        import sqlalchemy

        rows = [_snapshot_dict(snapshot) for snapshot in snapshots]
        if not rows:
            return 0
        with self.engine.begin() as conn:
            conn.execute(
                sqlalchemy.text(
                    """
                INSERT INTO option_chain_snapshots
                  (symbol, option_symbol, quote_timestamp, expiration, strike, option_right,
                   bid, ask, mid, implied_volatility, delta, gamma, theta, vega,
                   volume, open_interest, spot)
                VALUES
                  (:symbol, :option_symbol, :quote_timestamp, :expiration, :strike, :option_right,
                   :bid, :ask, :mid, :implied_volatility, :delta, :gamma, :theta, :vega,
                   :volume, :open_interest, :spot)
                ON CONFLICT (option_symbol, quote_timestamp) DO UPDATE SET
                  bid = EXCLUDED.bid,
                  ask = EXCLUDED.ask,
                  mid = EXCLUDED.mid,
                  implied_volatility = EXCLUDED.implied_volatility,
                  delta = EXCLUDED.delta,
                  gamma = EXCLUDED.gamma,
                  theta = EXCLUDED.theta,
                  vega = EXCLUDED.vega,
                  volume = EXCLUDED.volume,
                  open_interest = EXCLUDED.open_interest,
                  spot = EXCLUDED.spot
                """,
                ),
                rows,
            )
        return len(rows)

    def record_collection_result(self, result: CollectionResult) -> None:
        import sqlalchemy

        payload = json.dumps(result.__dict__, default=str, sort_keys=True)
        with self.engine.begin() as conn:
            conn.execute(
                sqlalchemy.text(
                    """
                INSERT INTO daily_runs
                  (run_id, run_type, started_at, completed_at, status, details_json)
                VALUES
                  (:run_id, 'collect_snapshots', :started_at, :completed_at, 'completed',
                   CAST(:details_json AS JSONB))
                ON CONFLICT (run_id) DO UPDATE SET
                  completed_at = EXCLUDED.completed_at,
                  status = EXCLUDED.status,
                  details_json = EXCLUDED.details_json
                """,
                ),
                {
                    "run_id": f"collect:{result.provider}:{result.collected_at.isoformat()}",
                    "started_at": result.collected_at,
                    "completed_at": datetime.now(result.collected_at.tzinfo),
                    "details_json": payload,
                },
            )

    def record_external_call(
        self,
        provider: str,
        call_type: str,
        request_payload: dict,
        response_payload: dict | list | None = None,
        response_text: str | None = None,
        captured_at: datetime | None = None,
        information_cutoff: datetime | None = None,
        source_timestamp: datetime | None = None,
    ) -> str:
        import sqlalchemy

        call = _external_call_record(
            provider=provider,
            call_type=call_type,
            request_payload=request_payload,
            response_payload=response_payload,
            response_text=response_text,
            captured_at=captured_at,
            information_cutoff=information_cutoff,
            source_timestamp=source_timestamp,
        )
        with self.engine.begin() as conn:
            conn.execute(
                sqlalchemy.text(
                    """
                    INSERT INTO external_call_cache
                      (call_id, provider, call_type, request_json, response_json, response_text,
                       captured_at, information_cutoff, source_timestamp, response_sha256)
                    VALUES
                      (:call_id, :provider, :call_type, CAST(:request_json AS JSONB),
                       CAST(:response_json AS JSONB), :response_text, :captured_at,
                       :information_cutoff, :source_timestamp, :response_sha256)
                    """
                ),
                call,
            )
        return str(call["call_id"])

    def generate_live_questions(
        self,
        max_abs_moneyness: float = 0.03,
        min_dte: float = 0.0,
        max_dte: float = 14.0,
        max_questions: int = 50,
        min_open_interest: float = 1.0,
        min_volume: float = 0.0,
        strike_window_size: int = 5,
        max_questions_per_ticker: int | None = None,
        source: str = "live_option_chain",
    ) -> list[LiveQuestion]:
        import sqlalchemy

        effective_max_questions_per_ticker = (
            max_questions
            if max_questions_per_ticker is None
            else max_questions_per_ticker
        )
        now = datetime.now().astimezone()
        query = sqlalchemy.text(
            """
            WITH latest AS (
              SELECT symbol, max(quote_timestamp) AS quote_timestamp
              FROM option_chain_snapshots
              GROUP BY symbol
            ),
            candidates AS (
              SELECT
                s.symbol,
                s.option_symbol,
                s.quote_timestamp,
                s.expiration,
                s.strike,
                s.spot,
                extract(epoch from (s.expiration - s.quote_timestamp)) / 86400.0 AS dte,
                s.spot / s.strike - 1.0 AS moneyness,
                s.open_interest,
                s.volume,
                CASE WHEN s.strike < s.spot THEN 'below' ELSE 'above' END AS strike_side,
                row_number() OVER (
                  PARTITION BY
                    s.symbol,
                    s.expiration,
                    CASE WHEN s.strike < s.spot THEN 'below' ELSE 'above' END
                  ORDER BY
                    CASE WHEN s.strike < s.spot THEN s.strike END DESC,
                    CASE WHEN s.strike >= s.spot THEN s.strike END ASC
                ) AS strike_side_rank
              FROM option_chain_snapshots s
              JOIN latest
                ON latest.symbol = s.symbol
               AND latest.quote_timestamp = s.quote_timestamp
              WHERE s.option_right = 'C'
                AND s.spot IS NOT NULL
                AND s.strike > 0
                AND extract(epoch from (s.expiration - s.quote_timestamp)) / 86400.0
                    BETWEEN :min_dte AND :max_dte
                AND coalesce(s.open_interest, 0) >= :min_open_interest
                AND coalesce(s.volume, 0) >= :min_volume
                AND NOT EXISTS (
                  SELECT 1
                  FROM option_examples e
                  JOIN live_questions q ON q.example_id = e.example_id
                  WHERE e.symbol = s.symbol
                    AND e.expiration = s.expiration
                    AND e.strike = s.strike
                    AND date_trunc('hour', e.forecast_timestamp)
                        = date_trunc('hour', s.quote_timestamp)
                )
            ),
            ladder AS (
              SELECT *
              FROM candidates
              WHERE strike_side_rank <= :strike_window_size
            ),
            ranked AS (
              SELECT
                *,
                row_number() OVER (
                  PARTITION BY symbol
                  ORDER BY expiration, strike_side_rank, strike_side, strike
                ) AS symbol_question_rank
              FROM ladder
            )
            SELECT
              symbol, option_symbol, quote_timestamp, expiration, strike, spot, dte, moneyness
            FROM ranked
            WHERE strike_side_rank <= :strike_window_size
              AND symbol_question_rank <= :max_questions_per_ticker
            ORDER BY symbol_question_rank, symbol, expiration, strike_side_rank, strike_side, strike
            LIMIT :max_questions
            """
        )
        with self.engine.begin() as conn:
            rows = conn.execute(
                query,
                {
                    "max_abs_moneyness": max_abs_moneyness,
                    "min_dte": min_dte,
                    "max_dte": max_dte,
                    "min_open_interest": min_open_interest,
                    "min_volume": min_volume,
                    "strike_window_size": strike_window_size,
                    "max_questions_per_ticker": effective_max_questions_per_ticker,
                    "max_questions": max_questions,
                },
            ).fetchall()
            questions = [_live_question_from_candidate(row) for row in rows]
            for question, row in zip(questions, rows, strict=True):
                conn.execute(
                    sqlalchemy.text(
                        """
                        INSERT INTO option_examples
                          (example_id, symbol, option_symbol, forecast_timestamp, expiration,
                           strike, spot, dte, moneyness, label, resolution_timestamp)
                        VALUES
                          (:example_id, :symbol, :option_symbol, :forecast_timestamp, :expiration,
                           :strike, :spot, :dte, :moneyness, NULL, :resolution_timestamp)
                        ON CONFLICT (example_id) DO UPDATE SET
                          spot = EXCLUDED.spot,
                          dte = EXCLUDED.dte,
                          moneyness = EXCLUDED.moneyness
                        """
                    ),
                    {
                        "example_id": question.example_id,
                        "symbol": row.symbol,
                        "option_symbol": row.option_symbol,
                        "forecast_timestamp": row.quote_timestamp,
                        "expiration": row.expiration,
                        "strike": row.strike,
                        "spot": row.spot,
                        "dte": row.dte,
                        "moneyness": row.moneyness,
                        "resolution_timestamp": row.expiration,
                    },
                )
                conn.execute(
                    sqlalchemy.text(
                        """
                        INSERT INTO live_questions
                          (question_id, example_id, question_text, generated_at,
                           information_cutoff, resolution_due, status, source)
                        VALUES
                          (:question_id, :example_id, :question_text, :generated_at,
                           :information_cutoff, :resolution_due, 'pending_prediction', :source)
                        ON CONFLICT (question_id) DO NOTHING
                        """
                    ),
                    {
                        "question_id": question.question_id,
                        "example_id": question.example_id,
                        "question_text": question.question_text,
                        "generated_at": now,
                        "information_cutoff": question.information_cutoff,
                        "resolution_due": question.expiration,
                        "source": source,
                    },
                )
            return questions

    def pending_prediction_prompts(self, limit: int | None = None) -> list[dict]:
        import sqlalchemy

        limit_clause = "" if limit is None else "LIMIT :limit"
        query = sqlalchemy.text(
            f"""
            SELECT question_id, example_id, question_text, generated_at,
                   information_cutoff, resolution_due
            FROM live_questions
            WHERE status = 'pending_prediction'
            ORDER BY generated_at, question_id
            {limit_clause}
            """
        )
        with self.engine.begin() as conn:
            rows = conn.execute(query, {} if limit is None else {"limit": limit}).fetchall()
            return [_prompt_record_postgres(conn, row) for row in rows]

    def record_llm_response(
        self,
        question_id: str,
        model: str,
        prompt: str,
        probability: float,
        reasoning: str | None,
        raw_response: str | None,
    ) -> str:
        import sqlalchemy

        response_id = str(uuid4())
        created_at = datetime.now().astimezone()
        with self.engine.begin() as conn:
            info_cutoff = conn.execute(
                sqlalchemy.text(
                    """
                    SELECT information_cutoff, example_id
                    FROM live_questions
                    WHERE question_id = :question_id
                    """
                ),
                {"question_id": question_id},
            ).fetchone()
            if info_cutoff is None:
                raise ValueError(f"unknown question_id {question_id}")
            prior_components = _prior_components_for_example_postgres(
                conn,
                sqlalchemy,
                info_cutoff.example_id,
            )
            posterior_probability = bayesian_binary_update(
                prior_components["combined_prior_probability"],
                probability,
                signal_weight=0.75,
            )
            conn.execute(
                sqlalchemy.text(
                    """
                    INSERT INTO llm_responses
                      (response_id, question_id, model, prompt, probability, reasoning,
                       raw_response, created_at, information_cutoff)
                    VALUES
                      (:response_id, :question_id, :model, :prompt, :probability, :reasoning,
                       :raw_response, :created_at, :information_cutoff)
                    """
                ),
                {
                    "response_id": response_id,
                    "question_id": question_id,
                    "model": model,
                    "prompt": prompt,
                    "probability": probability,
                    "reasoning": reasoning,
                    "raw_response": raw_response,
                    "created_at": created_at,
                    "information_cutoff": info_cutoff.information_cutoff,
                },
            )
            _record_external_call_postgres(
                conn,
                sqlalchemy,
                provider="openai" if model != "dry-run" else "deterministic",
                call_type="llm_prediction",
                request_payload={
                    "model": model,
                    "question_id": question_id,
                    "prompt": prompt,
                },
                response_payload={
                    "probability": probability,
                    "reasoning": reasoning,
                },
                response_text=raw_response,
                captured_at=created_at,
                information_cutoff=info_cutoff.information_cutoff,
                source_timestamp=created_at,
            )
            _record_prediction_trace_postgres(
                conn,
                response_id=response_id,
                example_id=info_cutoff.example_id,
                model=model,
                created_at=created_at,
                probability=probability,
                reasoning=reasoning,
                raw_response=raw_response,
                prior_components=prior_components,
                posterior_probability=posterior_probability,
            )
            conn.execute(
                sqlalchemy.text(
                    "UPDATE live_questions SET status = 'predicted' WHERE question_id = :question_id"
                ),
                {"question_id": question_id},
            )
        return response_id

    def resolve_due_live_questions(self, require_expiration_date: bool = True) -> int:
        import sqlalchemy

        with self.engine.begin() as conn:
            rows = conn.execute(
                sqlalchemy.text(
                    """
                    SELECT q.question_id, e.symbol, e.strike, e.expiration
                    FROM live_questions q
                    JOIN option_examples e ON e.example_id = q.example_id
                    LEFT JOIN live_resolutions r ON r.question_id = q.question_id
                    WHERE r.question_id IS NULL
                      AND q.resolution_due <= now()
                    """
                )
            ).fetchall()
            resolved = 0
            for row in rows:
                date_filter = (
                    "AND CAST(timestamp AS DATE) = CAST(:expiration AS DATE)"
                    if require_expiration_date
                    else ""
                )
                bar = conn.execute(
                    sqlalchemy.text(
                        f"""
                        SELECT close, timestamp
                        FROM underlying_bars
                        WHERE symbol = :symbol
                          AND timestamp <= :expiration
                          {date_filter}
                        ORDER BY timestamp DESC
                        LIMIT 1
                        """
                    ),
                    {"symbol": row.symbol, "expiration": row.expiration},
                ).fetchone()
                if bar is None:
                    continue
                label = int(float(bar.close) > float(row.strike))
                conn.execute(
                    sqlalchemy.text(
                        """
                        INSERT INTO live_resolutions
                          (question_id, resolved_at, underlying_close, label, source)
                        VALUES
                          (:question_id, :resolved_at, :underlying_close, :label, 'underlying_bars')
                        ON CONFLICT (question_id) DO UPDATE SET
                          resolved_at = EXCLUDED.resolved_at,
                          underlying_close = EXCLUDED.underlying_close,
                          label = EXCLUDED.label,
                          source = EXCLUDED.source
                        """
                    ),
                    {
                        "question_id": row.question_id,
                        "resolved_at": bar.timestamp,
                        "underlying_close": bar.close,
                        "label": label,
                    },
                )
                conn.execute(
                    sqlalchemy.text(
                        "UPDATE live_questions SET status = 'resolved' WHERE question_id = :question_id"
                    ),
                    {"question_id": row.question_id},
                )
                conn.execute(
                    sqlalchemy.text(
                        """
                        UPDATE option_examples
                        SET label = :label, resolution_timestamp = :resolved_at
                        WHERE example_id = (
                          SELECT example_id FROM live_questions WHERE question_id = :question_id
                        )
                        """
                    ),
                    {
                        "label": label,
                        "resolved_at": bar.timestamp,
                        "question_id": row.question_id,
                    },
                )
                resolved += 1
            return resolved

    def live_ticker_summary(self, ticker: str, limit: int = 5) -> dict:
        import sqlalchemy

        with self.engine.begin() as conn:
            return _live_ticker_summary_postgres(conn, sqlalchemy, ticker, limit)

    def evaluation_report(
        self,
        ticker: str | None = None,
        limit: int = 50,
        include_unresolved: bool = True,
    ) -> dict:
        import sqlalchemy

        with self.engine.begin() as conn:
            return _evaluation_report_postgres(
                conn,
                sqlalchemy,
                ticker,
                limit,
                include_unresolved,
            )

    def operational_status(self) -> dict:
        import sqlalchemy

        with self.engine.begin() as conn:
            return _operational_status_postgres(conn, sqlalchemy)


def _bar_row(bar: UnderlyingBar) -> tuple:
    return (
        bar.symbol,
        bar.timestamp,
        bar.open,
        bar.high,
        bar.low,
        bar.close,
        bar.volume,
        bar.adjusted_close,
    )


def _bar_dict(bar: UnderlyingBar) -> dict:
    return {
        "symbol": bar.symbol,
        "timestamp": bar.timestamp,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
        "adjusted_close": bar.adjusted_close,
    }


def _snapshot_row(snapshot: OptionSnapshot) -> tuple:
    return (
        snapshot.symbol,
        snapshot.option_symbol,
        snapshot.quote_timestamp,
        snapshot.expiration,
        snapshot.strike,
        snapshot.right.value,
        snapshot.bid,
        snapshot.ask,
        snapshot.mid,
        snapshot.implied_volatility,
        snapshot.delta,
        snapshot.gamma,
        snapshot.theta,
        snapshot.vega,
        snapshot.volume,
        snapshot.open_interest,
        snapshot.spot,
    )


def _snapshot_dict(snapshot: OptionSnapshot) -> dict:
    return {
        "symbol": snapshot.symbol,
        "option_symbol": snapshot.option_symbol,
        "quote_timestamp": snapshot.quote_timestamp,
        "expiration": snapshot.expiration,
        "strike": snapshot.strike,
        "option_right": snapshot.right.value,
        "bid": snapshot.bid,
        "ask": snapshot.ask,
        "mid": snapshot.mid,
        "implied_volatility": snapshot.implied_volatility,
        "delta": snapshot.delta,
        "gamma": snapshot.gamma,
        "theta": snapshot.theta,
        "vega": snapshot.vega,
        "volume": snapshot.volume,
        "open_interest": snapshot.open_interest,
        "spot": snapshot.spot,
    }


def _external_call_record(
    provider: str,
    call_type: str,
    request_payload: dict,
    response_payload: dict | list | None,
    response_text: str | None,
    captured_at: datetime | None,
    information_cutoff: datetime | None,
    source_timestamp: datetime | None,
) -> dict:
    captured_at = captured_at or datetime.now().astimezone()
    request_json = json.dumps(request_payload, default=str, sort_keys=True)
    response_json = (
        None
        if response_payload is None
        else json.dumps(response_payload, default=str, sort_keys=True)
    )
    response_fingerprint = json.dumps(
        {
            "response_json": response_payload,
            "response_text": response_text,
        },
        default=str,
        sort_keys=True,
    )
    return {
        "call_id": str(uuid4()),
        "provider": provider,
        "call_type": call_type,
        "request_json": request_json,
        "response_json": response_json,
        "response_text": response_text,
        "captured_at": captured_at,
        "information_cutoff": information_cutoff,
        "source_timestamp": source_timestamp,
        "response_sha256": hashlib.sha256(response_fingerprint.encode("utf-8")).hexdigest(),
    }


def _record_external_call_postgres(
    conn,
    sqlalchemy,
    provider: str,
    call_type: str,
    request_payload: dict,
    response_payload: dict | list | None,
    response_text: str | None,
    captured_at: datetime | None,
    information_cutoff: datetime | None,
    source_timestamp: datetime | None,
) -> str:
    call = _external_call_record(
        provider=provider,
        call_type=call_type,
        request_payload=request_payload,
        response_payload=response_payload,
        response_text=response_text,
        captured_at=captured_at,
        information_cutoff=information_cutoff,
        source_timestamp=source_timestamp,
    )
    conn.execute(
        sqlalchemy.text(
            """
            INSERT INTO external_call_cache
              (call_id, provider, call_type, request_json, response_json, response_text,
               captured_at, information_cutoff, source_timestamp, response_sha256)
            VALUES
              (:call_id, :provider, :call_type, CAST(:request_json AS JSONB),
               CAST(:response_json AS JSONB), :response_text, :captured_at,
               :information_cutoff, :source_timestamp, :response_sha256)
            """
        ),
        call,
    )
    return str(call["call_id"])


def _prediction_beliefs(
    probability: float,
    reasoning: str | None,
    raw_response: str | None,
    prior_components: dict,
    posterior_probability: float,
) -> dict:
    prior_probability = float(prior_components["combined_prior_probability"])
    return {
        "initial": {
            "probability": prior_probability,
            "confidence": 0.0,
            "evidence_for": [],
            "evidence_against": [],
            "open_questions": ["Awaiting LLM probability estimate from cutoff-bounded prompt."],
            "update_reasoning": "Prior belief before the LLM response is observed.",
            "prior_components": prior_components,
        },
        "llm": {
            "probability": float(probability),
            "confidence": abs(float(probability) - 0.5) * 2.0,
            "evidence_for": [],
            "evidence_against": [],
            "open_questions": [],
            "update_reasoning": reasoning or "Model returned a probability without separate reasoning.",
            "raw_response": raw_response,
        },
        "posterior": {
            "probability": float(posterior_probability),
            "confidence": abs(float(posterior_probability) - 0.5) * 2.0,
            "evidence_for": [],
            "evidence_against": [],
            "open_questions": [],
            "update_reasoning": (
                "Log-odds Bayesian update combining the prior components with the "
                "LLM probability signal."
            ),
            "prior_components": prior_components,
            "llm_probability": float(probability),
            "llm_signal_weight": 0.75,
        },
    }


def _record_prediction_trace_duckdb(
    conn,
    response_id: str,
    example_id: str,
    model: str,
    created_at: datetime,
    probability: float,
    reasoning: str | None,
    raw_response: str | None,
    prior_components: dict,
    posterior_probability: float,
) -> None:
    trial_id = f"llm:{response_id}"
    beliefs = _prediction_beliefs(
        probability,
        reasoning,
        raw_response,
        prior_components,
        posterior_probability,
    )
    prior_probability = float(prior_components["combined_prior_probability"])
    conn.execute(
        """
        INSERT INTO agent_trials
          (trial_id, example_id, seed, model, started_at, raw_probability, status)
        VALUES (?, ?, NULL, ?, ?, ?, 'completed')
        """,
        [trial_id, example_id, model, created_at, posterior_probability],
    )
    conn.executemany(
        """
        INSERT INTO agent_steps
          (trial_id, step_index, action_type, action_json, observation_ref, belief_json, probability)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            [
                trial_id,
                0,
                "initialize_belief",
                json.dumps({"type": "initialize"}),
                None,
                json.dumps(beliefs["initial"], sort_keys=True),
                prior_probability,
            ],
            [
                trial_id,
                1,
                "record_llm_probability",
                json.dumps({"type": "llm_response", "response_id": response_id}),
                response_id,
                json.dumps(beliefs["llm"], sort_keys=True),
                probability,
            ],
            [
                trial_id,
                2,
                "bayesian_update",
                json.dumps({"type": "bayesian_update", "response_id": response_id}),
                response_id,
                json.dumps(beliefs["posterior"], sort_keys=True),
                posterior_probability,
            ],
        ],
    )


def _record_prediction_trace_postgres(
    conn,
    response_id: str,
    example_id: str,
    model: str,
    created_at: datetime,
    probability: float,
    reasoning: str | None,
    raw_response: str | None,
    prior_components: dict,
    posterior_probability: float,
) -> None:
    import sqlalchemy

    trial_id = f"llm:{response_id}"
    beliefs = _prediction_beliefs(
        probability,
        reasoning,
        raw_response,
        prior_components,
        posterior_probability,
    )
    prior_probability = float(prior_components["combined_prior_probability"])
    conn.execute(
        sqlalchemy.text(
            """
            INSERT INTO agent_trials
              (trial_id, example_id, seed, model, started_at, raw_probability, status)
            VALUES
              (:trial_id, :example_id, NULL, :model, :started_at, :raw_probability, 'completed')
            """
        ),
        {
            "trial_id": trial_id,
            "example_id": example_id,
            "model": model,
            "started_at": created_at,
            "raw_probability": posterior_probability,
        },
    )
    conn.execute(
        sqlalchemy.text(
            """
            INSERT INTO agent_steps
              (trial_id, step_index, action_type, action_json, observation_ref, belief_json, probability)
            VALUES
              (:trial_id, :step_index, :action_type, CAST(:action_json AS JSONB),
               :observation_ref, CAST(:belief_json AS JSONB), :probability)
            """
        ),
        [
            {
                "trial_id": trial_id,
                "step_index": 0,
                "action_type": "initialize_belief",
                "action_json": json.dumps({"type": "initialize"}),
                "observation_ref": None,
                "belief_json": json.dumps(beliefs["initial"], sort_keys=True),
                "probability": prior_probability,
            },
            {
                "trial_id": trial_id,
                "step_index": 1,
                "action_type": "record_llm_probability",
                "action_json": json.dumps({"type": "llm_response", "response_id": response_id}),
                "observation_ref": response_id,
                "belief_json": json.dumps(beliefs["llm"], sort_keys=True),
                "probability": probability,
            },
            {
                "trial_id": trial_id,
                "step_index": 2,
                "action_type": "bayesian_update",
                "action_json": json.dumps({"type": "bayesian_update", "response_id": response_id}),
                "observation_ref": response_id,
                "belief_json": json.dumps(beliefs["posterior"], sort_keys=True),
                "probability": posterior_probability,
            },
        ],
    )


def _prior_components_for_example_duckdb(conn, example_id: str) -> dict:
    market_prior = _market_prior_for_example_duckdb(conn, example_id)
    historical_prior, historical_scope, historical_count = _historical_empirical_prior_duckdb(
        conn,
        example_id,
    )
    combined_prior = _combine_prior_components(market_prior, historical_prior)
    return {
        "market_implied_prior_probability": float(market_prior),
        "historical_empirical_prior_probability": historical_prior,
        "historical_empirical_scope": historical_scope,
        "historical_empirical_count": historical_count,
        "historical_signal_weight": 0.5 if historical_prior is not None else 0.0,
        "combined_prior_probability": float(combined_prior),
    }


def _prior_components_for_example_postgres(conn, sqlalchemy, example_id: str) -> dict:
    market_prior = _market_prior_for_example_postgres(conn, sqlalchemy, example_id)
    historical_prior, historical_scope, historical_count = _historical_empirical_prior_postgres(
        conn,
        sqlalchemy,
        example_id,
    )
    combined_prior = _combine_prior_components(market_prior, historical_prior)
    return {
        "market_implied_prior_probability": float(market_prior),
        "historical_empirical_prior_probability": historical_prior,
        "historical_empirical_scope": historical_scope,
        "historical_empirical_count": historical_count,
        "historical_signal_weight": 0.5 if historical_prior is not None else 0.0,
        "combined_prior_probability": float(combined_prior),
    }


def _market_prior_for_example_duckdb(conn, example_id: str) -> float:
    row = conn.execute(
        """
        SELECT e.spot, e.strike, e.dte, s.implied_volatility, s.delta
        FROM option_examples e
        LEFT JOIN option_chain_snapshots s
          ON s.symbol = e.symbol
         AND s.option_symbol = e.option_symbol
         AND s.quote_timestamp = e.forecast_timestamp
        WHERE e.example_id = ?
        """,
        [example_id],
    ).fetchone()
    if row is None:
        return 0.5
    return _market_prior_from_snapshot(*row)


def _market_prior_for_example_postgres(conn, sqlalchemy, example_id: str) -> float:
    row = conn.execute(
        sqlalchemy.text(
            """
            SELECT e.spot, e.strike, e.dte, s.implied_volatility, s.delta
            FROM option_examples e
            LEFT JOIN option_chain_snapshots s
              ON s.symbol = e.symbol
             AND s.option_symbol = e.option_symbol
             AND s.quote_timestamp = e.forecast_timestamp
            WHERE e.example_id = :example_id
            """
        ),
        {"example_id": example_id},
    ).fetchone()
    if row is None:
        return 0.5
    return _market_prior_from_snapshot(row.spot, row.strike, row.dte, row.implied_volatility, row.delta)


def _historical_empirical_prior_duckdb(conn, example_id: str) -> tuple[float | None, str | None, int]:
    row = conn.execute(
        """
        SELECT symbol, forecast_timestamp, dte, moneyness
        FROM option_examples
        WHERE example_id = ?
        """,
        [example_id],
    ).fetchone()
    if row is None:
        return None, None, 0
    symbol, forecast_timestamp, dte, moneyness = row
    scoped = conn.execute(
        """
        SELECT count(*), coalesce(sum(label), 0)
        FROM option_examples
        WHERE label IS NOT NULL
          AND forecast_timestamp < ?
          AND symbol = ?
          AND abs(moneyness - ?) <= 0.02
          AND abs(dte - ?) <= 7
        """,
        [forecast_timestamp, symbol, moneyness, dte],
    ).fetchone()
    prior = _smoothed_historical_prior(scoped[1], scoped[0], min_count=20)
    if prior is not None:
        return prior, "symbol_moneyness_dte", int(scoped[0])

    symbol_wide = conn.execute(
        """
        SELECT count(*), coalesce(sum(label), 0)
        FROM option_examples
        WHERE label IS NOT NULL
          AND forecast_timestamp < ?
          AND symbol = ?
        """,
        [forecast_timestamp, symbol],
    ).fetchone()
    prior = _smoothed_historical_prior(symbol_wide[1], symbol_wide[0], min_count=20)
    if prior is not None:
        return prior, "symbol", int(symbol_wide[0])

    global_wide = conn.execute(
        """
        SELECT count(*), coalesce(sum(label), 0)
        FROM option_examples
        WHERE label IS NOT NULL
          AND forecast_timestamp < ?
        """,
        [forecast_timestamp],
    ).fetchone()
    prior = _smoothed_historical_prior(global_wide[1], global_wide[0], min_count=50)
    if prior is not None:
        return prior, "global", int(global_wide[0])
    return None, None, 0


def _historical_empirical_prior_postgres(
    conn,
    sqlalchemy,
    example_id: str,
) -> tuple[float | None, str | None, int]:
    row = conn.execute(
        sqlalchemy.text(
            """
            SELECT symbol, forecast_timestamp, dte, moneyness
            FROM option_examples
            WHERE example_id = :example_id
            """
        ),
        {"example_id": example_id},
    ).fetchone()
    if row is None:
        return None, None, 0
    scoped = conn.execute(
        sqlalchemy.text(
            """
            SELECT count(*), coalesce(sum(label), 0)
            FROM option_examples
            WHERE label IS NOT NULL
              AND forecast_timestamp < :forecast_timestamp
              AND symbol = :symbol
              AND abs(moneyness - :moneyness) <= 0.02
              AND abs(dte - :dte) <= 7
            """
        ),
        {
            "forecast_timestamp": row.forecast_timestamp,
            "symbol": row.symbol,
            "moneyness": row.moneyness,
            "dte": row.dte,
        },
    ).fetchone()
    prior = _smoothed_historical_prior(scoped[1], scoped[0], min_count=20)
    if prior is not None:
        return prior, "symbol_moneyness_dte", int(scoped[0])

    symbol_wide = conn.execute(
        sqlalchemy.text(
            """
            SELECT count(*), coalesce(sum(label), 0)
            FROM option_examples
            WHERE label IS NOT NULL
              AND forecast_timestamp < :forecast_timestamp
              AND symbol = :symbol
            """
        ),
        {"forecast_timestamp": row.forecast_timestamp, "symbol": row.symbol},
    ).fetchone()
    prior = _smoothed_historical_prior(symbol_wide[1], symbol_wide[0], min_count=20)
    if prior is not None:
        return prior, "symbol", int(symbol_wide[0])

    global_wide = conn.execute(
        sqlalchemy.text(
            """
            SELECT count(*), coalesce(sum(label), 0)
            FROM option_examples
            WHERE label IS NOT NULL
              AND forecast_timestamp < :forecast_timestamp
            """
        ),
        {"forecast_timestamp": row.forecast_timestamp},
    ).fetchone()
    prior = _smoothed_historical_prior(global_wide[1], global_wide[0], min_count=50)
    if prior is not None:
        return prior, "global", int(global_wide[0])
    return None, None, 0


def _smoothed_historical_prior(successes, total, min_count: int) -> float | None:
    total = int(total or 0)
    if total < min_count:
        return None
    return float((float(successes or 0) + 1.0) / (total + 2.0))


def _combine_prior_components(market_prior: float, historical_prior: float | None) -> float:
    if historical_prior is None:
        return float(market_prior)
    return bayesian_binary_update(market_prior, historical_prior, signal_weight=0.5)


def _market_prior_from_snapshot(
    spot,
    strike,
    dte,
    implied_volatility,
    delta,
) -> float:
    try:
        if implied_volatility is not None and float(implied_volatility) > 0:
            return risk_neutral_call_itm_probability(
                spot=float(spot),
                strike=float(strike),
                dte=float(dte),
                volatility=float(implied_volatility),
            )
    except (TypeError, ValueError):
        pass
    try:
        delta_probability = delta_as_probability(None if delta is None else float(delta))
    except (TypeError, ValueError):
        delta_probability = None
    return 0.5 if delta_probability is None else float(delta_probability)


def _generate_live_questions_duckdb(
    conn,
    max_abs_moneyness: float,
    min_dte: float,
    max_dte: float,
    max_questions: int,
    min_open_interest: float,
    min_volume: float,
    strike_window_size: int,
    max_questions_per_ticker: int | None,
    source: str,
) -> list[LiveQuestion]:
    del max_abs_moneyness
    effective_max_questions_per_ticker = (
        max_questions if max_questions_per_ticker is None else max_questions_per_ticker
    )
    rows = conn.execute(
        """
        WITH latest AS (
          SELECT symbol, max(quote_timestamp) AS quote_timestamp
          FROM option_chain_snapshots
          GROUP BY symbol
        ),
        candidates AS (
          SELECT
            s.*,
            date_diff('second', s.quote_timestamp, s.expiration) / 86400.0 AS dte,
            s.spot / s.strike - 1.0 AS moneyness,
            CASE WHEN s.strike < s.spot THEN 'below' ELSE 'above' END AS strike_side,
            row_number() OVER (
              PARTITION BY
                s.symbol,
                s.expiration,
                CASE WHEN s.strike < s.spot THEN 'below' ELSE 'above' END
              ORDER BY
                CASE WHEN s.strike < s.spot THEN s.strike END DESC,
                CASE WHEN s.strike >= s.spot THEN s.strike END ASC
            ) AS strike_side_rank
          FROM option_chain_snapshots s
          JOIN latest
            ON latest.symbol = s.symbol
           AND latest.quote_timestamp = s.quote_timestamp
          WHERE s.option_right = 'C'
            AND s.spot IS NOT NULL
            AND s.strike > 0
            AND date_diff('second', s.quote_timestamp, s.expiration) / 86400.0 BETWEEN ? AND ?
            AND coalesce(s.open_interest, 0) >= ?
            AND coalesce(s.volume, 0) >= ?
            AND NOT EXISTS (
              SELECT 1
              FROM option_examples e
              JOIN live_questions q ON q.example_id = e.example_id
              WHERE e.symbol = s.symbol
                AND e.expiration = s.expiration
                AND e.strike = s.strike
                AND date_trunc('hour', e.forecast_timestamp)
                    = date_trunc('hour', s.quote_timestamp)
            )
        ),
        ladder AS (
          SELECT *
          FROM candidates
          WHERE strike_side_rank <= ?
        ),
        ranked AS (
          SELECT
            *,
            row_number() OVER (
              PARTITION BY symbol
              ORDER BY expiration, strike_side_rank, strike_side, strike
            ) AS symbol_question_rank
          FROM ladder
        )
        SELECT
          symbol, option_symbol, quote_timestamp, expiration, strike, spot, dte, moneyness
        FROM ranked
        WHERE strike_side_rank <= ?
          AND symbol_question_rank <= ?
        ORDER BY symbol_question_rank, symbol, expiration, strike_side_rank, strike_side, strike
        LIMIT ?
        """,
        [
            min_dte,
            max_dte,
            min_open_interest,
            min_volume,
            strike_window_size,
            strike_window_size,
            effective_max_questions_per_ticker,
            max_questions,
        ],
    ).fetchall()

    questions = [_live_question_from_candidate(row) for row in rows]
    now = datetime.now().astimezone()
    for question, row in zip(questions, rows, strict=True):
        symbol, option_symbol, quote_timestamp, expiration, strike, spot, dte, money = row
        conn.execute(
            """
            INSERT OR REPLACE INTO option_examples
              (example_id, symbol, option_symbol, forecast_timestamp, expiration,
               strike, spot, dte, moneyness, label, resolution_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """,
            [
                question.example_id,
                symbol,
                option_symbol,
                quote_timestamp,
                expiration,
                strike,
                spot,
                dte,
                money,
                expiration,
            ],
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO live_questions
              (question_id, example_id, question_text, generated_at,
               information_cutoff, resolution_due, status, source)
            VALUES (?, ?, ?, ?, ?, ?, 'pending_prediction', ?)
            """,
            [
                question.question_id,
                question.example_id,
                question.question_text,
                now,
                question.information_cutoff,
                question.expiration,
                source,
            ],
        )
    return questions


def _live_question_from_candidate(row) -> LiveQuestion:
    symbol, option_symbol, quote_timestamp, expiration, strike, _spot, _dte, _money = row[:8]
    example_id = make_example_id(symbol, option_symbol, quote_timestamp)
    return LiveQuestion(
        question_id=str(uuid4()),
        example_id=example_id,
        symbol=symbol,
        strike=float(strike),
        expiration=expiration,
        forecast_timestamp=quote_timestamp,
        information_cutoff=quote_timestamp,
        question_text=call_option_question_text(symbol, float(strike), expiration),
    )


def _pending_prediction_prompts_duckdb(conn, limit: int | None) -> list[dict]:
    limit_sql = "" if limit is None else f"LIMIT {int(limit)}"
    rows = conn.execute(
        f"""
        SELECT question_id, example_id, question_text, generated_at,
               information_cutoff, resolution_due
        FROM live_questions
        WHERE status = 'pending_prediction'
        ORDER BY generated_at, question_id
        {limit_sql}
        """
    ).fetchall()
    return [_prompt_record_duckdb(conn, row) for row in rows]


def _prompt_record_duckdb(conn, row) -> dict:
    question_id, example_id, question_text, generated_at, information_cutoff, resolution_due = row
    evidence = _evidence_block_duckdb(conn, question_id)
    question = LiveQuestion(
        question_id=question_id,
        example_id=example_id,
        symbol=_symbol_from_question_text(question_text),
        strike=0.0,
        expiration=resolution_due,
        forecast_timestamp=generated_at,
        information_cutoff=information_cutoff,
        question_text=question_text,
    )
    return {
        "question_id": question_id,
        "example_id": example_id,
        "information_cutoff": information_cutoff,
        "resolution_due": resolution_due,
        "prompt": live_llm_prompt(question, evidence),
    }


def _prompt_record_postgres(conn, row) -> dict:
    question_id = row.question_id
    evidence = _evidence_block_postgres(conn, question_id)
    question = LiveQuestion(
        question_id=question_id,
        example_id=row.example_id,
        symbol=_symbol_from_question_text(row.question_text),
        strike=0.0,
        expiration=row.resolution_due,
        forecast_timestamp=row.generated_at,
        information_cutoff=row.information_cutoff,
        question_text=row.question_text,
    )
    return {
        "question_id": question_id,
        "example_id": row.example_id,
        "information_cutoff": row.information_cutoff,
        "resolution_due": row.resolution_due,
        "prompt": live_llm_prompt(question, evidence),
    }


def _evidence_block_duckdb(conn, question_id: str) -> str:
    row = conn.execute(
        """
        SELECT
          e.symbol, e.forecast_timestamp, q.information_cutoff, e.expiration,
          e.strike, e.spot, e.dte, e.moneyness,
          s.bid, s.ask, s.mid, s.implied_volatility, s.delta, s.volume, s.open_interest,
          e.example_id
        FROM live_questions q
        JOIN option_examples e ON e.example_id = q.example_id
        LEFT JOIN option_chain_snapshots s
          ON s.symbol = e.symbol
         AND s.option_symbol = e.option_symbol
         AND s.quote_timestamp = e.forecast_timestamp
        WHERE q.question_id = ?
        """,
        [question_id],
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown question_id {question_id}")
    bars = conn.execute(
        """
        SELECT timestamp, open, high, low, close, volume
        FROM underlying_bars
        WHERE symbol = ?
          AND timestamp <= ?
        ORDER BY timestamp DESC
        LIMIT 5
        """,
        [row[0], row[1]],
    ).fetchall()
    news_items = _news_context_items_duckdb(conn, symbol=row[0], information_cutoff=row[2])
    prior_components = _prior_components_for_example_duckdb(conn, row[15])
    return _format_evidence_block(row, bars, news_items, prior_components)


def _evidence_block_postgres(conn, question_id: str) -> str:
    import sqlalchemy

    row = conn.execute(
        sqlalchemy.text(
            """
            SELECT
              e.symbol, e.forecast_timestamp, q.information_cutoff, e.expiration,
              e.strike, e.spot, e.dte, e.moneyness,
              s.bid, s.ask, s.mid, s.implied_volatility, s.delta, s.volume, s.open_interest,
              e.example_id
            FROM live_questions q
            JOIN option_examples e ON e.example_id = q.example_id
            LEFT JOIN option_chain_snapshots s
              ON s.symbol = e.symbol
             AND s.option_symbol = e.option_symbol
             AND s.quote_timestamp = e.forecast_timestamp
            WHERE q.question_id = :question_id
            """
        ),
        {"question_id": question_id},
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown question_id {question_id}")
    bars = conn.execute(
        sqlalchemy.text(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM underlying_bars
            WHERE symbol = :symbol
              AND timestamp <= :forecast_timestamp
            ORDER BY timestamp DESC
            LIMIT 5
            """
        ),
        {"symbol": row.symbol, "forecast_timestamp": row.forecast_timestamp},
    ).fetchall()
    news_items = _news_context_items_postgres(
        conn,
        sqlalchemy,
        symbol=row.symbol,
        information_cutoff=row.information_cutoff,
    )
    prior_components = _prior_components_for_example_postgres(conn, sqlalchemy, row.example_id)
    return _format_evidence_block(tuple(row), [tuple(bar) for bar in bars], news_items, prior_components)


def _format_evidence_block(
    row,
    bars,
    news_items: list[dict] | None = None,
    prior_components: dict | None = None,
) -> str:
    (
        symbol,
        forecast_timestamp,
        information_cutoff,
        expiration,
        strike,
        spot,
        dte,
        money,
        bid,
        ask,
        mid,
        iv,
        delta,
        volume,
        open_interest,
        _example_id,
    ) = row
    market_prior = _market_prior_from_snapshot(spot, strike, dte, iv, delta)
    prior_components = prior_components or {
        "market_implied_prior_probability": market_prior,
        "historical_empirical_prior_probability": None,
        "historical_empirical_scope": None,
        "historical_empirical_count": 0,
        "combined_prior_probability": market_prior,
    }
    bar_lines = [
        "recent_underlying_bars_most_recent_first:",
        *[
            f"- {timestamp}: open={open_} high={high} low={low} close={close} volume={volume}"
            for timestamp, open_, high, low, close, volume in bars
        ],
    ]
    news_lines = ["cached_web_news_context:"]
    for item in news_items or []:
        news_lines.append(
            "- "
            f"{item.get('published_at') or 'unknown_time'} "
            f"{item.get('title')} "
            f"source={item.get('source')} "
            f"link={item.get('link')}"
        )
    if len(news_lines) == 1:
        news_lines.append("- none_cached_for_this_forecast")
    return "\n".join(
        [
            f"symbol: {symbol}",
            f"forecast_timestamp: {forecast_timestamp}",
            f"information_cutoff: {information_cutoff}",
            f"expiration: {expiration}",
            f"strike: {strike}",
            f"spot_at_forecast: {spot}",
            f"days_to_expiration: {float(dte):.3f}",
            f"moneyness_spot_over_strike_minus_one: {float(money):.6f}",
            f"bid: {bid}",
            f"ask: {ask}",
            f"mid: {mid}",
            f"implied_volatility: {iv}",
            f"delta: {delta}",
            "market_implied_prior_probability: "
            f"{float(prior_components['market_implied_prior_probability']):.6f}",
            "historical_empirical_prior_probability: "
            f"{prior_components['historical_empirical_prior_probability']}",
            f"historical_empirical_scope: {prior_components['historical_empirical_scope']}",
            f"historical_empirical_count: {prior_components['historical_empirical_count']}",
            "combined_prior_probability: "
            f"{float(prior_components['combined_prior_probability']):.6f}",
            f"volume: {volume}",
            f"open_interest: {open_interest}",
            *bar_lines,
            *news_lines,
        ]
    )


def _news_context_items_duckdb(conn, symbol: str, information_cutoff) -> list[dict]:
    rows = conn.execute(
        """
        SELECT response_json
        FROM external_call_cache
        WHERE call_type = 'web_news_context'
          AND source_timestamp <= ?
        ORDER BY source_timestamp DESC, captured_at DESC
        LIMIT 10
        """,
        [information_cutoff],
    ).fetchall()
    return _news_context_items_from_cache_payloads([row[0] for row in rows], symbol)


def _news_context_items_postgres(conn, sqlalchemy, symbol: str, information_cutoff) -> list[dict]:
    rows = conn.execute(
        sqlalchemy.text(
            """
            SELECT response_json
            FROM external_call_cache
            WHERE call_type = 'web_news_context'
              AND source_timestamp <= :information_cutoff
            ORDER BY source_timestamp DESC, captured_at DESC
            LIMIT 10
            """
        ),
        {"information_cutoff": information_cutoff},
    ).fetchall()
    return _news_context_items_from_cache_payloads([row.response_json for row in rows], symbol)


def _news_context_items_from_cache_payloads(payloads: list, symbol: str) -> list[dict]:
    items: list[dict] = []
    for payload in payloads:
        parsed = _json_value(payload)
        if not isinstance(parsed, dict):
            continue
        for item in parsed.get("items", []):
            if isinstance(item, dict) and str(item.get("ticker", "")).upper() == symbol.upper():
                items.append(item)
    return items[:5]


def _live_ticker_summary_duckdb(conn, ticker: str, limit: int) -> dict:
    latest_snapshot = conn.execute(
        """
        WITH latest AS (
          SELECT max(quote_timestamp) AS quote_timestamp
          FROM option_chain_snapshots
          WHERE symbol = ?
        )
        SELECT
          latest.quote_timestamp,
          count(*) AS option_snapshot_count,
          sum(CASE WHEN option_right = 'C' THEN 1 ELSE 0 END) AS call_snapshot_count,
          avg(spot) AS avg_spot
        FROM latest
        LEFT JOIN option_chain_snapshots s
          ON s.symbol = ?
         AND s.quote_timestamp = latest.quote_timestamp
        GROUP BY latest.quote_timestamp
        """,
        [ticker, ticker],
    ).fetchone()
    questions = _rows_as_dicts(
        conn.execute(
            """
            SELECT
              q.question_id, q.question_text, q.status, q.generated_at, q.information_cutoff,
              e.example_id, e.option_symbol, e.forecast_timestamp, e.expiration,
              e.strike, e.spot, e.dte, e.moneyness,
              r.response_id, r.model, r.probability, r.reasoning, r.raw_response
            FROM live_questions q
            JOIN option_examples e ON e.example_id = q.example_id
            LEFT JOIN llm_responses r ON r.question_id = q.question_id
            WHERE e.symbol = ?
            ORDER BY q.generated_at DESC, q.question_id
            LIMIT ?
            """,
            [ticker, limit],
        )
    )
    belief_steps = _rows_as_dicts(
        conn.execute(
            """
            SELECT
              t.trial_id, t.example_id, t.model, t.raw_probability, t.status,
              s.step_index, s.action_type, s.action_json, s.observation_ref,
              s.belief_json, s.probability
            FROM agent_trials t
            JOIN agent_steps s ON s.trial_id = t.trial_id
            JOIN option_examples e ON e.example_id = t.example_id
            WHERE e.symbol = ?
            ORDER BY t.started_at DESC, t.trial_id, s.step_index
            LIMIT ?
            """,
            [ticker, limit * 2],
        )
    )
    return {
        "ticker": ticker,
        "latest_snapshot": _snapshot_summary_row(latest_snapshot),
        "questions": questions,
        "belief_steps": belief_steps,
    }


def _live_ticker_summary_postgres(conn, sqlalchemy, ticker: str, limit: int) -> dict:
    latest_snapshot = conn.execute(
        sqlalchemy.text(
            """
            WITH latest AS (
              SELECT max(quote_timestamp) AS quote_timestamp
              FROM option_chain_snapshots
              WHERE symbol = :ticker
            )
            SELECT
              latest.quote_timestamp,
              count(*) AS option_snapshot_count,
              sum(CASE WHEN option_right = 'C' THEN 1 ELSE 0 END) AS call_snapshot_count,
              avg(spot) AS avg_spot
            FROM latest
            LEFT JOIN option_chain_snapshots s
              ON s.symbol = :ticker
             AND s.quote_timestamp = latest.quote_timestamp
            GROUP BY latest.quote_timestamp
            """
        ),
        {"ticker": ticker},
    ).fetchone()
    questions = [
        dict(row._mapping)
        for row in conn.execute(
            sqlalchemy.text(
                """
                SELECT
                  q.question_id, q.question_text, q.status, q.generated_at, q.information_cutoff,
                  e.example_id, e.option_symbol, e.forecast_timestamp, e.expiration,
                  e.strike, e.spot, e.dte, e.moneyness,
                  r.response_id, r.model, r.probability, r.reasoning, r.raw_response
                FROM live_questions q
                JOIN option_examples e ON e.example_id = q.example_id
                LEFT JOIN llm_responses r ON r.question_id = q.question_id
                WHERE e.symbol = :ticker
                ORDER BY q.generated_at DESC, q.question_id
                LIMIT :limit
                """
            ),
            {"ticker": ticker, "limit": limit},
        ).fetchall()
    ]
    belief_steps = [
        dict(row._mapping)
        for row in conn.execute(
            sqlalchemy.text(
                """
                SELECT
                  t.trial_id, t.example_id, t.model, t.raw_probability, t.status,
                  s.step_index, s.action_type, s.action_json, s.observation_ref,
                  s.belief_json, s.probability
                FROM agent_trials t
                JOIN agent_steps s ON s.trial_id = t.trial_id
                JOIN option_examples e ON e.example_id = t.example_id
                WHERE e.symbol = :ticker
                ORDER BY t.started_at DESC, t.trial_id, s.step_index
                LIMIT :limit
                """
            ),
            {"ticker": ticker, "limit": limit * 2},
        ).fetchall()
    ]
    return {
        "ticker": ticker,
        "latest_snapshot": _snapshot_summary_row(latest_snapshot),
        "questions": questions,
        "belief_steps": belief_steps,
    }


def _evaluation_report_duckdb(
    conn,
    ticker: str | None,
    limit: int,
    include_unresolved: bool,
) -> dict:
    where = []
    params: list = []
    if ticker:
        where.append("e.symbol = ?")
        params.append(ticker.upper())
    if not include_unresolved:
        where.append("lr.question_id IS NOT NULL")
    where_sql = "" if not where else "WHERE " + " AND ".join(where)
    params.append(limit)
    cursor = conn.execute(
        f"""
        SELECT
          q.question_id,
          q.question_text,
          q.generated_at,
          q.information_cutoff AS question_information_cutoff,
          q.resolution_due,
          q.status,
          e.example_id,
          e.symbol,
          e.option_symbol,
          e.forecast_timestamp,
          e.expiration,
          e.strike,
          e.spot,
          e.dte,
          e.moneyness,
          r.response_id,
          r.model,
          r.probability,
          r.created_at AS prediction_created_at,
          r.information_cutoff AS response_information_cutoff,
          bs.posterior_probability,
          lr.resolved_at,
          lr.underlying_close,
          lr.label
        FROM live_questions q
        JOIN option_examples e ON e.example_id = q.example_id
        LEFT JOIN llm_responses r ON r.question_id = q.question_id
        LEFT JOIN (
          SELECT observation_ref AS response_id, probability AS posterior_probability
          FROM agent_steps
          WHERE action_type = 'bayesian_update'
          QUALIFY row_number() OVER (
            PARTITION BY observation_ref
            ORDER BY step_index DESC
          ) = 1
        ) bs ON bs.response_id = r.response_id
        LEFT JOIN live_resolutions lr ON lr.question_id = q.question_id
        {where_sql}
        ORDER BY coalesce(r.created_at, q.generated_at) DESC, q.question_id
        LIMIT ?
        """,
        params,
    )
    rows = _rows_as_dicts(cursor)
    cache_rows = _rows_as_dicts(
        conn.execute(
            """
            SELECT
              call_id, provider, call_type, request_json, captured_at,
              information_cutoff, source_timestamp, response_sha256
            FROM external_call_cache
            ORDER BY captured_at DESC
            """
        )
    )
    return _build_evaluation_report(rows, cache_rows, ticker, include_unresolved)


def _evaluation_report_postgres(
    conn,
    sqlalchemy,
    ticker: str | None,
    limit: int,
    include_unresolved: bool,
) -> dict:
    where = []
    params: dict = {"limit": limit}
    if ticker:
        where.append("e.symbol = :ticker")
        params["ticker"] = ticker.upper()
    if not include_unresolved:
        where.append("lr.question_id IS NOT NULL")
    where_sql = "" if not where else "WHERE " + " AND ".join(where)
    rows = [
        dict(row._mapping)
        for row in conn.execute(
            sqlalchemy.text(
                f"""
                SELECT
                  q.question_id,
                  q.question_text,
                  q.generated_at,
                  q.information_cutoff AS question_information_cutoff,
                  q.resolution_due,
                  q.status,
                  e.example_id,
                  e.symbol,
                  e.option_symbol,
                  e.forecast_timestamp,
                  e.expiration,
                  e.strike,
                  e.spot,
                  e.dte,
                  e.moneyness,
                  r.response_id,
                  r.model,
                  r.probability,
                  r.created_at AS prediction_created_at,
                  r.information_cutoff AS response_information_cutoff,
                  bs.posterior_probability,
                  lr.resolved_at,
                  lr.underlying_close,
                  lr.label
                FROM live_questions q
                JOIN option_examples e ON e.example_id = q.example_id
                LEFT JOIN llm_responses r ON r.question_id = q.question_id
                LEFT JOIN (
                  SELECT response_id, posterior_probability
                  FROM (
                    SELECT
                      observation_ref AS response_id,
                      probability AS posterior_probability,
                      row_number() OVER (
                        PARTITION BY observation_ref
                        ORDER BY step_index DESC
                      ) AS rn
                    FROM agent_steps
                    WHERE action_type = 'bayesian_update'
                  ) latest_belief
                  WHERE rn = 1
                ) bs ON bs.response_id = r.response_id
                LEFT JOIN live_resolutions lr ON lr.question_id = q.question_id
                {where_sql}
                ORDER BY coalesce(r.created_at, q.generated_at) DESC, q.question_id
                LIMIT :limit
                """
            ),
            params,
        ).fetchall()
    ]
    cache_rows = [
        dict(row._mapping)
        for row in conn.execute(
            sqlalchemy.text(
                """
                SELECT
                  call_id, provider, call_type, request_json, captured_at,
                  information_cutoff, source_timestamp, response_sha256
                FROM external_call_cache
                ORDER BY captured_at DESC
                """
            )
        ).fetchall()
    ]
    return _build_evaluation_report(rows, cache_rows, ticker, include_unresolved)


def _build_evaluation_report(
    rows: list[dict],
    cache_rows: list[dict],
    ticker: str | None,
    include_unresolved: bool,
) -> dict:
    llm_cache_by_question: dict[str, list[dict]] = {}
    market_cache: list[dict] = []
    for cache in cache_rows:
        payload = _json_value(cache.get("request_json"))
        cache = {**cache, "request_json": payload}
        if cache.get("call_type") == "llm_prediction":
            question_id = payload.get("question_id") if isinstance(payload, dict) else None
            if question_id:
                llm_cache_by_question.setdefault(str(question_id), []).append(cache)
        elif cache.get("call_type") == "market_snapshot":
            market_cache.append(cache)

    items = []
    for row in rows:
        llm_cache = llm_cache_by_question.get(str(row["question_id"]), [])
        matching_market_cache = [
            cache
            for cache in market_cache
            if _same_instant(cache.get("source_timestamp"), row.get("forecast_timestamp"))
        ]
        item = _evaluation_item(row, llm_cache, matching_market_cache)
        items.append(item)

    resolved = [item for item in items if item["label"] is not None]
    scorable = [item for item in resolved if item["probability"] is not None]
    brier = None
    accuracy = None
    posterior_scorable = [
        item for item in resolved if item.get("posterior_probability") is not None
    ]
    posterior_brier = None
    posterior_accuracy = None
    if scorable:
        brier = sum(
            (float(item["probability"]) - float(item["label"])) ** 2 for item in scorable
        ) / len(scorable)
        accuracy = sum(
            int((float(item["probability"]) >= 0.5) == bool(item["label"]))
            for item in scorable
        ) / len(scorable)
    if posterior_scorable:
        posterior_brier = sum(
            (float(item["posterior_probability"]) - float(item["label"])) ** 2
            for item in posterior_scorable
        ) / len(posterior_scorable)
        posterior_accuracy = sum(
            int((float(item["posterior_probability"]) >= 0.5) == bool(item["label"]))
            for item in posterior_scorable
        ) / len(posterior_scorable)
    leakage_failures = [
        item
        for item in items
        if not all(item["leakage_checks"].values())
    ]
    return {
        "ticker": None if ticker is None else ticker.upper(),
        "include_unresolved": include_unresolved,
        "summary": {
            "n_predictions": len(items),
            "n_resolved": len(resolved),
            "n_unresolved": len(items) - len(resolved),
            "n_scorable": len(scorable),
            "n_posterior_scorable": len(posterior_scorable),
            "n_leakage_check_failures": len(leakage_failures),
            "brier_score": brier,
            "accuracy_at_0_5": accuracy,
            "posterior_brier_score": posterior_brier,
            "posterior_accuracy_at_0_5": posterior_accuracy,
        },
        "predictions": items,
    }


def _operational_status_duckdb(conn) -> dict:
    option_count, latest_option = conn.execute(
        "SELECT count(*), max(quote_timestamp) FROM option_chain_snapshots"
    ).fetchone()
    bar_count, latest_bar = conn.execute(
        "SELECT count(*), max(timestamp) FROM underlying_bars"
    ).fetchone()
    total_questions = conn.execute("SELECT count(*) FROM live_questions").fetchone()[0]
    status_counts = {
        status: count
        for status, count in conn.execute(
            "SELECT status, count(*) FROM live_questions GROUP BY status"
        ).fetchall()
    }
    due_unresolved = conn.execute(
        """
        SELECT count(*)
        FROM live_questions q
        LEFT JOIN live_resolutions r ON r.question_id = q.question_id
        WHERE r.question_id IS NULL
          AND q.resolution_due <= current_timestamp
        """
    ).fetchone()[0]
    next_resolution_due = conn.execute(
        """
        SELECT min(q.resolution_due)
        FROM live_questions q
        LEFT JOIN live_resolutions r ON r.question_id = q.question_id
        WHERE r.question_id IS NULL
        """
    ).fetchone()[0]
    due_today = conn.execute(
        """
        SELECT count(*)
        FROM live_questions q
        LEFT JOIN live_resolutions r ON r.question_id = q.question_id
        WHERE r.question_id IS NULL
          AND CAST(q.resolution_due AS DATE) = current_date
        """
    ).fetchone()[0]
    due_next_24h = conn.execute(
        """
        SELECT count(*)
        FROM live_questions q
        LEFT JOIN live_resolutions r ON r.question_id = q.question_id
        WHERE r.question_id IS NULL
          AND q.resolution_due > current_timestamp
          AND q.resolution_due <= current_timestamp + INTERVAL 1 DAY
        """
    ).fetchone()[0]
    due_next_72h = conn.execute(
        """
        SELECT count(*)
        FROM live_questions q
        LEFT JOIN live_resolutions r ON r.question_id = q.question_id
        WHERE r.question_id IS NULL
          AND q.resolution_due > current_timestamp
          AND q.resolution_due <= current_timestamp + INTERVAL 3 DAY
        """
    ).fetchone()[0]
    oldest_due_unresolved = conn.execute(
        """
        SELECT min(q.resolution_due)
        FROM live_questions q
        LEFT JOIN live_resolutions r ON r.question_id = q.question_id
        WHERE r.question_id IS NULL
          AND q.resolution_due <= current_timestamp
        """
    ).fetchone()[0]
    prediction_count, latest_prediction = conn.execute(
        "SELECT count(*), max(created_at) FROM llm_responses"
    ).fetchone()
    resolution_count, latest_resolution = conn.execute(
        "SELECT count(*), max(resolved_at) FROM live_resolutions"
    ).fetchone()
    cache_count, latest_cache = conn.execute(
        "SELECT count(*), max(captured_at) FROM external_call_cache"
    ).fetchone()
    cache_type_counts = {
        call_type: count
        for call_type, count in conn.execute(
            "SELECT call_type, count(*) FROM external_call_cache GROUP BY call_type"
        ).fetchall()
    }
    latest_runs = [
        {"run_type": run_type, "started_at": started_at, "completed_at": completed_at, "status": status}
        for run_type, started_at, completed_at, status in conn.execute(
            """
            SELECT run_type, started_at, completed_at, status
            FROM (
              SELECT *, row_number() OVER (PARTITION BY run_type ORDER BY started_at DESC) AS rn
              FROM daily_runs
            )
            WHERE rn = 1
            ORDER BY run_type
            """
        ).fetchall()
    ]
    return _operational_status_payload(
        option_count=option_count,
        latest_option=latest_option,
        bar_count=bar_count,
        latest_bar=latest_bar,
        total_questions=total_questions,
        status_counts=status_counts,
        due_unresolved=due_unresolved,
        next_resolution_due=next_resolution_due,
        due_today=due_today,
        due_next_24h=due_next_24h,
        due_next_72h=due_next_72h,
        oldest_due_unresolved=oldest_due_unresolved,
        prediction_count=prediction_count,
        latest_prediction=latest_prediction,
        resolution_count=resolution_count,
        latest_resolution=latest_resolution,
        cache_count=cache_count,
        latest_cache=latest_cache,
        cache_type_counts=cache_type_counts,
        latest_runs=latest_runs,
    )


def _operational_status_postgres(conn, sqlalchemy) -> dict:
    option_count, latest_option = conn.execute(
        sqlalchemy.text("SELECT count(*), max(quote_timestamp) FROM option_chain_snapshots")
    ).fetchone()
    bar_count, latest_bar = conn.execute(
        sqlalchemy.text("SELECT count(*), max(timestamp) FROM underlying_bars")
    ).fetchone()
    total_questions = conn.execute(sqlalchemy.text("SELECT count(*) FROM live_questions")).fetchone()[0]
    status_counts = {
        row.status: row[1]
        for row in conn.execute(
            sqlalchemy.text("SELECT status, count(*) FROM live_questions GROUP BY status")
        ).fetchall()
    }
    due_unresolved = conn.execute(
        sqlalchemy.text(
            """
            SELECT count(*)
            FROM live_questions q
            LEFT JOIN live_resolutions r ON r.question_id = q.question_id
            WHERE r.question_id IS NULL
              AND q.resolution_due <= now()
            """
        )
    ).fetchone()[0]
    next_resolution_due = conn.execute(
        sqlalchemy.text(
            """
            SELECT min(q.resolution_due)
            FROM live_questions q
            LEFT JOIN live_resolutions r ON r.question_id = q.question_id
            WHERE r.question_id IS NULL
            """
        )
    ).fetchone()[0]
    due_today = conn.execute(
        sqlalchemy.text(
            """
            SELECT count(*)
            FROM live_questions q
            LEFT JOIN live_resolutions r ON r.question_id = q.question_id
            WHERE r.question_id IS NULL
              AND CAST(q.resolution_due AS DATE) = current_date
            """
        )
    ).fetchone()[0]
    due_next_24h = conn.execute(
        sqlalchemy.text(
            """
            SELECT count(*)
            FROM live_questions q
            LEFT JOIN live_resolutions r ON r.question_id = q.question_id
            WHERE r.question_id IS NULL
              AND q.resolution_due > now()
              AND q.resolution_due <= now() + interval '1 day'
            """
        )
    ).fetchone()[0]
    due_next_72h = conn.execute(
        sqlalchemy.text(
            """
            SELECT count(*)
            FROM live_questions q
            LEFT JOIN live_resolutions r ON r.question_id = q.question_id
            WHERE r.question_id IS NULL
              AND q.resolution_due > now()
              AND q.resolution_due <= now() + interval '3 day'
            """
        )
    ).fetchone()[0]
    oldest_due_unresolved = conn.execute(
        sqlalchemy.text(
            """
            SELECT min(q.resolution_due)
            FROM live_questions q
            LEFT JOIN live_resolutions r ON r.question_id = q.question_id
            WHERE r.question_id IS NULL
              AND q.resolution_due <= now()
            """
        )
    ).fetchone()[0]
    prediction_count, latest_prediction = conn.execute(
        sqlalchemy.text("SELECT count(*), max(created_at) FROM llm_responses")
    ).fetchone()
    resolution_count, latest_resolution = conn.execute(
        sqlalchemy.text("SELECT count(*), max(resolved_at) FROM live_resolutions")
    ).fetchone()
    cache_count, latest_cache = conn.execute(
        sqlalchemy.text("SELECT count(*), max(captured_at) FROM external_call_cache")
    ).fetchone()
    cache_type_counts = {
        row.call_type: row[1]
        for row in conn.execute(
            sqlalchemy.text("SELECT call_type, count(*) FROM external_call_cache GROUP BY call_type")
        ).fetchall()
    }
    latest_runs = [
        {
            "run_type": row.run_type,
            "started_at": row.started_at,
            "completed_at": row.completed_at,
            "status": row.status,
        }
        for row in conn.execute(
            sqlalchemy.text(
                """
                SELECT run_type, started_at, completed_at, status
                FROM (
                  SELECT
                    run_type, started_at, completed_at, status,
                    row_number() OVER (PARTITION BY run_type ORDER BY started_at DESC) AS rn
                  FROM daily_runs
                ) latest
                WHERE rn = 1
                ORDER BY run_type
                """
            )
        ).fetchall()
    ]
    return _operational_status_payload(
        option_count=option_count,
        latest_option=latest_option,
        bar_count=bar_count,
        latest_bar=latest_bar,
        total_questions=total_questions,
        status_counts=status_counts,
        due_unresolved=due_unresolved,
        next_resolution_due=next_resolution_due,
        due_today=due_today,
        due_next_24h=due_next_24h,
        due_next_72h=due_next_72h,
        oldest_due_unresolved=oldest_due_unresolved,
        prediction_count=prediction_count,
        latest_prediction=latest_prediction,
        resolution_count=resolution_count,
        latest_resolution=latest_resolution,
        cache_count=cache_count,
        latest_cache=latest_cache,
        cache_type_counts=cache_type_counts,
        latest_runs=latest_runs,
    )


def _operational_status_payload(
    option_count,
    latest_option,
    bar_count,
    latest_bar,
    total_questions,
    status_counts,
    due_unresolved,
    next_resolution_due,
    due_today,
    due_next_24h,
    due_next_72h,
    oldest_due_unresolved,
    prediction_count,
    latest_prediction,
    resolution_count,
    latest_resolution,
    cache_count,
    latest_cache,
    cache_type_counts,
    latest_runs,
) -> dict:
    return {
        "option_snapshots": {
            "count": int(option_count or 0),
            "latest": latest_option,
        },
        "underlying_bars": {
            "count": int(bar_count or 0),
            "latest": latest_bar,
        },
        "live_questions": {
            "total": int(total_questions or 0),
            "status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "due_unresolved": int(due_unresolved or 0),
        },
        "resolution_readiness": {
            "next_resolution_due": next_resolution_due,
            "due_today": int(due_today or 0),
            "due_next_24h": int(due_next_24h or 0),
            "due_next_72h": int(due_next_72h or 0),
            "oldest_due_unresolved": oldest_due_unresolved,
        },
        "llm_responses": {
            "count": int(prediction_count or 0),
            "latest": latest_prediction,
        },
        "live_resolutions": {
            "count": int(resolution_count or 0),
            "latest": latest_resolution,
        },
        "external_call_cache": {
            "count": int(cache_count or 0),
            "latest": latest_cache,
            "call_type_counts": {str(key): int(value) for key, value in cache_type_counts.items()},
        },
        "daily_runs": latest_runs,
    }


def _evaluation_item(row: dict, llm_cache: list[dict], market_cache: list[dict]) -> dict:
    probability = row.get("probability")
    posterior_probability = row.get("posterior_probability")
    label = row.get("label")
    prediction_created_at = row.get("prediction_created_at")
    resolution_due = row.get("resolution_due")
    resolved_at = row.get("resolved_at")
    question_cutoff = row.get("question_information_cutoff")
    response_cutoff = row.get("response_information_cutoff")
    latest_llm_cache = llm_cache[0] if llm_cache else None
    latest_market_cache = market_cache[0] if market_cache else None
    leakage_checks = {
        "question_cutoff_not_after_prediction": _lte_or_unknown(
            question_cutoff,
            prediction_created_at,
        ),
        "response_cutoff_not_after_prediction": _lte_or_unknown(
            response_cutoff,
            prediction_created_at,
        ),
        "prediction_not_after_resolution_due": _lte_or_unknown(
            prediction_created_at,
            resolution_due,
        ),
        "prediction_not_after_resolved_at": (
            True if resolved_at is None else _lte_or_unknown(prediction_created_at, resolution_due)
        ),
        "llm_cache_captured_not_after_resolution_due": (
            True
            if latest_llm_cache is None
            else _lte_or_unknown(latest_llm_cache.get("captured_at"), resolution_due)
        ),
        "market_cache_captured_not_after_prediction": (
            True
            if latest_market_cache is None
            else _lte_or_unknown(latest_market_cache.get("captured_at"), prediction_created_at)
        ),
    }
    return {
        "question_id": row.get("question_id"),
        "response_id": row.get("response_id"),
        "symbol": row.get("symbol"),
        "question_text": row.get("question_text"),
        "status": row.get("status"),
        "forecast_timestamp": row.get("forecast_timestamp"),
        "question_generated_at": row.get("generated_at"),
        "information_cutoff": question_cutoff,
        "prediction_created_at": prediction_created_at,
        "resolution_due": resolution_due,
        "resolved_at": resolved_at,
        "strike": row.get("strike"),
        "spot": row.get("spot"),
        "dte": row.get("dte"),
        "moneyness": row.get("moneyness"),
        "model": row.get("model"),
        "probability": probability,
        "posterior_probability": posterior_probability,
        "label": label,
        "underlying_close": row.get("underlying_close"),
        "brier": (
            None
            if probability is None or label is None
            else (float(probability) - float(label)) ** 2
        ),
        "posterior_brier": (
            None
            if posterior_probability is None or label is None
            else (float(posterior_probability) - float(label)) ** 2
        ),
        "correct_at_0_5": (
            None
            if probability is None or label is None
            else (float(probability) >= 0.5) == bool(label)
        ),
        "posterior_correct_at_0_5": (
            None
            if posterior_probability is None or label is None
            else (float(posterior_probability) >= 0.5) == bool(label)
        ),
        "external_cache": {
            "llm_prediction_records": len(llm_cache),
            "market_snapshot_records": len(market_cache),
            "latest_llm_call_id": None if latest_llm_cache is None else latest_llm_cache["call_id"],
            "latest_llm_response_sha256": (
                None if latest_llm_cache is None else latest_llm_cache["response_sha256"]
            ),
            "latest_market_call_id": (
                None if latest_market_cache is None else latest_market_cache["call_id"]
            ),
            "latest_market_response_sha256": (
                None if latest_market_cache is None else latest_market_cache["response_sha256"]
            ),
        },
        "leakage_checks": leakage_checks,
    }


def _json_value(value):
    if value is None or isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def _same_instant(left, right) -> bool:
    if left is None or right is None:
        return False
    return str(left) == str(right) or _isoish(left) == _isoish(right)


def _lte_or_unknown(left, right) -> bool:
    if left is None or right is None:
        return True
    return _isoish(left) <= _isoish(right)


def _isoish(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _snapshot_summary_row(row) -> dict | None:
    if row is None:
        return None
    values = tuple(row)
    if values[0] is None:
        return None
    return {
        "quote_timestamp": values[0],
        "option_snapshot_count": values[1],
        "call_snapshot_count": values[2],
        "avg_spot": values[3],
    }


def _rows_as_dicts(cursor) -> list[dict]:
    columns = [item[0] for item in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def _resolve_due_live_questions_duckdb(conn, require_expiration_date: bool = True) -> int:
    rows = conn.execute(
        """
        SELECT q.question_id, e.symbol, e.strike, e.expiration
        FROM live_questions q
        JOIN option_examples e ON e.example_id = q.example_id
        LEFT JOIN live_resolutions r ON r.question_id = q.question_id
        WHERE r.question_id IS NULL
          AND q.resolution_due <= current_timestamp
        """
    ).fetchall()
    resolved = 0
    for question_id, symbol, strike, expiration in rows:
        date_filter = (
            "AND CAST(timestamp AS DATE) = CAST(? AS DATE)"
            if require_expiration_date
            else ""
        )
        params = [symbol, expiration]
        if require_expiration_date:
            params.append(expiration)
        bar = conn.execute(
            f"""
            SELECT close, timestamp
            FROM underlying_bars
            WHERE symbol = ?
              AND timestamp <= ?
              {date_filter}
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
        if bar is None:
            continue
        close, resolved_at = bar
        label = int(float(close) > float(strike))
        conn.execute(
            """
            INSERT OR REPLACE INTO live_resolutions
              (question_id, resolved_at, underlying_close, label, source)
            VALUES (?, ?, ?, ?, 'underlying_bars')
            """,
            [question_id, resolved_at, close, label],
        )
        conn.execute(
            "UPDATE live_questions SET status = 'resolved' WHERE question_id = ?",
            [question_id],
        )
        conn.execute(
            """
            UPDATE option_examples
            SET label = ?, resolution_timestamp = ?
            WHERE example_id = (
              SELECT example_id FROM live_questions WHERE question_id = ?
            )
            """,
            [label, resolved_at, question_id],
        )
        resolved += 1
    return resolved


def _symbol_from_question_text(question_text: str) -> str:
    try:
        return question_text.split("$", 2)[1].split(" ", 1)[0]
    except IndexError:
        return ""
