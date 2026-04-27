from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from murphy.options_data import make_example_id


@dataclass(frozen=True)
class LiveQuestion:
    question_id: str
    example_id: str
    symbol: str
    strike: float
    expiration: datetime
    forecast_timestamp: datetime
    information_cutoff: datetime
    question_text: str


def call_option_question_text(symbol: str, strike: float, expiration: datetime) -> str:
    ticker = symbol.upper()
    return (
        f"Will the price of ${ticker} be greater than ${strike:.2f} "
        f"on {expiration.date().isoformat()}?"
    )


def live_llm_prompt(question: LiveQuestion, evidence_block: str) -> str:
    return f"""# Forecasting task
{question.question_text}

You are making this forecast at {question.forecast_timestamp.isoformat()}.
Use only information that would have been available at or before
{question.information_cutoff.isoformat()}.

# Evidence snapshot
{evidence_block}

Return strict JSON:
{{"probability": 0.0, "reasoning": "short explanation"}}
"""


def generate_live_questions_from_snapshots(
    db_path: str = "data/murphy.duckdb",
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
    """Generate unresolved questions from the latest stored option snapshots."""
    import duckdb

    del max_abs_moneyness
    effective_max_questions_per_ticker = (
        max_questions if max_questions_per_ticker is None else max_questions_per_ticker
    )
    con = duckdb.connect(db_path)
    try:
        rows = con.execute(
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

        questions: list[LiveQuestion] = []
        now = datetime.now(timezone.utc)
        for row in rows:
            symbol, option_symbol, quote_timestamp, expiration, strike, spot, dte, money = row
            example_id = make_example_id(symbol, option_symbol, quote_timestamp)
            question = LiveQuestion(
                question_id=str(uuid4()),
                example_id=example_id,
                symbol=symbol,
                strike=float(strike),
                expiration=expiration,
                forecast_timestamp=quote_timestamp,
                information_cutoff=quote_timestamp,
                question_text=call_option_question_text(symbol, float(strike), expiration),
            )
            questions.append(question)

            con.execute(
                """
                INSERT OR REPLACE INTO option_examples
                  (example_id, symbol, option_symbol, forecast_timestamp, expiration,
                   strike, spot, dte, moneyness, label, resolution_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                [
                    example_id,
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
            con.execute(
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
    finally:
        con.close()


def evidence_block_for_question(db_path: str, question_id: str) -> str:
    """Build a compact, date-safe evidence block for an LLM prompt."""
    import duckdb

    con = duckdb.connect(db_path, read_only=True)
    try:
        row = con.execute(
            """
            SELECT
              q.question_text,
              e.symbol,
              e.forecast_timestamp,
              e.expiration,
              e.strike,
              e.spot,
              e.dte,
              e.moneyness,
              s.bid,
              s.ask,
              s.mid,
              s.implied_volatility,
              s.delta,
              s.volume,
              s.open_interest
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

        (
            _question_text,
            symbol,
            forecast_timestamp,
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
        ) = row

        bars = con.execute(
            """
            SELECT timestamp, open, high, low, close, volume
            FROM underlying_bars
            WHERE symbol = ?
              AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 5
            """,
            [symbol, forecast_timestamp],
        ).fetchall()
        bar_lines = [
            "recent_underlying_bars_most_recent_first:",
            *[
                (
                    f"- {timestamp}: open={open_} high={high} low={low} "
                    f"close={close} volume={volume}"
                )
                for timestamp, open_, high, low, close, volume in bars
            ],
        ]

        return "\n".join(
            [
                f"symbol: {symbol}",
                f"forecast_timestamp: {forecast_timestamp}",
                f"expiration: {expiration}",
                f"strike: {strike}",
                f"spot_at_forecast: {spot}",
                f"days_to_expiration: {dte:.3f}",
                f"moneyness_spot_over_strike_minus_one: {money:.6f}",
                f"bid: {bid}",
                f"ask: {ask}",
                f"mid: {mid}",
                f"implied_volatility: {iv}",
                f"delta: {delta}",
                f"volume: {volume}",
                f"open_interest: {open_interest}",
                *bar_lines,
            ]
        )
    finally:
        con.close()


def export_pending_llm_prompts_jsonl(
    db_path: str = "data/murphy.duckdb",
    output_path: str | Path = "data/pending_llm_prompts.jsonl",
    limit: int | None = None,
) -> int:
    import duckdb

    con = duckdb.connect(db_path, read_only=True)
    try:
        limit_sql = "" if limit is None else f"LIMIT {int(limit)}"
        rows = con.execute(
            f"""
            SELECT
              question_id,
              example_id,
              question_text,
              generated_at,
              information_cutoff,
              resolution_due
            FROM live_questions
            WHERE status = 'pending_prediction'
            ORDER BY generated_at, question_id
            {limit_sql}
            """
        ).fetchall()
    finally:
        con.close()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for (
            question_id,
            example_id,
            question_text,
            generated_at,
            information_cutoff,
            resolution_due,
        ) in rows:
            question = LiveQuestion(
                question_id=question_id,
                example_id=example_id,
                symbol=question_text.split("$", 2)[1].split(" ", 1)[0],
                strike=0.0,
                expiration=resolution_due,
                forecast_timestamp=generated_at,
                information_cutoff=information_cutoff,
                question_text=question_text,
            )
            evidence = evidence_block_for_question(db_path, question_id)
            handle.write(
                json.dumps(
                    {
                        "question_id": question_id,
                        "example_id": example_id,
                        "information_cutoff": information_cutoff.isoformat(),
                        "resolution_due": resolution_due.isoformat(),
                        "prompt": live_llm_prompt(question, evidence),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            count += 1
    return count


def record_llm_response(
    db_path: str,
    question_id: str,
    model: str,
    prompt: str,
    probability: float,
    reasoning: str | None = None,
    raw_response: str | None = None,
) -> str:
    import duckdb

    response_id = str(uuid4())
    created_at = datetime.now(timezone.utc)
    con = duckdb.connect(db_path)
    try:
        info_cutoff = con.execute(
            "SELECT information_cutoff FROM live_questions WHERE question_id = ?",
            [question_id],
        ).fetchone()
        if info_cutoff is None:
            raise ValueError(f"unknown question_id {question_id}")
        con.execute(
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
        con.execute(
            "UPDATE live_questions SET status = 'predicted' WHERE question_id = ?",
            [question_id],
        )
        return response_id
    finally:
        con.close()


def resolve_due_live_questions(
    db_path: str = "data/murphy.duckdb",
    require_expiration_date: bool = True,
) -> int:
    """Resolve due live questions using stored underlying bars.

    This only uses data already stored in `underlying_bars`. A separate collector
    should load bars from the market-data provider before this runs.
    """
    import duckdb

    con = duckdb.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT
              q.question_id,
              e.symbol,
              e.strike,
              e.expiration
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
            bar = con.execute(
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
            con.execute(
                """
                INSERT OR REPLACE INTO live_resolutions
                  (question_id, resolved_at, underlying_close, label, source)
                VALUES (?, ?, ?, ?, 'underlying_bars')
                """,
                [question_id, resolved_at, close, label],
            )
            con.execute(
                "UPDATE live_questions SET status = 'resolved' WHERE question_id = ?",
                [question_id],
            )
            con.execute(
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
    finally:
        con.close()
