from __future__ import annotations

from pathlib import Path
from typing import Any


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS underlying_bars (
  symbol TEXT NOT NULL,
  timestamp TIMESTAMP NOT NULL,
  open DOUBLE NOT NULL,
  high DOUBLE NOT NULL,
  low DOUBLE NOT NULL,
  close DOUBLE NOT NULL,
  volume DOUBLE,
  adjusted_close DOUBLE,
  PRIMARY KEY (symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS option_chain_snapshots (
  symbol TEXT NOT NULL,
  option_symbol TEXT NOT NULL,
  quote_timestamp TIMESTAMP NOT NULL,
  expiration TIMESTAMP NOT NULL,
  strike DOUBLE NOT NULL,
  option_right TEXT NOT NULL,
  bid DOUBLE,
  ask DOUBLE,
  mid DOUBLE,
  implied_volatility DOUBLE,
  delta DOUBLE,
  gamma DOUBLE,
  theta DOUBLE,
  vega DOUBLE,
  volume DOUBLE,
  open_interest DOUBLE,
  spot DOUBLE,
  PRIMARY KEY (option_symbol, quote_timestamp)
);

CREATE TABLE IF NOT EXISTS option_examples (
  example_id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  option_symbol TEXT NOT NULL,
  forecast_timestamp TIMESTAMP NOT NULL,
  expiration TIMESTAMP NOT NULL,
  strike DOUBLE NOT NULL,
  spot DOUBLE NOT NULL,
  dte DOUBLE NOT NULL,
  moneyness DOUBLE NOT NULL,
  label INTEGER,
  resolution_timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS features (
  example_id TEXT NOT NULL,
  feature_name TEXT NOT NULL,
  feature_value DOUBLE NOT NULL,
  feature_timestamp TIMESTAMP NOT NULL,
  PRIMARY KEY (example_id, feature_name, feature_timestamp)
);

CREATE TABLE IF NOT EXISTS agent_trials (
  trial_id TEXT PRIMARY KEY,
  example_id TEXT NOT NULL,
  seed INTEGER,
  model TEXT,
  started_at TIMESTAMP NOT NULL,
  raw_probability DOUBLE,
  status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_steps (
  trial_id TEXT NOT NULL,
  step_index INTEGER NOT NULL,
  action_type TEXT NOT NULL,
  action_json JSON NOT NULL,
  observation_ref TEXT,
  belief_json JSON NOT NULL,
  probability DOUBLE NOT NULL,
  PRIMARY KEY (trial_id, step_index)
);

CREATE TABLE IF NOT EXISTS forecasts (
  example_id TEXT NOT NULL,
  method TEXT NOT NULL,
  raw_probability DOUBLE NOT NULL,
  aggregate_probability DOUBLE,
  calibrated_probability DOUBLE,
  created_at TIMESTAMP NOT NULL,
  PRIMARY KEY (example_id, method, created_at)
);

CREATE TABLE IF NOT EXISTS eval_results (
  run_id TEXT NOT NULL,
  split TEXT NOT NULL,
  method TEXT NOT NULL,
  brier_score DOUBLE NOT NULL,
  log_loss DOUBLE NOT NULL,
  auc DOUBLE,
  calibration_error DOUBLE,
  created_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_runs (
  run_id TEXT PRIMARY KEY,
  run_type TEXT NOT NULL,
  started_at TIMESTAMP NOT NULL,
  completed_at TIMESTAMP,
  status TEXT NOT NULL,
  details_json JSON
);

CREATE TABLE IF NOT EXISTS external_call_cache (
  call_id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  call_type TEXT NOT NULL,
  request_json JSON NOT NULL,
  response_json JSON,
  response_text TEXT,
  captured_at TIMESTAMP NOT NULL,
  information_cutoff TIMESTAMP,
  source_timestamp TIMESTAMP,
  response_sha256 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS live_questions (
  question_id TEXT PRIMARY KEY,
  example_id TEXT NOT NULL,
  question_text TEXT NOT NULL,
  generated_at TIMESTAMP NOT NULL,
  information_cutoff TIMESTAMP NOT NULL,
  resolution_due TIMESTAMP NOT NULL,
  status TEXT NOT NULL,
  source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_responses (
  response_id TEXT PRIMARY KEY,
  question_id TEXT NOT NULL,
  model TEXT NOT NULL,
  prompt TEXT NOT NULL,
  probability DOUBLE NOT NULL,
  reasoning TEXT,
  raw_response TEXT,
  created_at TIMESTAMP NOT NULL,
  information_cutoff TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS live_resolutions (
  question_id TEXT PRIMARY KEY,
  resolved_at TIMESTAMP NOT NULL,
  underlying_close DOUBLE NOT NULL,
  label INTEGER NOT NULL,
  source TEXT NOT NULL
);
"""


class MurphyDb:
    """Small DuckDB wrapper for project schema and inserts."""

    def __init__(self, path: str | Path = "data/murphy.duckdb") -> None:
        import duckdb

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.path))

    def initialize(self) -> None:
        self.conn.execute(SCHEMA_SQL)

    def close(self) -> None:
        self.conn.close()

    def insert_dict(self, table: str, row: dict[str, Any]) -> None:
        columns = list(row)
        placeholders = ", ".join(["?"] * len(columns))
        column_sql = ", ".join(columns)
        self.conn.execute(
            f"INSERT OR REPLACE INTO {table} ({column_sql}) VALUES ({placeholders})",
            [row[column] for column in columns],
        )

    def asof_features(self, example_id: str, forecast_timestamp: str) -> list[tuple[str, float]]:
        query = """
        SELECT feature_name, feature_value
        FROM features
        WHERE example_id = ?
          AND feature_timestamp <= ?
        QUALIFY row_number() OVER (
          PARTITION BY feature_name
          ORDER BY feature_timestamp DESC
        ) = 1
        ORDER BY feature_name
        """
        return self.conn.execute(query, [example_id, forecast_timestamp]).fetchall()
