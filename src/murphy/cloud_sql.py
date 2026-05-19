from __future__ import annotations

from murphy.cloud_config import CloudSqlConfig


POSTGRES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS underlying_bars (
  symbol TEXT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL,
  open DOUBLE PRECISION NOT NULL,
  high DOUBLE PRECISION NOT NULL,
  low DOUBLE PRECISION NOT NULL,
  close DOUBLE PRECISION NOT NULL,
  volume DOUBLE PRECISION,
  adjusted_close DOUBLE PRECISION,
  PRIMARY KEY (symbol, timestamp)
);

CREATE TABLE IF NOT EXISTS option_chain_snapshots (
  symbol TEXT NOT NULL,
  option_symbol TEXT NOT NULL,
  quote_timestamp TIMESTAMPTZ NOT NULL,
  expiration TIMESTAMPTZ NOT NULL,
  strike DOUBLE PRECISION NOT NULL,
  option_right TEXT NOT NULL,
  bid DOUBLE PRECISION,
  ask DOUBLE PRECISION,
  mid DOUBLE PRECISION,
  implied_volatility DOUBLE PRECISION,
  delta DOUBLE PRECISION,
  gamma DOUBLE PRECISION,
  theta DOUBLE PRECISION,
  vega DOUBLE PRECISION,
  volume DOUBLE PRECISION,
  open_interest DOUBLE PRECISION,
  spot DOUBLE PRECISION,
  PRIMARY KEY (option_symbol, quote_timestamp)
);

CREATE TABLE IF NOT EXISTS option_examples (
  example_id TEXT PRIMARY KEY,
  symbol TEXT NOT NULL,
  option_symbol TEXT NOT NULL,
  forecast_timestamp TIMESTAMPTZ NOT NULL,
  expiration TIMESTAMPTZ NOT NULL,
  strike DOUBLE PRECISION NOT NULL,
  spot DOUBLE PRECISION NOT NULL,
  dte DOUBLE PRECISION NOT NULL,
  moneyness DOUBLE PRECISION NOT NULL,
  label INTEGER,
  resolution_timestamp TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS agent_trials (
  trial_id TEXT PRIMARY KEY,
  example_id TEXT NOT NULL REFERENCES option_examples(example_id),
  seed INTEGER,
  model TEXT,
  started_at TIMESTAMPTZ NOT NULL,
  raw_probability DOUBLE PRECISION,
  status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_steps (
  trial_id TEXT NOT NULL REFERENCES agent_trials(trial_id),
  step_index INTEGER NOT NULL,
  action_type TEXT NOT NULL,
  action_json JSONB NOT NULL,
  observation_ref TEXT,
  belief_json JSONB NOT NULL,
  probability DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (trial_id, step_index)
);

CREATE TABLE IF NOT EXISTS forecasts (
  example_id TEXT NOT NULL REFERENCES option_examples(example_id),
  method TEXT NOT NULL,
  raw_probability DOUBLE PRECISION NOT NULL,
  aggregate_probability DOUBLE PRECISION,
  calibrated_probability DOUBLE PRECISION,
  created_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (example_id, method, created_at)
);

CREATE TABLE IF NOT EXISTS live_questions (
  question_id TEXT PRIMARY KEY,
  example_id TEXT NOT NULL REFERENCES option_examples(example_id),
  question_text TEXT NOT NULL,
  generated_at TIMESTAMPTZ NOT NULL,
  information_cutoff TIMESTAMPTZ NOT NULL,
  resolution_due TIMESTAMPTZ NOT NULL,
  status TEXT NOT NULL,
  source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_responses (
  response_id TEXT PRIMARY KEY,
  question_id TEXT NOT NULL REFERENCES live_questions(question_id),
  model TEXT NOT NULL,
  prompt TEXT NOT NULL,
  probability DOUBLE PRECISION NOT NULL,
  reasoning TEXT,
  raw_response TEXT,
  created_at TIMESTAMPTZ NOT NULL,
  information_cutoff TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS live_resolutions (
  question_id TEXT PRIMARY KEY REFERENCES live_questions(question_id),
  resolved_at TIMESTAMPTZ NOT NULL,
  underlying_close DOUBLE PRECISION NOT NULL,
  label INTEGER NOT NULL,
  source TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily_runs (
  run_id TEXT PRIMARY KEY,
  run_type TEXT NOT NULL,
  started_at TIMESTAMPTZ NOT NULL,
  completed_at TIMESTAMPTZ,
  status TEXT NOT NULL,
  details_json JSONB
);

CREATE TABLE IF NOT EXISTS external_call_cache (
  call_id TEXT PRIMARY KEY,
  provider TEXT NOT NULL,
  call_type TEXT NOT NULL,
  request_json JSONB NOT NULL,
  response_json JSONB,
  response_text TEXT,
  captured_at TIMESTAMPTZ NOT NULL,
  information_cutoff TIMESTAMPTZ,
  source_timestamp TIMESTAMPTZ,
  response_sha256 TEXT NOT NULL
);
"""


def create_cloud_sql_engine(config: CloudSqlConfig):
    """Create a SQLAlchemy engine for Cloud SQL Postgres.

    Uses Google's Cloud SQL Python Connector. Import is deferred so local DuckDB
    workflows do not need cloud dependencies installed.
    """
    import sqlalchemy
    from google.cloud.sql.connector import Connector, IPTypes

    connector = Connector(refresh_strategy="LAZY")
    ip_type = IPTypes.PRIVATE if config.use_private_ip else IPTypes.PUBLIC

    def getconn():
        return connector.connect(
            config.instance_connection_name,
            "pg8000",
            user=config.user,
            password=config.password,
            db=config.database,
            ip_type=ip_type,
        )

    engine = sqlalchemy.create_engine("postgresql+pg8000://", creator=getconn)
    setattr(engine, "_murphy_cloud_sql_connector", connector)
    return engine


def initialize_cloud_sql(config: CloudSqlConfig) -> None:
    engine = create_cloud_sql_engine(config)
    try:
        with engine.begin() as conn:
            for statement in _split_sql_statements(POSTGRES_SCHEMA_SQL):
                conn.exec_driver_sql(statement)
    finally:
        close_cloud_sql_engine(engine)


def close_cloud_sql_engine(engine) -> None:
    engine.dispose()
    connector = getattr(engine, "_murphy_cloud_sql_connector", None)
    if connector is not None:
        connector.close()


def _split_sql_statements(sql: str) -> list[str]:
    return [statement.strip() for statement in sql.split(";") if statement.strip()]
