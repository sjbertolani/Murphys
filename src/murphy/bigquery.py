from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from murphy.cloud_config import BigQueryConfig, CloudSqlConfig
from murphy.cloud_sql import close_cloud_sql_engine, create_cloud_sql_engine


MIRROR_TABLES = [
    "underlying_bars",
    "option_chain_snapshots",
    "option_examples",
    "live_questions",
    "llm_responses",
    "live_resolutions",
    "forecasts",
    "daily_runs",
    "external_call_cache",
]


@dataclass(frozen=True)
class BigQueryTable:
    name: str
    schema: list[dict[str, str]]


BIGQUERY_TABLES = [
    BigQueryTable(
        name="underlying_bars",
        schema=[
            {"name": "symbol", "type": "STRING"},
            {"name": "timestamp", "type": "TIMESTAMP"},
            {"name": "open", "type": "FLOAT"},
            {"name": "high", "type": "FLOAT"},
            {"name": "low", "type": "FLOAT"},
            {"name": "close", "type": "FLOAT"},
            {"name": "volume", "type": "FLOAT"},
            {"name": "adjusted_close", "type": "FLOAT"},
        ],
    ),
    BigQueryTable(
        name="option_chain_snapshots",
        schema=[
            {"name": "symbol", "type": "STRING"},
            {"name": "option_symbol", "type": "STRING"},
            {"name": "quote_timestamp", "type": "TIMESTAMP"},
            {"name": "expiration", "type": "TIMESTAMP"},
            {"name": "strike", "type": "FLOAT"},
            {"name": "option_right", "type": "STRING"},
            {"name": "bid", "type": "FLOAT"},
            {"name": "ask", "type": "FLOAT"},
            {"name": "mid", "type": "FLOAT"},
            {"name": "implied_volatility", "type": "FLOAT"},
            {"name": "delta", "type": "FLOAT"},
            {"name": "gamma", "type": "FLOAT"},
            {"name": "theta", "type": "FLOAT"},
            {"name": "vega", "type": "FLOAT"},
            {"name": "volume", "type": "FLOAT"},
            {"name": "open_interest", "type": "FLOAT"},
            {"name": "spot", "type": "FLOAT"},
        ],
    ),
    BigQueryTable(
        name="option_examples",
        schema=[
            {"name": "example_id", "type": "STRING"},
            {"name": "symbol", "type": "STRING"},
            {"name": "option_symbol", "type": "STRING"},
            {"name": "forecast_timestamp", "type": "TIMESTAMP"},
            {"name": "expiration", "type": "TIMESTAMP"},
            {"name": "strike", "type": "FLOAT"},
            {"name": "spot", "type": "FLOAT"},
            {"name": "dte", "type": "FLOAT"},
            {"name": "moneyness", "type": "FLOAT"},
            {"name": "label", "type": "INTEGER"},
            {"name": "resolution_timestamp", "type": "TIMESTAMP"},
        ],
    ),
    BigQueryTable(
        name="live_questions",
        schema=[
            {"name": "question_id", "type": "STRING"},
            {"name": "example_id", "type": "STRING"},
            {"name": "question_text", "type": "STRING"},
            {"name": "generated_at", "type": "TIMESTAMP"},
            {"name": "information_cutoff", "type": "TIMESTAMP"},
            {"name": "resolution_due", "type": "TIMESTAMP"},
            {"name": "status", "type": "STRING"},
            {"name": "source", "type": "STRING"},
        ],
    ),
    BigQueryTable(
        name="llm_responses",
        schema=[
            {"name": "response_id", "type": "STRING"},
            {"name": "question_id", "type": "STRING"},
            {"name": "model", "type": "STRING"},
            {"name": "prompt", "type": "STRING"},
            {"name": "probability", "type": "FLOAT"},
            {"name": "reasoning", "type": "STRING"},
            {"name": "raw_response", "type": "STRING"},
            {"name": "created_at", "type": "TIMESTAMP"},
            {"name": "information_cutoff", "type": "TIMESTAMP"},
        ],
    ),
    BigQueryTable(
        name="live_resolutions",
        schema=[
            {"name": "question_id", "type": "STRING"},
            {"name": "resolved_at", "type": "TIMESTAMP"},
            {"name": "underlying_close", "type": "FLOAT"},
            {"name": "label", "type": "INTEGER"},
            {"name": "source", "type": "STRING"},
        ],
    ),
    BigQueryTable(
        name="forecasts",
        schema=[
            {"name": "example_id", "type": "STRING"},
            {"name": "method", "type": "STRING"},
            {"name": "raw_probability", "type": "FLOAT"},
            {"name": "aggregate_probability", "type": "FLOAT"},
            {"name": "calibrated_probability", "type": "FLOAT"},
            {"name": "created_at", "type": "TIMESTAMP"},
        ],
    ),
    BigQueryTable(
        name="daily_runs",
        schema=[
            {"name": "run_id", "type": "STRING"},
            {"name": "run_type", "type": "STRING"},
            {"name": "started_at", "type": "TIMESTAMP"},
            {"name": "completed_at", "type": "TIMESTAMP"},
            {"name": "status", "type": "STRING"},
            {"name": "details_json", "type": "STRING"},
        ],
    ),
    BigQueryTable(
        name="external_call_cache",
        schema=[
            {"name": "call_id", "type": "STRING"},
            {"name": "provider", "type": "STRING"},
            {"name": "call_type", "type": "STRING"},
            {"name": "request_json", "type": "STRING"},
            {"name": "response_json", "type": "STRING"},
            {"name": "response_text", "type": "STRING"},
            {"name": "captured_at", "type": "TIMESTAMP"},
            {"name": "information_cutoff", "type": "TIMESTAMP"},
            {"name": "source_timestamp", "type": "TIMESTAMP"},
            {"name": "response_sha256", "type": "STRING"},
        ],
    ),
]


def table_id(config: BigQueryConfig, table: str) -> str:
    return f"{config.project_id}.{config.dataset}.{table}"


def initialize_bigquery_dataset(config: BigQueryConfig) -> None:
    from google.cloud import bigquery

    client = bigquery.Client(project=config.project_id, location=config.location)
    dataset_id = f"{config.project_id}.{config.dataset}"
    dataset = bigquery.Dataset(dataset_id)
    dataset.location = config.location
    client.create_dataset(dataset, exists_ok=True)

    for table in BIGQUERY_TABLES:
        schema = _bigquery_schema(bigquery, table.name)
        client.create_table(bigquery.Table(table_id(config, table.name), schema=schema), exists_ok=True)


def append_dataframe(config: BigQueryConfig, table: str, dataframe) -> None:
    load_dataframe(config, table, dataframe, write_disposition="WRITE_APPEND")


def load_dataframe(
    config: BigQueryConfig,
    table: str,
    dataframe,
    write_disposition: str = "WRITE_APPEND",
) -> None:
    from google.cloud import bigquery

    client = bigquery.Client(project=config.project_id, location=config.location)
    job_config = bigquery.LoadJobConfig(
        schema=_bigquery_schema(bigquery, table),
        write_disposition=write_disposition,
    )
    job = client.load_table_from_dataframe(dataframe, table_id(config, table), job_config=job_config)
    job.result()


def mirror_cloud_sql_to_bigquery(
    cloud_sql: CloudSqlConfig,
    bigquery_config: BigQueryConfig,
    tables: list[str] | None = None,
    write_disposition: str = "WRITE_TRUNCATE",
) -> dict[str, int]:
    """Mirror allowed Cloud SQL tables into BigQuery for analytics.

    Cloud SQL remains the transactional source of truth. The default BigQuery
    load mode replaces each analytic table, making repeated scheduled exports
    idempotent even though BigQuery does not enforce our Cloud SQL primary keys.
    """
    import pandas as pd

    selected_tables = _validate_table_names(tables or MIRROR_TABLES)
    initialize_bigquery_dataset(bigquery_config)
    engine = create_cloud_sql_engine(cloud_sql)
    counts: dict[str, int] = {}
    try:
        with engine.begin() as source_conn:
            for table in selected_tables:
                dataframe = pd.read_sql_query(f"SELECT * FROM {table}", source_conn)
                dataframe = normalize_dataframe_for_bigquery(table, dataframe)
                load_dataframe(
                    bigquery_config,
                    table,
                    dataframe,
                    write_disposition=write_disposition,
                )
                counts[table] = len(dataframe)
    finally:
        close_cloud_sql_engine(engine)
    return counts


def bigquery_table_freshness(
    config: BigQueryConfig,
    tables: list[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Return BigQuery table metadata useful for operational status reports."""
    from google.cloud import bigquery

    client = bigquery.Client(project=config.project_id, location=config.location)
    selected_tables = _validate_table_names(tables or MIRROR_TABLES)
    freshness: dict[str, dict[str, object]] = {}
    for table in selected_tables:
        metadata = client.get_table(table_id(config, table))
        freshness[table] = {
            "num_rows": metadata.num_rows,
            "modified": metadata.modified,
        }
    return freshness


def normalize_dataframe_for_bigquery(table: str, dataframe):
    """Select schema columns and encode JSON-like values as stable strings."""
    normalized = dataframe.copy()
    for column in _json_string_columns(table):
        if column in normalized.columns:
            normalized[column] = normalized[column].map(_json_value_to_string)
    columns = [field["name"] for field in _table_definition(table).schema]
    for column in columns:
        if column not in normalized.columns:
            normalized[column] = None
    return normalized[columns]


def _bigquery_schema(bigquery_module: Any, table: str) -> list[Any]:
    return [
        bigquery_module.SchemaField(field["name"], field["type"], mode="NULLABLE")
        for field in _table_definition(table).schema
    ]


def _table_definition(table: str) -> BigQueryTable:
    for definition in BIGQUERY_TABLES:
        if definition.name == table:
            return definition
    raise ValueError(f"Unsupported BigQuery table: {table}")


def _validate_table_names(tables: list[str]) -> list[str]:
    allowed = {table.name for table in BIGQUERY_TABLES}
    unknown = sorted(set(tables) - allowed)
    if unknown:
        raise ValueError(f"Unsupported BigQuery mirror tables: {', '.join(unknown)}")
    return tables


def _json_string_columns(table: str) -> set[str]:
    if table == "daily_runs":
        return {"details_json"}
    if table == "external_call_cache":
        return {"request_json", "response_json"}
    return set()


def _json_value_to_string(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, sort_keys=True)
