from __future__ import annotations

from dataclasses import dataclass

from murphy.cloud_config import BigQueryConfig


@dataclass(frozen=True)
class BigQueryTable:
    name: str
    schema: list[dict[str, str]]


BIGQUERY_TABLES = [
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
        name="llm_responses",
        schema=[
            {"name": "response_id", "type": "STRING"},
            {"name": "question_id", "type": "STRING"},
            {"name": "model", "type": "STRING"},
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
        schema = [
            bigquery.SchemaField(field["name"], field["type"], mode="NULLABLE")
            for field in table.schema
        ]
        client.create_table(bigquery.Table(table_id(config, table.name), schema=schema), exists_ok=True)


def append_dataframe(config: BigQueryConfig, table: str, dataframe) -> None:
    from google.cloud import bigquery

    client = bigquery.Client(project=config.project_id, location=config.location)
    job = client.load_table_from_dataframe(dataframe, table_id(config, table))
    job.result()

