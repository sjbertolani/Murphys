from __future__ import annotations

from pathlib import Path

from murphy.cloud_config import CloudSqlConfig
from murphy.cloud_sql import close_cloud_sql_engine, create_cloud_sql_engine
from murphy.db import MurphyDb


EXPORT_TABLES = [
    "underlying_bars",
    "option_chain_snapshots",
    "option_examples",
    "live_questions",
    "llm_responses",
    "live_resolutions",
    "daily_runs",
    "external_call_cache",
]


def export_cloud_sql_to_duckdb(
    cloud_sql: CloudSqlConfig,
    duckdb_path: str | Path = "data/murphy_offline.duckdb",
    tables: list[str] | None = None,
) -> dict[str, int]:
    """Copy selected Cloud SQL tables into a local DuckDB file for offline analytics."""
    import pandas as pd

    selected_tables = tables or EXPORT_TABLES
    engine = create_cloud_sql_engine(cloud_sql)
    duck = MurphyDb(duckdb_path)
    duck.initialize()
    counts: dict[str, int] = {}

    try:
        with engine.begin() as source_conn:
            for table in selected_tables:
                dataframe = pd.read_sql_query(f"SELECT * FROM {table}", source_conn)
                duck.conn.execute(f"DELETE FROM {table}")
                if dataframe.empty:
                    counts[table] = 0
                    continue
                duck.conn.register("export_df", dataframe)
                duck.conn.execute(f"INSERT INTO {table} SELECT * FROM export_df")
                duck.conn.unregister("export_df")
                counts[table] = len(dataframe)
    finally:
        duck.close()
        close_cloud_sql_engine(engine)

    return counts
