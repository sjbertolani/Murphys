from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class CloudSqlConfig:
    instance_connection_name: str
    database: str
    user: str
    password: str
    use_private_ip: bool = False


@dataclass(frozen=True)
class BigQueryConfig:
    project_id: str
    dataset: str
    location: str = "US"


@dataclass(frozen=True)
class CloudConfig:
    cloud_sql: CloudSqlConfig | None
    bigquery: BigQueryConfig | None


def load_cloud_config_from_env() -> CloudConfig:
    cloud_sql = None
    if os.environ.get("CLOUD_SQL_CONNECTION_NAME"):
        cloud_sql = CloudSqlConfig(
            instance_connection_name=os.environ["CLOUD_SQL_CONNECTION_NAME"],
            database=os.environ["DB_NAME"],
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASS"],
            use_private_ip=os.environ.get("PRIVATE_IP", "").lower() in {"1", "true", "yes"},
        )

    bigquery = None
    if os.environ.get("BQ_PROJECT_ID") and os.environ.get("BQ_DATASET"):
        bigquery = BigQueryConfig(
            project_id=os.environ["BQ_PROJECT_ID"],
            dataset=os.environ["BQ_DATASET"],
            location=os.environ.get("BQ_LOCATION", "US"),
        )

    return CloudConfig(cloud_sql=cloud_sql, bigquery=bigquery)

