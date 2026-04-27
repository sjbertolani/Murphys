from __future__ import annotations

from murphy.cloud_config import load_cloud_config_from_env


def test_load_cloud_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("CLOUD_SQL_CONNECTION_NAME", "project:region:instance")
    monkeypatch.setenv("DB_NAME", "murphy")
    monkeypatch.setenv("DB_USER", "murphy_app")
    monkeypatch.setenv("DB_PASS", "secret")
    monkeypatch.setenv("BQ_PROJECT_ID", "project")
    monkeypatch.setenv("BQ_DATASET", "murphy")

    config = load_cloud_config_from_env()

    assert config.cloud_sql is not None
    assert config.cloud_sql.database == "murphy"
    assert config.bigquery is not None
    assert config.bigquery.dataset == "murphy"

