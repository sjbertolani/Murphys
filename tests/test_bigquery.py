from __future__ import annotations

import pandas as pd
import pytest

from murphy.bigquery import (
    MIRROR_TABLES,
    normalize_dataframe_for_bigquery,
    table_id,
)
from murphy.cloud_config import BigQueryConfig


def test_bigquery_table_id() -> None:
    config = BigQueryConfig(project_id="project", dataset="murphy", location="US")
    assert table_id(config, "live_questions") == "project.murphy.live_questions"


def test_normalize_external_call_cache_json_columns() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "call_id": "call-1",
                "provider": "openai",
                "call_type": "llm_prediction",
                "request_json": {"question_id": "q1", "model": "dry-run"},
                "response_json": {"probability": 0.5},
                "response_text": None,
                "captured_at": "2026-04-27T00:00:00Z",
                "information_cutoff": "2026-04-26T20:00:00Z",
                "source_timestamp": "2026-04-27T00:00:00Z",
                "response_sha256": "abc",
                "extra_column": "ignored",
            }
        ]
    )

    normalized = normalize_dataframe_for_bigquery("external_call_cache", dataframe)

    assert list(normalized.columns) == [
        "call_id",
        "provider",
        "call_type",
        "request_json",
        "response_json",
        "response_text",
        "captured_at",
        "information_cutoff",
        "source_timestamp",
        "response_sha256",
    ]
    assert normalized.loc[0, "request_json"] == '{"model": "dry-run", "question_id": "q1"}'
    assert normalized.loc[0, "response_json"] == '{"probability": 0.5}'


def test_normalize_rejects_unknown_table() -> None:
    with pytest.raises(ValueError, match="Unsupported BigQuery table"):
        normalize_dataframe_for_bigquery("not_a_table", pd.DataFrame())


def test_mirror_table_list_has_only_configured_tables() -> None:
    for table in MIRROR_TABLES:
        normalized = normalize_dataframe_for_bigquery(table, pd.DataFrame())
        assert normalized.empty
