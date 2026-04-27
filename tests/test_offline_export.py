from __future__ import annotations

import json

import pandas as pd

from murphy.offline_export import normalize_json_columns


def test_normalize_json_columns_serializes_python_objects() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "request_json": {"tickers": ["AAPL"], "min_dte": 0},
                "response_json": [{"ok": True}],
            }
        ]
    )

    normalized = normalize_json_columns(dataframe, ["request_json", "response_json"])

    assert json.loads(normalized.loc[0, "request_json"]) == {
        "min_dte": 0,
        "tickers": ["AAPL"],
    }
    assert json.loads(normalized.loc[0, "response_json"]) == [{"ok": True}]


def test_normalize_json_columns_leaves_strings_and_nulls() -> None:
    dataframe = pd.DataFrame(
        [
            {"details_json": '{"ok": true}'},
            {"details_json": None},
        ]
    )

    normalized = normalize_json_columns(dataframe, ["details_json"])

    assert normalized.loc[0, "details_json"] == '{"ok": true}'
    assert normalized.loc[1, "details_json"] is None
