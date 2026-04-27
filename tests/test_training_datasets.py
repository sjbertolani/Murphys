from __future__ import annotations

import json

from murphy.training.datasets import (
    scalar_resolved_prediction_row,
    scalar_sft_dataset_from_report,
    write_jsonl,
)


def _resolved_item(label=1, leakage_ok=True):
    return {
        "question_text": "Will the price of $AAPL be greater than $200.00 on 2026-05-04?",
        "symbol": "AAPL",
        "forecast_timestamp": "2026-04-27T16:00:00+00:00",
        "information_cutoff": "2026-04-27T16:00:00+00:00",
        "resolution_due": "2026-05-04T21:00:00+00:00",
        "strike": 200.0,
        "spot": 201.0,
        "dte": 7.0,
        "moneyness": 0.005,
        "model": "dry-run",
        "probability": 0.5,
        "label": label,
        "external_cache": {
            "latest_llm_response_sha256": "abc",
            "latest_market_response_sha256": "def",
        },
        "leakage_checks": {
            "question_cutoff_not_after_prediction": leakage_ok,
            "response_cutoff_not_after_prediction": True,
        },
    }


def test_scalar_resolved_prediction_row_targets_realized_label() -> None:
    row = scalar_resolved_prediction_row(_resolved_item(label=0))

    assert "Will the price of $AAPL" in row["input"]
    assert "llm_cache_sha256=abc" in row["input"]
    assert json.loads(row["output"]) == {"label": 0, "probability": 0.0}


def test_scalar_sft_dataset_from_report_filters_unresolved_and_leakage_failures() -> None:
    report = {
        "predictions": [
            _resolved_item(label=1),
            {**_resolved_item(label=None), "label": None},
            _resolved_item(label=0, leakage_ok=False),
        ]
    }

    rows = scalar_sft_dataset_from_report(report)
    rows_with_failures = scalar_sft_dataset_from_report(report, require_leakage_checks=False)

    assert len(rows) == 1
    assert json.loads(rows[0]["output"])["label"] == 1
    assert len(rows_with_failures) == 2


def test_write_jsonl(tmp_path) -> None:
    output = tmp_path / "scalar.jsonl"
    count = write_jsonl([{"input": "x", "output": "{}"}], output)

    assert count == 1
    assert output.read_text(encoding="utf-8") == '{"input": "x", "output": "{}"}\n'
