from __future__ import annotations

import json

from murphy.training.datasets import (
    contract_group_key,
    grouped_prediction_splits,
    scalar_resolved_prediction_row,
    scalar_sft_dataset_from_report,
    scalar_sft_split_datasets_from_report,
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


def test_contract_group_key_ignores_forecast_hour() -> None:
    first = _resolved_item()
    second = {
        **_resolved_item(),
        "forecast_timestamp": "2026-04-27T17:00:00+00:00",
        "information_cutoff": "2026-04-27T17:00:00+00:00",
    }

    assert contract_group_key(first) == contract_group_key(second)


def test_grouped_prediction_splits_do_not_split_contract_groups() -> None:
    same_contract_late = {
        **_resolved_item(),
        "forecast_timestamp": "2026-04-29T16:00:00+00:00",
    }
    same_contract_later = {
        **_resolved_item(),
        "forecast_timestamp": "2026-04-29T17:00:00+00:00",
    }
    earlier_contract = {
        **_resolved_item(),
        "symbol": "MSFT",
        "strike": 400.0,
        "forecast_timestamp": "2026-04-27T16:00:00+00:00",
    }
    middle_contract = {
        **_resolved_item(),
        "symbol": "NVDA",
        "strike": 210.0,
        "forecast_timestamp": "2026-04-28T16:00:00+00:00",
    }

    annotated, manifest = grouped_prediction_splits(
        [same_contract_late, same_contract_later, earlier_contract, middle_contract],
        test_fraction=0.34,
    )

    splits_by_group = {}
    for item in annotated:
        splits_by_group.setdefault(item["contract_group_key"], set()).add(item["dataset_split"])

    assert all(len(splits) == 1 for splits in splits_by_group.values())
    assert manifest["contract_group_counts"] == {"test": 1, "train": 2}
    assert manifest["row_counts"] == {"test": 2, "train": 2}


def test_scalar_sft_split_datasets_from_report_include_auditable_metadata() -> None:
    report = {
        "predictions": [
            _resolved_item(),
            {
                **_resolved_item(),
                "forecast_timestamp": "2026-04-27T17:00:00+00:00",
                "information_cutoff": "2026-04-27T17:00:00+00:00",
            },
            {**_resolved_item(), "symbol": "MSFT", "strike": 400.0},
        ]
    }

    split_rows, manifest = scalar_sft_split_datasets_from_report(
        report,
        test_fraction=0.5,
    )
    all_rows = split_rows["train"] + split_rows["test"]

    assert manifest["n_contract_groups"] == 2
    assert {row["metadata"]["dataset_split"] for row in all_rows} == {"train", "test"}
    assert all(row["metadata"]["contract_group_key"] for row in all_rows)
    assert all("input" in row and "output" in row for row in all_rows)


def test_write_jsonl(tmp_path) -> None:
    output = tmp_path / "scalar.jsonl"
    count = write_jsonl([{"input": "x", "output": "{}"}], output)

    assert count == 1
    assert output.read_text(encoding="utf-8") == '{"input": "x", "output": "{}"}\n'
