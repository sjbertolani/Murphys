from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from murphy.schemas import OptionExample


def scalar_sft_row(example: OptionExample, probability: float, label: int | None) -> dict[str, str]:
    output = {"probability": probability, "label": label}
    return {
        "input": (
            "Forecast whether this near-the-money call option expires in the money. "
            f"Symbol={example.symbol}; option={example.option_symbol}; "
            f"forecast_timestamp={example.forecast_timestamp.isoformat()}; "
            f"expiration={example.expiration.isoformat()}; strike={example.strike}; "
            f"spot={example.spot}; dte={example.dte:.3f}; moneyness={example.moneyness:.6f}."
        ),
        "output": json.dumps(output, sort_keys=True),
    }


def scalar_sft_dataset(
    examples: list[OptionExample],
    probabilities: list[float],
) -> list[dict[str, str]]:
    if len(examples) != len(probabilities):
        raise ValueError("examples and probabilities must have the same length")
    return [
        scalar_sft_row(example, probability, example.label)
        for example, probability in zip(examples, probabilities, strict=True)
    ]


def contract_group_key(item: dict) -> str:
    """Return the leakage boundary for repeated hourly forecasts of one contract.

    Multiple hourly rows can ask the same ticker/strike/expiry question with
    different information cutoffs. They must stay in the same train/test split.
    """
    symbol = str(item.get("symbol") or "UNKNOWN").upper()
    expiration = _isoish(item.get("resolution_due") or item.get("expiration"))
    strike = _strike_key(item.get("strike"))
    return f"{symbol}|{expiration}|{strike}"


def grouped_prediction_splits(
    items: list[dict],
    *,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.0,
) -> tuple[list[dict], dict]:
    """Assign train/validation/test splits without crossing contract groups."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test_fraction + validation_fraction must be less than 1")

    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[contract_group_key(item)].append(item)

    split_by_group: dict[str, str] = {}
    warnings: list[str] = []
    sorted_groups = sorted(
        groups,
        key=lambda key: (
            min(_time_key(item.get("forecast_timestamp")) for item in groups[key]),
            key,
        ),
    )
    n_groups = len(sorted_groups)

    if n_groups == 0:
        warnings.append("no_resolved_rows")
    elif n_groups == 1:
        split_by_group[sorted_groups[0]] = "train"
        warnings.append("fewer_than_two_contract_groups")
    else:
        test_count = max(1, round(n_groups * test_fraction))
        test_count = min(test_count, n_groups - 1)
        remaining_after_test = n_groups - test_count
        validation_count = 0
        if validation_fraction > 0:
            if remaining_after_test < 2:
                warnings.append("not_enough_groups_for_validation_split")
            else:
                validation_count = max(1, round(n_groups * validation_fraction))
                validation_count = min(validation_count, remaining_after_test - 1)

        train_end = n_groups - test_count - validation_count
        validation_end = n_groups - test_count
        for key in sorted_groups[:train_end]:
            split_by_group[key] = "train"
        for key in sorted_groups[train_end:validation_end]:
            split_by_group[key] = "validation"
        for key in sorted_groups[validation_end:]:
            split_by_group[key] = "test"

    annotated = []
    for item in items:
        group_key = contract_group_key(item)
        split = split_by_group.get(group_key, "train")
        annotated.append(
            {
                **item,
                "contract_group_key": group_key,
                "dataset_split": split,
            }
        )

    manifest = _split_manifest(annotated, n_groups=n_groups, warnings=warnings)
    return annotated, manifest


def scalar_resolved_prediction_row(
    item: dict,
    *,
    include_metadata: bool = False,
) -> dict[str, Any]:
    """Build one ScalarLM SFT row from a resolved live prediction report item."""
    label = item.get("label")
    if label is None:
        raise ValueError("Resolved ScalarLM rows require a label")
    output = {
        "probability": float(label),
        "label": int(label),
    }
    input_text = (
        "Forecast whether this near-the-money call option expires in the money. "
        f"Question={item.get('question_text')}; "
        f"Symbol={item.get('symbol')}; "
        f"forecast_timestamp={item.get('forecast_timestamp')}; "
        f"information_cutoff={item.get('information_cutoff')}; "
        f"expiration={item.get('resolution_due')}; "
        f"strike={item.get('strike')}; "
        f"spot={item.get('spot')}; "
        f"dte={item.get('dte')}; "
        f"moneyness={item.get('moneyness')}; "
        f"original_model={item.get('model')}; "
        f"original_probability={item.get('probability')}; "
        f"llm_cache_sha256={item.get('external_cache', {}).get('latest_llm_response_sha256')}; "
        f"market_cache_sha256={item.get('external_cache', {}).get('latest_market_response_sha256')}."
    )
    row: dict[str, Any] = {"input": input_text, "output": json.dumps(output, sort_keys=True)}
    if include_metadata:
        row["metadata"] = {
            "question_id": item.get("question_id"),
            "response_id": item.get("response_id"),
            "symbol": item.get("symbol"),
            "forecast_timestamp": _isoish(item.get("forecast_timestamp")),
            "information_cutoff": _isoish(item.get("information_cutoff")),
            "resolution_due": _isoish(item.get("resolution_due")),
            "strike": item.get("strike"),
            "label": item.get("label"),
            "contract_group_key": item.get("contract_group_key") or contract_group_key(item),
            "dataset_split": item.get("dataset_split"),
        }
    return row


def scalar_sft_dataset_from_report(
    report: dict,
    require_leakage_checks: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    for item in report.get("predictions", []):
        if item.get("label") is None:
            continue
        if require_leakage_checks and not all(item.get("leakage_checks", {}).values()):
            continue
        rows.append(scalar_resolved_prediction_row(item))
    return rows


def scalar_sft_split_datasets_from_report(
    report: dict,
    *,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.0,
    require_leakage_checks: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict]:
    """Build ScalarLM JSONL rows partitioned by a leakage-safe contract split."""
    candidates = []
    for item in report.get("predictions", []):
        if item.get("label") is None:
            continue
        if require_leakage_checks and not all(item.get("leakage_checks", {}).values()):
            continue
        candidates.append(item)

    annotated, manifest = grouped_prediction_splits(
        candidates,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
    )
    split_rows: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "test": [],
    }
    if validation_fraction > 0:
        split_rows["validation"] = []
    for item in annotated:
        split = item["dataset_split"]
        split_rows.setdefault(split, []).append(
            scalar_resolved_prediction_row(item, include_metadata=True)
        )
    return split_rows, manifest


def write_jsonl(rows: list[dict[str, Any]], output_path: str | Path) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)


def _split_manifest(annotated: list[dict], *, n_groups: int, warnings: list[str]) -> dict:
    split_counts: dict[str, int] = defaultdict(int)
    split_group_counts: dict[str, set[str]] = defaultdict(set)
    for item in annotated:
        split = item.get("dataset_split") or "train"
        group_key = item.get("contract_group_key") or contract_group_key(item)
        split_counts[str(split)] += 1
        split_group_counts[str(split)].add(str(group_key))
    return {
        "n_rows": len(annotated),
        "n_contract_groups": n_groups,
        "row_counts": dict(sorted(split_counts.items())),
        "contract_group_counts": {
            split: len(groups) for split, groups in sorted(split_group_counts.items())
        },
        "warnings": warnings,
        "grouping_rule": "symbol|resolution_due|strike",
    }


def _time_key(value) -> str:
    return _isoish(value)


def _isoish(value) -> str:
    if value is None:
        return "UNKNOWN"
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _strike_key(value) -> str:
    if value is None:
        return "UNKNOWN"
    try:
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)
