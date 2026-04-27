from __future__ import annotations

import json
from pathlib import Path

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


def scalar_resolved_prediction_row(item: dict) -> dict[str, str]:
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
    return {"input": input_text, "output": json.dumps(output, sort_keys=True)}


def scalar_sft_dataset_from_report(
    report: dict,
    require_leakage_checks: bool = True,
) -> list[dict[str, str]]:
    rows = []
    for item in report.get("predictions", []):
        if item.get("label") is None:
            continue
        if require_leakage_checks and not all(item.get("leakage_checks", {}).values()):
            continue
        rows.append(scalar_resolved_prediction_row(item))
    return rows


def write_jsonl(rows: list[dict[str, str]], output_path: str | Path) -> int:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(rows)
