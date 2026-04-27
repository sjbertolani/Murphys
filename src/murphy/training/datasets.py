from __future__ import annotations

import json

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

