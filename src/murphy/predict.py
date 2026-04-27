from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from murphy.priors import clamp_probability


class PromptRepository(Protocol):
    def pending_prediction_prompts(self, limit: int | None = None) -> list[dict]: ...

    def record_llm_response(
        self,
        question_id: str,
        model: str,
        prompt: str,
        probability: float,
        reasoning: str | None,
        raw_response: str | None,
    ) -> str: ...


@dataclass(frozen=True)
class PredictionResult:
    question_id: str
    response_id: str
    probability: float


class OpenAiPredictor:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self.model = model

    def predict(self, prompt: str) -> tuple[float, str | None, str]:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=self.model,
            input=prompt,
            temperature=0.2,
        )
        raw_text = response.output_text
        probability, reasoning = parse_probability_response(raw_text)
        return probability, reasoning, raw_text


class DeterministicPredictor:
    """Test helper and emergency dry-run predictor."""

    def __init__(self, probability: float = 0.5, model: str = "dry-run") -> None:
        self.probability = probability
        self.model = model

    def predict(self, prompt: str) -> tuple[float, str, str]:
        del prompt
        raw = json.dumps({"probability": self.probability, "reasoning": "dry run"})
        return self.probability, "dry run", raw


def predict_pending(
    repository: PromptRepository,
    predictor,
    model: str,
    limit: int | None = None,
) -> list[PredictionResult]:
    records = repository.pending_prediction_prompts(limit=limit)
    results: list[PredictionResult] = []
    for record in records:
        probability, reasoning, raw_response = predictor.predict(record["prompt"])
        response_id = repository.record_llm_response(
            question_id=record["question_id"],
            model=model,
            prompt=record["prompt"],
            probability=probability,
            reasoning=reasoning,
            raw_response=raw_response,
        )
        results.append(
            PredictionResult(
                question_id=record["question_id"],
                response_id=response_id,
                probability=probability,
            )
        )
    return results


def parse_probability_response(raw_text: str) -> tuple[float, str | None]:
    text = raw_text.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _extract_json_object(text)
    probability = clamp_probability(float(payload["probability"]), eps=0.01)
    reasoning = payload.get("reasoning")
    return probability, None if reasoning is None else str(reasoning)


def _extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse JSON probability response: {text[:200]}")
    return json.loads(match.group(0))

