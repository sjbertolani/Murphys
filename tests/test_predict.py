from __future__ import annotations

from murphy.predict import DeterministicPredictor, parse_probability_response, predict_pending


class FakePromptRepository:
    def __init__(self) -> None:
        self.recorded = []

    def pending_prediction_prompts(self, limit=None):
        del limit
        return [
            {
                "question_id": "q1",
                "prompt": "Return probability.",
            }
        ]

    def record_llm_response(
        self,
        question_id,
        model,
        prompt,
        probability,
        reasoning,
        raw_response,
    ):
        self.recorded.append(
            {
                "question_id": question_id,
                "model": model,
                "prompt": prompt,
                "probability": probability,
                "reasoning": reasoning,
                "raw_response": raw_response,
            }
        )
        return "r1"


def test_parse_probability_response() -> None:
    probability, reasoning = parse_probability_response(
        '{"probability": 0.73, "reasoning": "spot is above strike"}'
    )

    assert probability == 0.73
    assert reasoning == "spot is above strike"


def test_predict_pending_records_response() -> None:
    repository = FakePromptRepository()
    results = predict_pending(
        repository,
        DeterministicPredictor(probability=0.42),
        model="dry-run",
    )

    assert len(results) == 1
    assert results[0].probability == 0.42
    assert repository.recorded[0]["probability"] == 0.42

