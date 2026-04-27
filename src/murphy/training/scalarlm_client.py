from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScalarLmConfig:
    api_url: str
    max_steps: int = 200
    learning_rate: float = 3e-3


class ScalarLmTrainer:
    """Thin wrapper around ScalarLM's Python client.

    Import is deferred so the rest of the project works without ScalarLM installed.
    """

    def __init__(self, config: ScalarLmConfig) -> None:
        self.config = config

    def submit_sft(self, dataset: list[dict[str, str]]) -> dict[str, Any]:
        import scalarlm

        scalarlm.api_url = self.config.api_url
        llm = scalarlm.SupermassiveIntelligence()
        return llm.train(
            dataset,
            train_args={
                "max_steps": self.config.max_steps,
                "learning_rate": self.config.learning_rate,
            },
        )

