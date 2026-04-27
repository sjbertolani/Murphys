from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from murphy.agent.aggregation import logit_mean
from murphy.agent.belief import initial_belief
from murphy.agent.tools import OptionTools, ToolObservation
from murphy.schemas import BeliefState, Confidence, OptionExample


@dataclass(frozen=True)
class AgentStep:
    index: int
    action: str
    observation: ToolObservation
    belief: BeliefState


@dataclass(frozen=True)
class AgentTrial:
    trial_id: str
    example_id: str
    started_at: datetime
    steps: list[AgentStep] = field(default_factory=list)
    probability: float = 0.5
    status: str = "completed"


class HeuristicOptionAgent:
    """A deterministic BLF-shaped baseline before wiring in an LLM.

    This gives us trace storage and tool plumbing immediately. The LLM runner can
    later implement the same input/output contract.
    """

    def __init__(self, tools: OptionTools) -> None:
        self.tools = tools

    def run(self, example: OptionExample, prior_probability: float = 0.5) -> AgentTrial:
        steps: list[AgentStep] = []
        belief = initial_belief(prior_probability)

        observation = self.tools.fetch_option_snapshot(example)
        belief = _update_from_snapshot(belief, observation.payload)
        steps.append(AgentStep(1, observation.name, observation, belief))

        observation = self.tools.compute_underlying_features(example)
        belief = _update_from_underlying_features(belief, observation.payload)
        steps.append(AgentStep(2, observation.name, observation, belief))

        observation = self.tools.compute_market_prior(example)
        belief = _update_from_market_prior(belief, observation.payload)
        steps.append(AgentStep(3, observation.name, observation, belief))

        return AgentTrial(
            trial_id=str(uuid4()),
            example_id=example.example_id,
            started_at=datetime.now(timezone.utc),
            steps=steps,
            probability=belief.probability,
            status="completed",
        )


def aggregate_trials(trials: list[AgentTrial]) -> float:
    return logit_mean([trial.probability for trial in trials])


def _update_from_snapshot(belief: BeliefState, payload: dict[str, Any]) -> BeliefState:
    evidence_for = list(belief.evidence_for)
    evidence_against = list(belief.evidence_against)
    probability = belief.probability

    moneyness = payload.get("moneyness")
    if moneyness is not None:
        if moneyness > 0:
            evidence_for.append(f"Underlying is above strike at forecast time: moneyness={moneyness:.4f}.")
            probability += min(moneyness * 2.0, 0.08)
        else:
            evidence_against.append(
                f"Underlying is below strike at forecast time: moneyness={moneyness:.4f}."
            )
            probability += max(moneyness * 2.0, -0.08)

    spread = payload.get("bid_ask_spread_fraction")
    if spread is not None and spread > 0.25:
        evidence_against.append(f"Wide option spread suggests noisy market signal: spread={spread:.3f}.")

    return BeliefState(
        probability=_clip_agent_probability(probability),
        confidence=Confidence.LOW,
        evidence_for=evidence_for,
        evidence_against=evidence_against,
        open_questions=["Check realized volatility and market-implied probability."],
        update_reasoning="Updated from option snapshot, moneyness, and liquidity context.",
    )


def _update_from_underlying_features(belief: BeliefState, payload: dict[str, Any]) -> BeliefState:
    evidence_for = list(belief.evidence_for)
    evidence_against = list(belief.evidence_against)
    probability = belief.probability

    trend = payload.get("trailing_return")
    if trend is not None:
        if trend > 0.03:
            evidence_for.append(f"Positive recent underlying trend: trailing_return={trend:.3f}.")
            probability += 0.03
        elif trend < -0.03:
            evidence_against.append(f"Negative recent underlying trend: trailing_return={trend:.3f}.")
            probability -= 0.03

    rv = payload.get("realized_volatility")
    if rv is not None and rv > 0.6:
        evidence_for.append(f"High realized volatility keeps upside probability alive: rv={rv:.3f}.")
        probability += 0.01

    return BeliefState(
        probability=_clip_agent_probability(probability),
        confidence=Confidence.MEDIUM,
        evidence_for=evidence_for,
        evidence_against=evidence_against,
        open_questions=["Compare this to the option-implied prior."],
        update_reasoning="Updated from recent underlying history.",
    )


def _update_from_market_prior(belief: BeliefState, payload: dict[str, Any]) -> BeliefState:
    market_probability = payload.get("probability")
    if market_probability is None:
        return belief

    probability = 0.65 * market_probability + 0.35 * belief.probability
    evidence = f"Market prior from {payload.get('source')}: p={market_probability:.3f}."
    return BeliefState(
        probability=_clip_agent_probability(probability),
        confidence=Confidence.MEDIUM,
        evidence_for=[*belief.evidence_for, evidence],
        evidence_against=belief.evidence_against,
        open_questions=[],
        update_reasoning="Blended market-implied prior with accumulated belief.",
    )


def _clip_agent_probability(probability: float) -> float:
    return min(max(probability, 0.01), 0.99)

