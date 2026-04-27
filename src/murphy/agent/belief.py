from __future__ import annotations

from murphy.schemas import BeliefState, Confidence


def initial_belief(prior_probability: float = 0.5, reason: str = "Initial prior.") -> BeliefState:
    return BeliefState(
        probability=prior_probability,
        confidence=Confidence.LOW,
        evidence_for=[],
        evidence_against=[],
        open_questions=[
            "Check option-chain context.",
            "Check underlying price trend and realized volatility.",
            "Compare market-implied prior to empirical base rates.",
        ],
        update_reasoning=reason,
    )


def belief_to_prompt_block(belief: BeliefState) -> str:
    return belief.model_dump_json(indent=2)

