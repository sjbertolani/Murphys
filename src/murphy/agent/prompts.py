from __future__ import annotations

from murphy.agent.belief import belief_to_prompt_block
from murphy.schemas import BeliefState, OptionExample


SYSTEM_PROMPT = """You are an expert probabilistic forecaster.
Your task is to estimate whether a near-the-money call option will expire in the money.
Use only evidence available at the forecast timestamp.
Maintain a structured belief state with probability, confidence, evidence, open questions, and update reasoning.
Return probabilities between 0.01 and 0.99.
"""


def question_prompt(example: OptionExample, belief: BeliefState) -> str:
    return f"""# Option Forecast
Symbol: {example.symbol}
Option: {example.option_symbol}
Forecast timestamp: {example.forecast_timestamp.isoformat()}
Expiration: {example.expiration.isoformat()}
Strike: {example.strike}
Spot: {example.spot}
DTE: {example.dte:.3f}
Moneyness S/K - 1: {example.moneyness:.6f}

Binary event:
Will the underlying close above the call strike at expiration?

Current belief:
{belief_to_prompt_block(belief)}
"""

