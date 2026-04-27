from __future__ import annotations

from datetime import datetime, timedelta, timezone

from murphy.agent.runner import HeuristicOptionAgent, aggregate_trials
from murphy.agent.tools import InMemoryToolStore, OptionTools
from murphy.options_data import build_option_example, is_near_money
from murphy.schemas import OptionSnapshot, UnderlyingBar


def test_build_option_example_and_agent_trace() -> None:
    now = datetime(2026, 1, 2, 21, 0, tzinfo=timezone.utc)
    expiration = now + timedelta(days=30)
    snapshot = OptionSnapshot(
        symbol="XYZ",
        option_symbol="XYZ-20260201-C-100",
        quote_timestamp=now,
        expiration=expiration,
        strike=100.0,
        bid=4.8,
        ask=5.2,
        mid=5.0,
        implied_volatility=0.30,
        delta=0.52,
        volume=100,
        open_interest=1000,
        spot=101.0,
    )
    example = build_option_example(snapshot, close_at_expiration=103.0, resolution_timestamp=expiration)

    assert example.label == 1
    assert is_near_money(example.spot, example.strike)

    bars = [
        UnderlyingBar(
            symbol="XYZ",
            timestamp=now - timedelta(days=days),
            open=100.0 - days * 0.1,
            high=101.0 - days * 0.1,
            low=99.0 - days * 0.1,
            close=100.0 - days * 0.1,
            volume=1000,
        )
        for days in range(30, 0, -1)
    ]
    store = InMemoryToolStore(
        snapshots={example.example_id: snapshot},
        histories={"XYZ": bars},
    )
    agent = HeuristicOptionAgent(OptionTools(store))
    trial = agent.run(example)

    assert len(trial.steps) == 3
    assert 0.01 <= trial.probability <= 0.99
    assert aggregate_trials([trial]) == trial.probability

