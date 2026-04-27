from __future__ import annotations

from datetime import datetime, timezone

from murphy.live import call_option_question_text, live_llm_prompt


def test_call_option_question_text() -> None:
    expiration = datetime(2026, 5, 15, tzinfo=timezone.utc)
    text = call_option_question_text("aapl", 215.0, expiration)

    assert text == "Will the price of $AAPL be greater than $215.00 on 2026-05-15?"


def test_live_prompt_includes_cutoff() -> None:
    from murphy.live import LiveQuestion

    timestamp = datetime(2026, 5, 8, 20, 0, tzinfo=timezone.utc)
    question = LiveQuestion(
        question_id="q1",
        example_id="e1",
        symbol="AAPL",
        strike=215.0,
        expiration=datetime(2026, 5, 15, tzinfo=timezone.utc),
        forecast_timestamp=timestamp,
        information_cutoff=timestamp,
        question_text="Will the price of $AAPL be greater than $215.00 on 2026-05-15?",
    )

    prompt = live_llm_prompt(question, "spot: 214.8")
    assert "Use only information" in prompt
    assert timestamp.isoformat() in prompt
    assert "spot: 214.8" in prompt

