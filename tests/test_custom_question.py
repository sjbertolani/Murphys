from __future__ import annotations

from datetime import datetime, timedelta, timezone

from murphy.custom_question import (
    CURRENT_PRIMARY_CANDIDATE,
    LEARNED_ENSEMBLE_CANDIDATE,
    WALK_FORWARD_SHADOW_CANDIDATE,
    score_custom_question,
)
from murphy.predict import DeterministicPredictor
from murphy.schemas import OptionRight, OptionSnapshot, UnderlyingBar
from murphy.web_context import NewsContextItem


class FakeRepository:
    def __init__(self, predictions: list[dict]) -> None:
        self._predictions = predictions
        self.external_calls: list[dict] = []

    def evaluation_report(self, ticker=None, limit=10000, include_unresolved=False):
        del ticker, limit, include_unresolved
        return {"predictions": self._predictions}

    def record_external_call(self, **payload):
        self.external_calls.append(payload)
        return f"call-{len(self.external_calls)}"

    def close(self) -> None:
        return None


class FakeMarketProvider:
    name = "fake-market"

    def __init__(self, bars: list[UnderlyingBar], snapshots: list[OptionSnapshot]) -> None:
        self._bars = bars
        self._snapshots = snapshots

    def fetch_underlying_bars(self, tickers, lookback_days, as_of=None):
        del tickers, lookback_days, as_of
        return list(self._bars)

    def fetch_option_chain_snapshots(self, tickers, min_dte, max_dte, as_of=None):
        del tickers, min_dte, max_dte, as_of
        return list(self._snapshots)


class FakeNewsProvider:
    name = "fake-news"

    def __init__(self, items: list[NewsContextItem]) -> None:
        self._items = items

    def fetch(self, tickers, limit_per_ticker=5):
        del tickers, limit_per_ticker
        return list(self._items)


def test_score_custom_question_uses_platt_llm_when_history_is_deep_enough() -> None:
    as_of = datetime(2026, 5, 11, 20, 0, tzinfo=timezone.utc)
    rows = []
    for index in range(20):
        label = index % 2
        probability = 0.25 if label == 0 else 0.75
        posterior = 0.22 if label == 0 else 0.78
        rows.append(
            {
                "question_id": f"q-{index}",
                "example_id": f"ex-{index}",
                "symbol": "AAPL",
                "forecast_timestamp": as_of - timedelta(days=30 - index),
                "resolution_due": as_of - timedelta(days=27 - index),
                "strike": 295.0 + (index % 2),
                "spot": 292.0 + (index % 2),
                "dte": 35.0,
                "moneyness": -0.005,
                "probability": probability,
                "posterior_probability": posterior,
                "label": label,
                "leakage_checks": {"ok": True},
            }
        )
    repository = FakeRepository(rows)
    provider = FakeMarketProvider(
        bars=[
            UnderlyingBar(
                symbol="AAPL",
                timestamp=as_of - timedelta(days=offset),
                open=290.0 + offset,
                high=291.0 + offset,
                low=289.0 + offset,
                close=290.5 + offset,
                volume=1000 + offset,
            )
            for offset in range(3, 0, -1)
        ],
        snapshots=[
            OptionSnapshot(
                symbol="AAPL",
                option_symbol="AAPL260615C00295000",
                quote_timestamp=as_of,
                expiration=datetime(2026, 6, 15, 21, 0, tzinfo=timezone.utc),
                strike=295.0,
                right=OptionRight.CALL,
                bid=4.8,
                ask=5.2,
                mid=5.0,
                implied_volatility=0.24,
                volume=1200,
                open_interest=5000,
                spot=293.4,
            )
        ],
    )
    news_provider = FakeNewsProvider(
        [
            NewsContextItem(
                ticker="AAPL",
                title="Apple announces supply update",
                link="https://example.com/apple",
                published_at=as_of - timedelta(hours=2),
                source="Example",
                summary="Short summary",
            )
        ]
    )

    result = score_custom_question(
        repository,
        provider,
        DeterministicPredictor(probability=0.61, model="dry-run"),
        symbol="AAPL",
        strike=295.0,
        target_date="2026-06-15",
        model="dry-run",
        news_provider=news_provider,
        min_train_groups=2,
        report_limit=100,
        as_of=as_of,
    )

    assert result["question_text"] == "Will the price of $AAPL be greater than $295.00 on 2026-06-15?"
    assert result["selected_method"] == WALK_FORWARD_SHADOW_CANDIDATE
    assert result["selected_method"] == "platt_llm_probability"
    assert result["platt_llm_probability"] is not None
    assert result["learned_logit_ensemble_probability"] is not None
    assert result["calibration"]["used"] is True
    assert result["calibration"]["method"] == "platt_llm_probability"
    assert result["calibration_candidates"][LEARNED_ENSEMBLE_CANDIDATE]["used"] is True
    assert result["selected_prediction_at_0_5"] in {0, 1}
    assert result["prior_components"]["historical_empirical_scope"] == "symbol_moneyness_dte"
    assert len(repository.external_calls) == 3


def test_score_custom_question_falls_back_to_fixed_posterior_when_history_is_thin() -> None:
    as_of = datetime(2026, 5, 11, 20, 0, tzinfo=timezone.utc)
    repository = FakeRepository(
        [
            {
                "question_id": "q-1",
                "example_id": "ex-1",
                "symbol": "AAPL",
                "forecast_timestamp": as_of - timedelta(days=5),
                "resolution_due": as_of - timedelta(days=2),
                "strike": 300.0,
                "spot": 298.0,
                "dte": 6.0,
                "moneyness": -0.006,
                "probability": 0.45,
                "posterior_probability": 0.47,
                "label": 0,
                "leakage_checks": {"ok": True},
            }
        ]
    )
    provider = FakeMarketProvider(
        bars=[
            UnderlyingBar(
                symbol="AAPL",
                timestamp=as_of - timedelta(days=1),
                open=292.0,
                high=294.0,
                low=291.0,
                close=293.0,
                volume=1000,
            )
        ],
        snapshots=[
            OptionSnapshot(
                symbol="AAPL",
                option_symbol="AAPL260619C00295000",
                quote_timestamp=as_of,
                expiration=datetime(2026, 6, 19, 21, 0, tzinfo=timezone.utc),
                strike=295.0,
                right=OptionRight.CALL,
                bid=5.0,
                ask=5.5,
                mid=5.25,
                implied_volatility=0.22,
                volume=100,
                open_interest=1000,
                spot=293.0,
            )
        ],
    )

    result = score_custom_question(
        repository,
        provider,
        DeterministicPredictor(probability=0.52, model="dry-run"),
        symbol="AAPL",
        strike=295.0,
        target_date="06-15-26",
        model="dry-run",
        news_provider=FakeNewsProvider([]),
        min_train_groups=5,
        report_limit=100,
        as_of=as_of,
    )

    assert result["selected_method"] == CURRENT_PRIMARY_CANDIDATE
    assert result["platt_llm_probability"] is None
    assert result["learned_logit_ensemble_probability"] is None
    assert result["calibration"]["fallback_reason"] == "not_enough_prior_contract_groups"
    assert result["proxy_option"]["expiration"].startswith("2026-06-19")
    assert result["proxy_option"]["expiry_gap_days"] == 4.0
