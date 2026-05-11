from __future__ import annotations

from datetime import datetime, timedelta, timezone

from murphy.walk_forward import build_walk_forward_report, render_walk_forward_markdown


class FakeRepository:
    def __init__(self) -> None:
        self.closed = False

    def evaluation_report(self, ticker=None, limit=10000, include_unresolved=False):
        assert ticker is None
        assert limit == 10000
        assert include_unresolved is False
        start = datetime(2026, 4, 27, 16, 0, tzinfo=timezone.utc)
        rows = []
        labels = [0, 1, 0, 1, 0, 1, 1, 0]
        llm_probs = [0.25, 0.65, 0.35, 0.55, 0.75, 0.45, 0.8, 0.2]
        posterior_probs = [0.2, 0.7, 0.3, 0.6, 0.65, 0.4, 0.82, 0.18]
        for index, (label, llm_prob, posterior_prob) in enumerate(
            zip(labels, llm_probs, posterior_probs, strict=True)
        ):
            rows.append(
                {
                    "question_id": f"q-{index}",
                    "symbol": "AAPL",
                    "question_text": f"Will AAPL finish above {100 + index}?",
                    "forecast_timestamp": start + timedelta(hours=index),
                    "resolution_due": start + timedelta(days=index + 1),
                    "strike": float(100 + index),
                    "probability": llm_prob,
                    "posterior_probability": posterior_prob,
                    "label": label,
                    "leakage_checks": {"ok": True},
                }
            )
        rows.append(
            {
                "question_id": "bad-leakage",
                "symbol": "AAPL",
                "forecast_timestamp": start + timedelta(hours=9),
                "resolution_due": start + timedelta(days=9),
                "strike": 200.0,
                "probability": 0.9,
                "posterior_probability": 0.9,
                "label": 1,
                "leakage_checks": {"ok": False},
            }
        )
        return {"summary": {}, "predictions": rows}

    def close(self) -> None:
        self.closed = True


def test_build_walk_forward_report_scores_expanding_folds() -> None:
    report = build_walk_forward_report(
        FakeRepository(),
        n_folds=2,
        min_train_groups=4,
        min_test_groups=1,
    )

    assert report["eligible_rows"] == 8
    assert report["eligible_contract_groups"] == 8
    assert report["excluded_rows"]["leakage_failures"] == 1
    assert report["n_folds"] == 2
    methods = {row["method"] for row in report["summary"]}
    assert "llm_probability" in methods
    assert "fixed_blf_posterior" in methods
    assert "platt_llm_probability" in methods
    assert "learned_logit_ensemble" in methods
    assert all(row["n"] > 0 for row in report["summary"])


def test_render_walk_forward_markdown_contains_core_sections() -> None:
    report = build_walk_forward_report(
        FakeRepository(),
        n_folds=2,
        min_train_groups=4,
        min_test_groups=1,
    )

    markdown = render_walk_forward_markdown(report)

    assert "# Murphy Walk-Forward Evaluation" in markdown
    assert "## Method Summary" in markdown
    assert "fixed_blf_posterior" in markdown
    assert "learned_logit_ensemble" in markdown
