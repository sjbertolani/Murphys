from __future__ import annotations

from murphy.analysis_report import build_analysis_report, render_markdown_report


class FakeRepository:
    def evaluation_report(self, ticker=None, limit=10000, include_unresolved=True):
        assert ticker is None
        assert limit == 10000
        assert include_unresolved is True
        return {
            "summary": {
                "n_predictions": 4,
                "n_resolved": 3,
                "n_unresolved": 1,
                "n_scorable": 3,
                "n_posterior_scorable": 3,
                "n_leakage_check_failures": 1,
                "brier_score": 0.145,
                "accuracy_at_0_5": 0.5,
                "posterior_brier_score": 0.12,
                "posterior_accuracy_at_0_5": 1.0,
            },
            "predictions": [
                {
                    "symbol": "AAPL",
                    "question_text": "Will AAPL finish above 200?",
                    "probability": 0.7,
                    "posterior_probability": 0.8,
                    "label": 1,
                    "brier": 0.09,
                    "posterior_brier": 0.04,
                    "dte": 7,
                    "forecast_timestamp": "2026-04-27T16:00:00+00:00",
                    "resolution_due": "2026-05-04T21:00:00+00:00",
                    "strike": 200.0,
                    "leakage_checks": {"ok": True},
                },
                {
                    "symbol": "AAPL",
                    "question_text": "Will AAPL finish above 200?",
                    "probability": 0.6,
                    "posterior_probability": 0.7,
                    "label": 0,
                    "brier": 0.36,
                    "posterior_brier": 0.49,
                    "dte": 8,
                    "forecast_timestamp": "2026-04-27T17:00:00+00:00",
                    "resolution_due": "2026-05-04T21:00:00+00:00",
                    "strike": 200.0,
                    "leakage_checks": {"ok": False},
                },
                {
                    "symbol": "MSFT",
                    "question_text": "Will MSFT finish above 400?",
                    "probability": 0.55,
                    "posterior_probability": 0.58,
                    "label": 1,
                    "brier": 0.2025,
                    "posterior_brier": 0.1764,
                    "dte": 12,
                    "forecast_timestamp": "2026-04-28T16:00:00+00:00",
                    "resolution_due": "2026-05-04T21:00:00+00:00",
                    "strike": 400.0,
                    "leakage_checks": {"ok": True},
                },
                {
                    "symbol": "MSFT",
                    "question_text": "Will MSFT finish above 400?",
                    "probability": 0.55,
                    "posterior_probability": 0.58,
                    "label": None,
                    "brier": None,
                    "posterior_brier": None,
                    "dte": 12,
                    "forecast_timestamp": "2026-04-29T16:00:00+00:00",
                    "resolution_due": "2026-05-04T21:00:00+00:00",
                    "strike": 400.0,
                    "leakage_checks": {"ok": True},
                },
            ],
        }


def test_build_analysis_report_summarizes_predictions() -> None:
    report = build_analysis_report(FakeRepository())

    assert report["summary"]["n_predictions"] == 4
    assert report["ticker_breakdown"][0]["ticker"] == "AAPL"
    assert report["ticker_breakdown"][0]["n_resolved"] == 2
    assert report["dte_breakdown"][0]["dte_bucket"] == "5-10"
    assert report["repeated_questions"][0]["count"] == 2
    assert len(report["leakage_failures"]) == 1
    assert report["grouped_split"]["eligible_rows"] == 2
    assert report["grouped_split"]["excluded_rows"] == {
        "leakage_failures": 1,
        "unresolved": 1,
    }
    assert report["grouped_split"]["manifest"]["n_contract_groups"] == 2
    assert report["grouped_split"]["split_breakdown"][0]["split"] == "train"
    assert report["grouped_split"]["split_breakdown"][1]["split"] == "test"


def test_render_markdown_report_contains_core_sections() -> None:
    markdown = render_markdown_report(build_analysis_report(FakeRepository()))

    assert "# Murphy Offline Analysis Report" in markdown
    assert "## Ticker Breakdown" in markdown
    assert "## Grouped Dataset Split" in markdown
    assert "symbol|resolution_due|strike" in markdown
    assert "Will AAPL finish above 200?" in markdown
