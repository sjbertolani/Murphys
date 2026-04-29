from __future__ import annotations

import json
import sys

from murphy import cli


class FakeRepository:
    def __init__(self) -> None:
        self.closed = False

    def evaluation_report(self, ticker=None, limit=50, include_unresolved=True):
        return {
            "ticker": ticker,
            "include_unresolved": include_unresolved,
            "summary": {"n_predictions": limit},
            "predictions": [],
        }

    def close(self) -> None:
        self.closed = True


def test_evaluation_report_cli_does_not_require_analysis_split_args(monkeypatch, capsys) -> None:
    repository = FakeRepository()
    monkeypatch.setattr(sys, "argv", ["murphy", "evaluation-report", "--ticker", "AAPL"])
    monkeypatch.setattr(cli, "_repository_for_backend", lambda _backend, _db: repository)

    cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output["ticker"] == "AAPL"
    assert output["summary"]["n_predictions"] == 50
    assert repository.closed is True
