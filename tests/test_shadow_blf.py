from __future__ import annotations

from datetime import datetime, timedelta, timezone

from murphy.db import MurphyDb
from murphy.shadow_blf import SHADOW_METHOD, run_shadow_blf_v2


class FakeRepository:
    def __init__(self, db_path, predictions: list[dict]) -> None:
        self.db = MurphyDb(db_path)
        self.db.initialize()
        self._predictions = predictions

    def evaluation_report(self, ticker=None, limit=10000, include_unresolved=True):
        del ticker, limit, include_unresolved
        return {"predictions": self._predictions}

    def close(self) -> None:
        self.db.close()


def test_run_shadow_blf_v2_persists_shadow_forecast(tmp_path) -> None:
    db_path = tmp_path / "murphy.duckdb"
    start = datetime(2026, 5, 1, 16, 0, tzinfo=timezone.utc)
    resolved_rows = []
    for index, (label, probability, posterior) in enumerate(
        [
            (0, 0.20, 0.18),
            (1, 0.70, 0.74),
            (0, 0.35, 0.30),
            (1, 0.60, 0.66),
        ]
    ):
        forecast_timestamp = start + timedelta(hours=index)
        resolution_due = forecast_timestamp + timedelta(days=index + 1)
        resolved_rows.append(
            {
                "example_id": f"example-{index}",
                "question_id": f"question-{index}",
                "symbol": "AAPL",
                "forecast_timestamp": forecast_timestamp,
                "resolution_due": resolution_due,
                "strike": float(200 + index),
                "probability": probability,
                "posterior_probability": posterior,
                "label": label,
                "leakage_checks": {"ok": True},
            }
        )
    unresolved = {
        "example_id": "example-target",
        "question_id": "question-target",
        "symbol": "AAPL",
        "forecast_timestamp": start + timedelta(days=3),
        "resolution_due": start + timedelta(days=10),
        "strike": 250.0,
        "probability": 0.58,
        "posterior_probability": 0.62,
        "label": None,
        "leakage_checks": {"ok": True},
    }
    repository = FakeRepository(db_path, [*resolved_rows, unresolved])
    try:
        result = run_shadow_blf_v2(
            repository,
            limit=10,
            min_train_groups=2,
        )

        assert result["n_shadow_forecasts"] == 1
        assert result["n_skipped"] == 0
        assert result["forecasts"][0]["question_id"] == "question-target"

        forecast_row = repository.db.conn.execute(
            """
            SELECT method, raw_probability, aggregate_probability, calibrated_probability
            FROM forecasts
            """
        ).fetchone()
        assert forecast_row[0] == SHADOW_METHOD
        assert 0.0 < float(forecast_row[1]) < 1.0
        assert 0.0 < float(forecast_row[2]) < 1.0
        assert 0.0 < float(forecast_row[3]) < 1.0

        assert repository.db.conn.execute("SELECT count(*) FROM agent_trials").fetchone()[0] == 3
        assert repository.db.conn.execute("SELECT count(*) FROM agent_steps").fetchone()[0] == 9
    finally:
        repository.close()
