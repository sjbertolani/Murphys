from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from typing import Any
from uuid import uuid4

import numpy as np

from murphy.priors import clamp_probability, logit
from murphy.training.datasets import contract_group_key


SHADOW_METHOD = "shadow_blf_v2_learned_logit_ensemble"


@dataclass(frozen=True)
class ShadowForecastResult:
    example_id: str
    question_id: str
    symbol: str
    raw_probability: float
    fixed_posterior_probability: float
    calibrated_probability: float
    aggregate_probability: float
    prior_contract_groups: int


def run_shadow_blf_v2(
    repository,
    *,
    ticker: str | None = None,
    limit: int = 50,
    min_train_groups: int = 100,
    report_limit: int = 10000,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Score unresolved rows with the selected walk-forward calibration candidate.

    This is shadow-only: it writes a separate forecast method and trace trials,
    but leaves raw LLM responses and current fixed posterior behavior untouched.
    """
    created_at = created_at or datetime.now(UTC)
    report = repository.evaluation_report(
        ticker=ticker,
        limit=report_limit,
        include_unresolved=True,
    )
    predictions = report.get("predictions", [])
    resolved_rows = _eligible_resolved(predictions)
    target_rows = _eligible_targets(predictions)[:limit]
    selected = []
    skipped = []
    for target in target_rows:
        training_rows = _prior_training_rows(resolved_rows, target)
        prior_groups = {row["contract_group_key"] for row in training_rows}
        if len(prior_groups) < min_train_groups:
            skipped.append(
                {
                    "question_id": target.get("question_id"),
                    "example_id": target.get("example_id"),
                    "reason": "not_enough_prior_contract_groups",
                    "prior_contract_groups": len(prior_groups),
                }
            )
            continue
        calibrated_probability = _fit_and_predict_learned_ensemble(training_rows, target)
        raw_probability = _probability(target["probability"])
        fixed_posterior_probability = _probability(target["posterior_probability"])
        aggregate_probability = calibrated_probability
        result = ShadowForecastResult(
            example_id=str(target["example_id"]),
            question_id=str(target["question_id"]),
            symbol=str(target.get("symbol") or ""),
            raw_probability=raw_probability,
            fixed_posterior_probability=fixed_posterior_probability,
            calibrated_probability=calibrated_probability,
            aggregate_probability=aggregate_probability,
            prior_contract_groups=len(prior_groups),
        )
        _persist_shadow_result(repository, result, created_at)
        selected.append(result)
    return {
        "created_at": created_at,
        "method": SHADOW_METHOD,
        "selected_parameters": {
            "candidate": "learned_logit_ensemble",
            "min_train_groups": min_train_groups,
            "trial_probabilities": [
                "raw_llm_probability",
                "fixed_blf_posterior",
                "learned_logit_ensemble",
            ],
            "aggregation": (
                "selected shadow probability is learned_logit_ensemble; "
                "raw_llm_probability and fixed_blf_posterior remain stored as comparison traces"
            ),
        },
        "n_resolved_training_rows": len(resolved_rows),
        "n_candidate_targets": len(target_rows),
        "n_shadow_forecasts": len(selected),
        "n_skipped": len(skipped),
        "forecasts": [result.__dict__ for result in selected],
        "skipped": skipped,
    }


def _eligible_resolved(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in predictions:
        if item.get("label") is None:
            continue
        if item.get("probability") is None or item.get("posterior_probability") is None:
            continue
        if not all(item.get("leakage_checks", {}).values()):
            continue
        timestamp = _parse_datetime(item.get("forecast_timestamp"))
        if timestamp is None:
            continue
        rows.append({**item, "forecast_timestamp": timestamp, "contract_group_key": contract_group_key(item)})
    return sorted(rows, key=lambda item: (item["forecast_timestamp"], item["contract_group_key"]))


def _eligible_targets(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in predictions:
        if item.get("label") is not None:
            continue
        if item.get("probability") is None or item.get("posterior_probability") is None:
            continue
        if not all(item.get("leakage_checks", {}).values()):
            continue
        timestamp = _parse_datetime(item.get("forecast_timestamp"))
        if timestamp is None:
            continue
        rows.append({**item, "forecast_timestamp": timestamp, "contract_group_key": contract_group_key(item)})
    return sorted(rows, key=lambda item: (item["forecast_timestamp"], item["contract_group_key"]))


def _prior_training_rows(
    resolved_rows: list[dict[str, Any]],
    target: dict[str, Any],
) -> list[dict[str, Any]]:
    target_time = target["forecast_timestamp"]
    target_group = target["contract_group_key"]
    return [
        row
        for row in resolved_rows
        if row["forecast_timestamp"] < target_time and row["contract_group_key"] != target_group
    ]


def _fit_and_predict_learned_ensemble(
    training_rows: list[dict[str, Any]],
    target: dict[str, Any],
) -> float:
    from sklearn.linear_model import LogisticRegression

    labels = np.asarray([int(row["label"]) for row in training_rows], dtype=int)
    if len(set(labels.tolist())) < 2:
        raise ValueError("learned ensemble requires both labels in training data")
    x_train = np.asarray(
        [
            [logit(_probability(row["probability"])), logit(_probability(row["posterior_probability"]))]
            for row in training_rows
        ],
        dtype=float,
    )
    x_target = np.asarray(
        [[logit(_probability(target["probability"])), logit(_probability(target["posterior_probability"]))]],
        dtype=float,
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, labels)
    return clamp_probability(float(model.predict_proba(x_target)[0, 1]), eps=0.01)


def _persist_shadow_result(
    repository,
    result: ShadowForecastResult,
    created_at: datetime,
) -> None:
    if hasattr(repository, "db"):
        _persist_shadow_result_duckdb(repository.db.conn, result, created_at)
        return
    if hasattr(repository, "engine"):
        import sqlalchemy

        with repository.engine.begin() as conn:
            _ensure_postgres_forecasts_table(conn, sqlalchemy)
            _persist_shadow_result_postgres(conn, sqlalchemy, result, created_at)
        return
    raise TypeError(f"Unsupported repository type: {type(repository)!r}")


def _persist_shadow_result_duckdb(conn, result: ShadowForecastResult, created_at: datetime) -> None:
    trial_rows, step_rows = _trial_and_step_rows(result, created_at)
    conn.execute(
        """
        INSERT OR REPLACE INTO forecasts
          (example_id, method, raw_probability, aggregate_probability, calibrated_probability, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            result.example_id,
            SHADOW_METHOD,
            result.calibrated_probability,
            result.aggregate_probability,
            result.calibrated_probability,
            created_at,
        ],
    )
    conn.executemany(
        """
        INSERT INTO agent_trials
          (trial_id, example_id, seed, model, started_at, raw_probability, status)
        VALUES (?, ?, ?, ?, ?, ?, 'completed')
        """,
        trial_rows,
    )
    conn.executemany(
        """
        INSERT INTO agent_steps
          (trial_id, step_index, action_type, action_json, observation_ref, belief_json, probability)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        step_rows,
    )


def _ensure_postgres_forecasts_table(conn, sqlalchemy) -> None:
    conn.execute(
        sqlalchemy.text(
            """
            CREATE TABLE IF NOT EXISTS forecasts (
              example_id TEXT NOT NULL REFERENCES option_examples(example_id),
              method TEXT NOT NULL,
              raw_probability DOUBLE PRECISION NOT NULL,
              aggregate_probability DOUBLE PRECISION,
              calibrated_probability DOUBLE PRECISION,
              created_at TIMESTAMPTZ NOT NULL,
              PRIMARY KEY (example_id, method, created_at)
            )
            """
        )
    )


def _persist_shadow_result_postgres(conn, sqlalchemy, result: ShadowForecastResult, created_at: datetime) -> None:
    trial_rows, step_rows = _trial_and_step_rows(result, created_at)
    conn.execute(
        sqlalchemy.text(
            """
            INSERT INTO forecasts
              (example_id, method, raw_probability, aggregate_probability, calibrated_probability, created_at)
            VALUES
              (:example_id, :method, :raw_probability, :aggregate_probability,
               :calibrated_probability, :created_at)
            """
        ),
        {
            "example_id": result.example_id,
            "method": SHADOW_METHOD,
            "raw_probability": result.calibrated_probability,
            "aggregate_probability": result.aggregate_probability,
            "calibrated_probability": result.calibrated_probability,
            "created_at": created_at,
        },
    )
    conn.execute(
        sqlalchemy.text(
            """
            INSERT INTO agent_trials
              (trial_id, example_id, seed, model, started_at, raw_probability, status)
            VALUES
              (:trial_id, :example_id, :seed, :model, :started_at, :raw_probability, 'completed')
            """
        ),
        [
            {
                "trial_id": trial_id,
                "example_id": example_id,
                "seed": seed,
                "model": model,
                "started_at": started_at,
                "raw_probability": raw_probability,
            }
            for trial_id, example_id, seed, model, started_at, raw_probability in trial_rows
        ],
    )
    conn.execute(
        sqlalchemy.text(
            """
            INSERT INTO agent_steps
              (trial_id, step_index, action_type, action_json, observation_ref, belief_json, probability)
            VALUES
              (:trial_id, :step_index, :action_type, CAST(:action_json AS JSONB),
               :observation_ref, CAST(:belief_json AS JSONB), :probability)
            """
        ),
        [
            {
                "trial_id": trial_id,
                "step_index": step_index,
                "action_type": action_type,
                "action_json": action_json,
                "observation_ref": observation_ref,
                "belief_json": belief_json,
                "probability": probability,
            }
            for trial_id, step_index, action_type, action_json, observation_ref, belief_json, probability in step_rows
        ],
    )


def _trial_and_step_rows(
    result: ShadowForecastResult,
    created_at: datetime,
) -> tuple[list[tuple], list[tuple]]:
    prefix = f"shadow-blf-v2:{result.question_id}:{uuid4()}"
    trial_rows = [
        (f"{prefix}:raw-llm", result.example_id, 1, SHADOW_METHOD, created_at, result.raw_probability),
        (
            f"{prefix}:fixed-posterior",
            result.example_id,
            2,
            SHADOW_METHOD,
            created_at,
            result.fixed_posterior_probability,
        ),
        (
            f"{prefix}:learned-ensemble",
            result.example_id,
            3,
            SHADOW_METHOD,
            created_at,
            result.calibrated_probability,
        ),
    ]
    raw_trial_id = trial_rows[0][0]
    posterior_trial_id = trial_rows[1][0]
    ensemble_trial_id = trial_rows[2][0]
    step_rows = [
        *_belief_steps(
            raw_trial_id,
            result.question_id,
            "raw_llm_probability",
            result.raw_probability,
            "Submit the already-recorded LLM probability as one independent shadow trial.",
        ),
        *_belief_steps(
            posterior_trial_id,
            result.question_id,
            "fixed_blf_posterior",
            result.fixed_posterior_probability,
            "Submit the current fixed BLF posterior as one independent shadow trial.",
        ),
        *_belief_steps(
            ensemble_trial_id,
            result.question_id,
            "learned_logit_ensemble",
            result.calibrated_probability,
            "Submit the learned walk-forward logit ensemble as one independent shadow trial.",
        ),
    ]
    return trial_rows, step_rows


def _belief_steps(
    trial_id: str,
    question_id: str,
    action_name: str,
    probability: float,
    reasoning: str,
) -> list[tuple]:
    initial = {
        "probability": 0.5,
        "confidence": 0.0,
        "evidence_for": [],
        "evidence_against": [],
        "open_questions": ["Evaluate this shadow trial candidate."],
        "update_reasoning": "Shadow BLF v2 trial initialized.",
    }
    final = {
        "probability": probability,
        "confidence": abs(probability - 0.5) * 2.0,
        "evidence_for": [f"Candidate source: {action_name}."],
        "evidence_against": [],
        "open_questions": [],
        "update_reasoning": reasoning,
    }
    return [
        (
            trial_id,
            0,
            "initialize_shadow_belief",
            json.dumps({"type": "initialize_shadow_belief"}),
            question_id,
            json.dumps(initial, sort_keys=True),
            0.5,
        ),
        (
            trial_id,
            1,
            action_name,
            json.dumps({"type": action_name, "question_id": question_id}),
            question_id,
            json.dumps(final, sort_keys=True),
            probability,
        ),
        (
            trial_id,
            2,
            "submit_shadow_probability",
            json.dumps({"type": "submit_shadow_probability", "source": action_name}),
            question_id,
            json.dumps(final, sort_keys=True),
            probability,
        ),
    ]


def _probability(value: Any) -> float:
    return clamp_probability(float(value), eps=0.01)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None
