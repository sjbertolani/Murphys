from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from statistics import mean
from typing import Any

import numpy as np

from murphy.metrics import brier_score, expected_calibration_error, log_loss
from murphy.priors import clamp_probability, logit
from murphy.training.datasets import contract_group_key


def build_walk_forward_report(
    repository,
    *,
    ticker: str | None = None,
    limit: int = 10000,
    n_folds: int = 5,
    min_train_groups: int = 20,
    min_test_groups: int = 1,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate resolved forecasts with expanding-window, contract-group folds.

    Contract groups are the leakage boundary for repeated hourly forecasts of
    the same ticker/strike/expiry question. Each fold trains/tunes only on
    earlier contract groups and tests on later groups.
    """
    evaluation = repository.evaluation_report(
        ticker=ticker,
        limit=limit,
        include_unresolved=False,
    )
    candidates = _eligible_predictions(evaluation.get("predictions", []))
    folds, split_warnings = _walk_forward_folds(
        candidates,
        n_folds=n_folds,
        min_train_groups=min_train_groups,
        min_test_groups=min_test_groups,
    )
    method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_rows: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds, start=1):
        train_rows = fold["train_rows"]
        test_rows = fold["test_rows"]
        predictions = _fold_predictions(train_rows, test_rows)
        for method, probabilities in predictions.items():
            labels = [int(item["label"]) for item in test_rows]
            scored_rows = [
                {
                    "fold": fold_index,
                    "question_id": item.get("question_id"),
                    "contract_group_key": item["contract_group_key"],
                    "forecast_timestamp": item.get("forecast_timestamp"),
                    "symbol": item.get("symbol"),
                    "probability": float(probability),
                    "label": int(item["label"]),
                }
                for item, probability in zip(test_rows, probabilities, strict=True)
            ]
            method_rows[method].extend(scored_rows)
            fold_rows.append(
                {
                    "fold": fold_index,
                    "method": method,
                    "train_rows": len(train_rows),
                    "train_contract_groups": len(fold["train_group_keys"]),
                    "test_rows": len(test_rows),
                    "test_contract_groups": len(fold["test_group_keys"]),
                    **_metrics(probabilities, labels),
                }
            )

    summary_rows = [
        {"method": method, **_metrics_for_rows(rows)}
        for method, rows in sorted(method_rows.items())
    ]
    recommendation = _candidate_recommendation(summary_rows, fold_rows)
    return {
        "generated_at": generated_at or datetime.now(UTC),
        "ticker": None if ticker is None else ticker.upper(),
        "limit": limit,
        "n_folds_requested": n_folds,
        "n_folds": len(folds),
        "min_train_groups": min_train_groups,
        "min_test_groups": min_test_groups,
        "eligible_rows": len(candidates),
        "eligible_contract_groups": len({item["contract_group_key"] for item in candidates}),
        "excluded_rows": _excluded_counts(evaluation.get("predictions", [])),
        "methods": _method_descriptions(),
        "recommendation": recommendation,
        "summary": summary_rows,
        "folds": fold_rows,
        "warnings": split_warnings,
    }


def render_walk_forward_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Murphy Walk-Forward Evaluation",
        "",
        f"Generated: {report['generated_at']}",
        f"Ticker filter: {report['ticker'] or 'all'}",
        f"Eligible resolved rows: {report['eligible_rows']}",
        f"Eligible contract groups: {report['eligible_contract_groups']}",
        f"Folds: {report['n_folds']} / requested {report['n_folds_requested']}",
        "",
        "## Method Summary",
        "",
        _table(report["summary"]),
        "",
        "## Recommendation",
        "",
        _recommendation_markdown(report["recommendation"]),
        "",
        "## Fold Metrics",
        "",
        _table(report["folds"]),
        "",
        "## Methods",
        "",
    ]
    for method in report["methods"]:
        lines.append(f"- `{method['method']}`: {method['description']}")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    lines.extend(["", "## Excluded Rows", "", _table([report["excluded_rows"]])])
    return "\n".join(lines) + "\n"


def _eligible_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in predictions:
        if item.get("label") is None or item.get("probability") is None:
            continue
        if not all(item.get("leakage_checks", {}).values()):
            continue
        forecast_timestamp = _parse_datetime(item.get("forecast_timestamp"))
        if forecast_timestamp is None:
            continue
        rows.append(
            {
                **item,
                "forecast_timestamp": forecast_timestamp,
                "contract_group_key": contract_group_key(item),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            item["forecast_timestamp"],
            str(item.get("symbol") or ""),
            str(item.get("resolution_due") or ""),
            float(item.get("strike") or 0.0),
            str(item.get("question_id") or ""),
        ),
    )


def _walk_forward_folds(
    rows: list[dict[str, Any]],
    *,
    n_folds: int,
    min_train_groups: int,
    min_test_groups: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if min_train_groups <= 0:
        raise ValueError("min_train_groups must be positive")
    if min_test_groups <= 0:
        raise ValueError("min_test_groups must be positive")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in rows:
        groups[item["contract_group_key"]].append(item)
    ordered_group_keys = sorted(
        groups,
        key=lambda key: (
            min(item["forecast_timestamp"] for item in groups[key]),
            key,
        ),
    )
    warnings: list[str] = []
    if len(ordered_group_keys) < min_train_groups + min_test_groups:
        warnings.append("not_enough_contract_groups_for_walk_forward")
        return [], warnings

    available_test_groups = len(ordered_group_keys) - min_train_groups
    fold_count = min(n_folds, available_test_groups // min_test_groups)
    if fold_count <= 0:
        warnings.append("not_enough_test_groups_for_requested_minimum")
        return [], warnings
    test_group_count = max(min_test_groups, available_test_groups // fold_count)

    folds = []
    test_start = min_train_groups
    for fold_index in range(fold_count):
        test_end = test_start + test_group_count
        if fold_index == fold_count - 1:
            test_end = len(ordered_group_keys)
        test_end = min(test_end, len(ordered_group_keys))
        if test_end <= test_start:
            break
        train_group_keys = ordered_group_keys[:test_start]
        test_group_keys = ordered_group_keys[test_start:test_end]
        folds.append(
            {
                "train_group_keys": train_group_keys,
                "test_group_keys": test_group_keys,
                "train_rows": _rows_for_groups(groups, train_group_keys),
                "test_rows": _rows_for_groups(groups, test_group_keys),
            }
        )
        test_start = test_end
    return folds, warnings


def _fold_predictions(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[str, list[float]]:
    methods: dict[str, list[float]] = {}
    labels = [int(item["label"]) for item in train_rows]
    base_rate = (sum(labels) + 1.0) / (len(labels) + 2.0)
    methods["empirical_base_rate"] = [base_rate] * len(test_rows)
    methods["llm_probability"] = [_probability(item["probability"]) for item in test_rows]

    if all(item.get("posterior_probability") is not None for item in test_rows):
        methods["fixed_blf_posterior"] = [
            _probability(item["posterior_probability"]) for item in test_rows
        ]

    llm_calibrator = _fit_platt(
        [_probability(item["probability"]) for item in train_rows],
        labels,
    )
    if llm_calibrator is not None:
        methods["platt_llm_probability"] = llm_calibrator(methods["llm_probability"])

    posterior_train_rows = [
        item for item in train_rows if item.get("posterior_probability") is not None
    ]
    if posterior_train_rows and all(item.get("posterior_probability") is not None for item in test_rows):
        posterior_labels = [int(item["label"]) for item in posterior_train_rows]
        train_posterior = [
            _probability(item["posterior_probability"]) for item in posterior_train_rows
        ]
        test_posterior = [_probability(item["posterior_probability"]) for item in test_rows]
        posterior_calibrator = _fit_platt(train_posterior, posterior_labels)
        if posterior_calibrator is not None:
            methods["platt_fixed_blf_posterior"] = posterior_calibrator(test_posterior)
        ensemble = _fit_logit_ensemble(
            [_probability(item["probability"]) for item in posterior_train_rows],
            train_posterior,
            posterior_labels,
        )
        if ensemble is not None:
            methods["learned_logit_ensemble"] = ensemble(
                methods["llm_probability"],
                test_posterior,
            )
    return methods


def _fit_platt(train_probabilities: list[float], labels: list[int]):
    if len(set(labels)) < 2:
        return None
    from sklearn.linear_model import LogisticRegression

    x_train = np.array([logit(p) for p in train_probabilities], dtype=float).reshape(-1, 1)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, np.asarray(labels, dtype=int))

    def predict(test_probabilities: list[float]) -> list[float]:
        x_test = np.array([logit(p) for p in test_probabilities], dtype=float).reshape(-1, 1)
        return [float(value) for value in model.predict_proba(x_test)[:, 1]]

    return predict


def _fit_logit_ensemble(
    train_llm: list[float],
    train_posterior: list[float],
    labels: list[int],
):
    if len(set(labels)) < 2:
        return None
    from sklearn.linear_model import LogisticRegression

    x_train = np.asarray(
        [[logit(llm), logit(posterior)] for llm, posterior in zip(train_llm, train_posterior)],
        dtype=float,
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, np.asarray(labels, dtype=int))

    def predict(test_llm: list[float], test_posterior: list[float]) -> list[float]:
        x_test = np.asarray(
            [[logit(llm), logit(posterior)] for llm, posterior in zip(test_llm, test_posterior)],
            dtype=float,
        )
        return [float(value) for value in model.predict_proba(x_test)[:, 1]]

    return predict


def _metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    probabilities = [float(row["probability"]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    return _metrics(probabilities, labels)


def _metrics(probabilities: list[float], labels: list[int]) -> dict[str, Any]:
    predicted = [1 if probability >= 0.5 else 0 for probability in probabilities]
    n = len(labels)
    return {
        "n": n,
        "brier_score": brier_score(probabilities, labels),
        "log_loss": log_loss(probabilities, labels),
        "accuracy_at_0_5": sum(int(p == y) for p, y in zip(predicted, labels, strict=True)) / n,
        "ece_10": expected_calibration_error(probabilities, labels, bins=10),
        "avg_probability": mean(probabilities),
        "positive_rate": mean(labels),
        "true_negative_0_to_0": sum(
            int(p == 0 and y == 0) for p, y in zip(predicted, labels, strict=True)
        ),
        "true_positive_1_to_1": sum(
            int(p == 1 and y == 1) for p, y in zip(predicted, labels, strict=True)
        ),
        "false_negative_0_to_1": sum(
            int(p == 0 and y == 1) for p, y in zip(predicted, labels, strict=True)
        ),
        "false_positive_1_to_0": sum(
            int(p == 1 and y == 0) for p, y in zip(predicted, labels, strict=True)
        ),
    }


def _excluded_counts(predictions: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "unresolved": sum(1 for item in predictions if item.get("label") is None),
        "missing_probability": sum(
            1 for item in predictions if item.get("label") is not None and item.get("probability") is None
        ),
        "leakage_failures": sum(
            1
            for item in predictions
            if item.get("label") is not None and not all(item.get("leakage_checks", {}).values())
        ),
        "missing_forecast_timestamp": sum(
            1
            for item in predictions
            if item.get("label") is not None and _parse_datetime(item.get("forecast_timestamp")) is None
        ),
    }


def _method_descriptions() -> list[dict[str, str]]:
    return [
        {
            "method": "empirical_base_rate",
            "description": "Laplace-smoothed positive rate from earlier walk-forward training rows.",
        },
        {
            "method": "llm_probability",
            "description": "Raw probability returned by the LLM at forecast time.",
        },
        {
            "method": "fixed_blf_posterior",
            "description": "Current hand-coded log-odds posterior combining priors and the LLM signal.",
        },
        {
            "method": "platt_llm_probability",
            "description": "Walk-forward Platt calibration trained on earlier raw LLM probabilities.",
        },
        {
            "method": "platt_fixed_blf_posterior",
            "description": "Walk-forward Platt calibration trained on earlier fixed posterior probabilities.",
        },
        {
            "method": "learned_logit_ensemble",
            "description": "Walk-forward logistic blend of raw LLM and fixed posterior logits.",
        },
    ]


def _candidate_recommendation(
    summary_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_method = {str(row["method"]): row for row in summary_rows}
    candidate_methods = [
        "learned_logit_ensemble",
        "platt_fixed_blf_posterior",
        "platt_llm_probability",
        "fixed_blf_posterior",
        "llm_probability",
    ]
    candidates = [by_method[method] for method in candidate_methods if method in by_method]
    if not candidates:
        return {
            "selected_shadow_candidate": None,
            "current_primary_candidate": None,
            "selection_metric": "unavailable",
            "rationale": ["No candidate methods were scored."],
            "guardrails": ["Collect more resolved rows before selecting a calibration candidate."],
        }

    selected = min(
        candidates,
        key=lambda row: (
            float(row["log_loss"]),
            float(row["brier_score"]),
            float(row["ece_10"]),
        ),
    )
    primary = by_method.get("fixed_blf_posterior") or by_method.get("llm_probability")
    llm = by_method.get("llm_probability")
    selected_fold_rows = [
        row for row in fold_rows if row.get("method") == selected.get("method")
    ]
    llm_fold_rows = {row.get("fold"): row for row in fold_rows if row.get("method") == "llm_probability"}
    unstable_folds = []
    for row in selected_fold_rows:
        llm_row = llm_fold_rows.get(row.get("fold"))
        if llm_row is None:
            continue
        if float(row["brier_score"]) > float(llm_row["brier_score"]) + 0.05:
            unstable_folds.append(row.get("fold"))

    rationale = [
        (
            f"{selected['method']} has the lowest walk-forward log loss "
            f"({float(selected['log_loss']):.4f}) among scored candidates."
        ),
        (
            f"Its Brier score is {float(selected['brier_score']):.4f} and ECE is "
            f"{float(selected['ece_10']):.4f}."
        ),
    ]
    if llm is not None:
        rationale.append(
            (
                f"Raw LLM baseline: log_loss={float(llm['log_loss']):.4f}, "
                f"brier={float(llm['brier_score']):.4f}, ece={float(llm['ece_10']):.4f}."
            )
        )
    guardrails = [
        "Use as a shadow calibration candidate first; do not replace stored raw LLM probabilities.",
        "Train/tune only on contract groups strictly earlier than the forecasted contract group.",
        "Keep recording raw LLM probability, fixed posterior, and selected calibrated probability side by side.",
    ]
    if unstable_folds:
        guardrails.append(
            (
                "Do not enable until at least 100 prior contract groups are available; "
                f"small-training folds were unstable: {unstable_folds}."
            )
        )
    return {
        "selected_shadow_candidate": selected["method"],
        "current_primary_candidate": None if primary is None else primary["method"],
        "selection_metric": "lowest walk-forward log loss; Brier and ECE used as secondary checks",
        "unstable_folds": unstable_folds,
        "rationale": rationale,
        "guardrails": guardrails,
    }


def _recommendation_markdown(recommendation: dict[str, Any]) -> str:
    lines = [
        f"Selected shadow candidate: `{recommendation.get('selected_shadow_candidate')}`",
        f"Current primary candidate: `{recommendation.get('current_primary_candidate')}`",
        f"Selection metric: {recommendation.get('selection_metric')}",
        "",
        "Rationale:",
    ]
    for item in recommendation.get("rationale", []):
        lines.append(f"- {item}")
    lines.extend(["", "Guardrails:"])
    for item in recommendation.get("guardrails", []):
        lines.append(f"- {item}")
    return "\n".join(lines)


def _rows_for_groups(
    groups: dict[str, list[dict[str, Any]]],
    group_keys: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for group_key in group_keys:
        rows.extend(groups[group_key])
    return rows


def _probability(value: Any) -> float:
    return clamp_probability(float(value), eps=0.01)


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("\n", " ")
