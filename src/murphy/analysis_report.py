from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from statistics import mean


def build_analysis_report(
    repository,
    *,
    ticker: str | None = None,
    limit: int = 10000,
    include_unresolved: bool = True,
    generated_at: datetime | None = None,
) -> dict:
    """Build an offline analysis summary from cached live forecast rows."""
    evaluation = repository.evaluation_report(
        ticker=ticker,
        limit=limit,
        include_unresolved=include_unresolved,
    )
    predictions = evaluation["predictions"]
    resolved = [item for item in predictions if item.get("label") is not None]
    leakage_failures = [
        item for item in predictions if not all(item.get("leakage_checks", {}).values())
    ]
    return {
        "generated_at": generated_at or datetime.now(UTC),
        "ticker": None if ticker is None else ticker.upper(),
        "limit": limit,
        "include_unresolved": include_unresolved,
        "summary": evaluation["summary"],
        "ticker_breakdown": _ticker_breakdown(predictions),
        "dte_breakdown": _dte_breakdown(predictions),
        "calibration": {
            "llm_probability": _calibration_bins(resolved, "probability"),
            "posterior_probability": _calibration_bins(resolved, "posterior_probability"),
        },
        "repeated_questions": _repeated_questions(predictions),
        "leakage_failures": leakage_failures[:25],
        "dataset_notes": _dataset_notes(evaluation["summary"], predictions, leakage_failures),
    }


def render_markdown_report(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# Murphy Offline Analysis Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Ticker filter: {report['ticker'] or 'all'}",
        f"Prediction row limit: {report['limit']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in [
        "n_predictions",
        "n_resolved",
        "n_unresolved",
        "n_scorable",
        "n_posterior_scorable",
        "n_leakage_check_failures",
        "brier_score",
        "posterior_brier_score",
        "accuracy_at_0_5",
        "posterior_accuracy_at_0_5",
    ]:
        lines.append(f"| {key} | {_format_value(summary.get(key))} |")

    lines.extend(["", "## Ticker Breakdown", "", _table(report["ticker_breakdown"])])
    lines.extend(["", "## DTE Breakdown", "", _table(report["dte_breakdown"])])
    lines.extend(
        [
            "",
            "## Calibration Bins",
            "",
            "### LLM Probability",
            "",
            _table(report["calibration"]["llm_probability"]),
            "",
            "### Posterior Probability",
            "",
            _table(report["calibration"]["posterior_probability"]),
        ]
    )
    lines.extend(["", "## Repeated Questions", "", _table(report["repeated_questions"][:20])])
    lines.extend(["", "## Dataset Notes", ""])
    for note in report["dataset_notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _ticker_breakdown(predictions: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in predictions:
        groups[str(item.get("symbol") or "UNKNOWN")].append(item)
    rows = []
    for symbol, items in sorted(groups.items()):
        rows.append({"ticker": symbol, **_group_metrics(items)})
    return rows


def _dte_breakdown(predictions: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in predictions:
        groups[_dte_bucket(item.get("dte"))].append(item)
    rows = []
    for bucket in ["0-5", "5-10", "10-20", "20+", "unknown"]:
        if bucket in groups:
            rows.append({"dte_bucket": bucket, **_group_metrics(groups[bucket])})
    return rows


def _group_metrics(items: list[dict]) -> dict:
    resolved = [item for item in items if item.get("label") is not None]
    return {
        "n_predictions": len(items),
        "n_resolved": len(resolved),
        "avg_probability": _average(item.get("probability") for item in items),
        "avg_posterior_probability": _average(item.get("posterior_probability") for item in items),
        "brier_score": _average(item.get("brier") for item in resolved),
        "posterior_brier_score": _average(item.get("posterior_brier") for item in resolved),
        "leakage_failures": sum(
            1 for item in items if not all(item.get("leakage_checks", {}).values())
        ),
    }


def _calibration_bins(
    resolved_predictions: list[dict],
    probability_key: str,
    bins: int = 10,
) -> list[dict]:
    rows = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        in_bin = [
            item
            for item in resolved_predictions
            if item.get(probability_key) is not None
            and lower <= float(item[probability_key]) < upper
        ]
        if index == bins - 1:
            in_bin = [
                item
                for item in resolved_predictions
                if item.get(probability_key) is not None
                and lower <= float(item[probability_key]) <= upper
            ]
        rows.append(
            {
                "bin": f"{lower:.1f}-{upper:.1f}",
                "n": len(in_bin),
                "mean_probability": _average(item.get(probability_key) for item in in_bin),
                "empirical_rate": _average(item.get("label") for item in in_bin),
            }
        )
    return rows


def _repeated_questions(predictions: list[dict]) -> list[dict]:
    counts = Counter(str(item.get("question_text")) for item in predictions)
    rows = []
    for question_text, count in counts.most_common():
        if count <= 1:
            continue
        rows.append({"count": count, "question_text": question_text})
    return rows


def _dataset_notes(summary: dict, predictions: list[dict], leakage_failures: list[dict]) -> list[str]:
    notes = []
    if summary.get("n_resolved", 0) == 0:
        notes.append("No resolved labels yet; scoring and calibration will remain empty.")
    if leakage_failures:
        notes.append(f"{len(leakage_failures)} rows failed at least one leakage check.")
    if not predictions:
        notes.append("No prediction rows matched this report filter.")
    if summary.get("n_unresolved", 0):
        notes.append("Unresolved rows are included for coverage but excluded from scoring.")
    return notes or ["No dataset warnings for this report."]


def _dte_bucket(value) -> str:
    if value is None:
        return "unknown"
    value = float(value)
    if value < 5:
        return "0-5"
    if value < 10:
        return "5-10"
    if value < 20:
        return "10-20"
    return "20+"


def _average(values) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return None if not cleaned else mean(cleaned)


def _table(rows: list[dict]) -> str:
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


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("\n", " ")
