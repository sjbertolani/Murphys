from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime
from statistics import mean

from murphy.training.datasets import grouped_prediction_splits


def build_analysis_report(
    repository,
    *,
    ticker: str | None = None,
    limit: int = 10000,
    include_unresolved: bool = True,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.0,
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
        "grouped_split": _grouped_split_summary(
            predictions,
            test_fraction=test_fraction,
            validation_fraction=validation_fraction,
        ),
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
    split = report["grouped_split"]
    lines.extend(
        [
            "",
            "## Grouped Dataset Split",
            "",
            f"Grouping rule: `{split['manifest']['grouping_rule']}`",
            f"Test fraction: {_format_value(split['test_fraction'])}",
            f"Validation fraction: {_format_value(split['validation_fraction'])}",
            "",
            "### Manifest",
            "",
            _table(_manifest_rows(split["manifest"])),
            "",
            "### Split Breakdown",
            "",
            _table(split["split_breakdown"]),
        ]
    )
    if split["manifest"].get("warnings"):
        lines.extend(["", "### Split Warnings", ""])
        for warning in split["manifest"]["warnings"]:
            lines.append(f"- {warning}")
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


def _grouped_split_summary(
    predictions: list[dict],
    *,
    test_fraction: float,
    validation_fraction: float,
) -> dict:
    eligible = [
        item
        for item in predictions
        if item.get("label") is not None and all(item.get("leakage_checks", {}).values())
    ]
    annotated, manifest = grouped_prediction_splits(
        eligible,
        test_fraction=test_fraction,
        validation_fraction=validation_fraction,
    )
    return {
        "test_fraction": test_fraction,
        "validation_fraction": validation_fraction,
        "eligible_rows": len(eligible),
        "excluded_rows": {
            "unresolved": sum(1 for item in predictions if item.get("label") is None),
            "leakage_failures": sum(
                1
                for item in predictions
                if item.get("label") is not None
                and not all(item.get("leakage_checks", {}).values())
            ),
        },
        "manifest": manifest,
        "split_breakdown": _split_breakdown(annotated, validation_fraction),
    }


def _split_breakdown(annotated: list[dict], validation_fraction: float) -> list[dict]:
    split_order = ["train"]
    if validation_fraction > 0:
        split_order.append("validation")
    split_order.append("test")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in annotated:
        grouped[str(item.get("dataset_split") or "train")].append(item)

    rows = []
    for split in split_order:
        items = grouped.get(split, [])
        groups = {item.get("contract_group_key") for item in items}
        tickers = {str(item.get("symbol") or "UNKNOWN") for item in items}
        labels = [item.get("label") for item in items if item.get("label") is not None]
        rows.append(
            {
                "split": split,
                "n_rows": len(items),
                "n_contract_groups": len(groups),
                "n_tickers": len(tickers),
                "positive_rate": _average(labels),
                "avg_probability": _average(item.get("probability") for item in items),
                "avg_posterior_probability": _average(
                    item.get("posterior_probability") for item in items
                ),
            }
        )
    return rows


def _manifest_rows(manifest: dict) -> list[dict]:
    row_counts = manifest.get("row_counts", {})
    group_counts = manifest.get("contract_group_counts", {})
    splits = sorted(set(row_counts) | set(group_counts))
    rows = [
        {
            "metric": "total",
            "rows": manifest.get("n_rows", 0),
            "contract_groups": manifest.get("n_contract_groups", 0),
        }
    ]
    for split in splits:
        rows.append(
            {
                "metric": split,
                "rows": row_counts.get(split, 0),
                "contract_groups": group_counts.get(split, 0),
            }
        )
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
