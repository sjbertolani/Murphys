from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any


def build_operational_status_report(
    repository_status: dict[str, Any],
    bigquery_status: dict[str, dict[str, Any]] | None = None,
    max_snapshot_age_hours: float = 6.0,
    max_bigquery_age_hours: float = 24.0,
) -> dict[str, Any]:
    generated_at = datetime.now(UTC)
    warnings = operational_warnings(
        repository_status,
        bigquery_status=bigquery_status,
        now=generated_at,
        max_snapshot_age_hours=max_snapshot_age_hours,
        max_bigquery_age_hours=max_bigquery_age_hours,
    )
    return {
        "generated_at": generated_at,
        "repository": repository_status,
        "bigquery": bigquery_status,
        "warnings": warnings,
    }


def operational_warnings(
    status: dict[str, Any],
    bigquery_status: dict[str, dict[str, Any]] | None = None,
    now: datetime | None = None,
    max_snapshot_age_hours: float = 6.0,
    max_bigquery_age_hours: float = 24.0,
) -> list[dict[str, str]]:
    now = _ensure_aware(now or datetime.now(UTC))
    warnings: list[dict[str, str]] = []

    snapshot_count = int(status.get("option_snapshots", {}).get("count") or 0)
    latest_snapshot = _parse_datetime(status.get("option_snapshots", {}).get("latest"))
    if snapshot_count == 0:
        warnings.append(_warning("no_option_snapshots", "No option snapshots are stored."))
    elif latest_snapshot is None:
        warnings.append(_warning("missing_snapshot_timestamp", "Option snapshots have no latest timestamp."))
    elif now - latest_snapshot > timedelta(hours=max_snapshot_age_hours):
        warnings.append(
            _warning(
                "stale_option_snapshots",
                f"Latest option snapshot is older than {max_snapshot_age_hours:g} hours.",
            )
        )

    live_status_counts = status.get("live_questions", {}).get("status_counts", {})
    total_questions = int(status.get("live_questions", {}).get("total") or 0)
    if snapshot_count > 0 and total_questions == 0:
        warnings.append(
            _warning(
                "no_live_questions_generated",
                "Snapshots exist but no live questions have been generated.",
            )
        )
    pending = int(live_status_counts.get("pending_prediction") or 0)
    if pending:
        warnings.append(
            _warning("pending_predictions", f"{pending} live questions still need predictions.")
        )

    due_unresolved = int(status.get("live_questions", {}).get("due_unresolved") or 0)
    if due_unresolved:
        warnings.append(
            _warning(
                "due_unresolved_questions",
                f"{due_unresolved} live questions are past resolution_due without labels.",
            )
        )

    if bigquery_status is not None:
        for table, table_status in bigquery_status.items():
            modified = _parse_datetime(table_status.get("modified"))
            if modified is None:
                warnings.append(
                    _warning(
                        "missing_bigquery_modified",
                        f"BigQuery table {table} has no modified timestamp.",
                    )
                )
            elif now - modified > timedelta(hours=max_bigquery_age_hours):
                warnings.append(
                    _warning(
                        "stale_bigquery_mirror",
                        f"BigQuery table {table} is older than {max_bigquery_age_hours:g} hours.",
                    )
                )

    return warnings


def _warning(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _parse_datetime(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_aware(value)
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return _ensure_aware(datetime.fromisoformat(normalized))
        except ValueError:
            return None
    return None


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
