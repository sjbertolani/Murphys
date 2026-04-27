from __future__ import annotations

import pytest

from murphy.cloud_storage import resolve_gcs_uri, split_gcs_uri


def test_split_gcs_uri() -> None:
    assert split_gcs_uri("gs://bucket/path/to/file.jsonl") == ("bucket", "path/to/file.jsonl")


def test_split_gcs_uri_rejects_directory_uri() -> None:
    with pytest.raises(ValueError, match="GCS URI"):
        split_gcs_uri("gs://bucket/")


def test_resolve_gcs_uri_directory_adds_timestamp_and_filename() -> None:
    resolved = resolve_gcs_uri("gs://bucket/scalar/", default_filename="rows.jsonl")

    assert resolved.startswith("gs://bucket/scalar/")
    assert resolved.endswith("_rows.jsonl")


def test_resolve_gcs_uri_template() -> None:
    resolved = resolve_gcs_uri(
        "gs://bucket/scalar/scalar_{timestamp}.jsonl",
        default_filename="ignored.jsonl",
    )

    assert resolved.startswith("gs://bucket/scalar/scalar_")
    assert resolved.endswith(".jsonl")
