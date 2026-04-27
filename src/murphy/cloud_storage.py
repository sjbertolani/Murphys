from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse


def upload_file_to_gcs(local_path: str | Path, gcs_uri: str) -> str:
    """Upload a local file to a GCS URI and return the resolved object URI."""
    from google.cloud import storage

    resolved_uri = resolve_gcs_uri(gcs_uri, default_filename=Path(local_path).name)
    bucket_name, object_name = split_gcs_uri(resolved_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(str(local_path))
    return resolved_uri


def resolve_gcs_uri(gcs_uri: str, default_filename: str) -> str:
    """Resolve directory/template-style GCS URIs into a concrete object URI."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if "{timestamp}" in gcs_uri:
        return gcs_uri.format(timestamp=timestamp)
    if gcs_uri.endswith("/"):
        return f"{gcs_uri}{timestamp}_{default_filename}"
    return gcs_uri


def split_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    parsed = urlparse(gcs_uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("GCS URI must look like gs://bucket/path/to/object")
    return parsed.netloc, parsed.path.lstrip("/")
