# Next Implementation Plan

This phase makes the live forecasting system easier to operate while it waits
for the first real option expiries to resolve.

## 1. Operational Status And Warnings

Implemented first because it gives fast feedback from scheduled Cloud Run logs.

- Add `murphy daily-status`.
- Report latest option snapshot, underlying bar, prediction, resolution, and
  external-call cache timestamps.
- Report live question counts by status.
- Warn when snapshots are stale, no questions have been generated, predictions
  are pending, or due questions are unresolved.
- Optionally include BigQuery table freshness.

## 2. Persistent ScalarLM Datasets

The local ScalarLM export command now supports persistent upload:

- Export only resolved live predictions by default.
- Filter out leakage-check failures by default.
- Write JSONL locally.
- Optionally upload the JSONL to GCS using `--gcs-uri`.

The intended production target is:

```bash
murphy export-scalar-sft-dataset \
  --backend cloud-sql \
  --output /tmp/scalar_sft_resolved.jsonl \
  --gcs-uri gs://murphys-494519-murphy-artifacts/scalar_sft/
```

## 3. Near-Term Follow Ups

- Treat each hourly market snapshot as a distinct forecast instance. The plain
  English question can repeat for the same ticker/strike/expiry, but the option
  price, spot, IV, volume/open interest, cached web context, prompt, and LLM
  response belong to the forecast hour when they were captured.
- Add notification channels to the GCP log-based alert policies.
- Add a weekly scheduled ScalarLM dataset export after enough resolutions exist.
- Expand ticker coverage slowly once AAPL has a complete prediction-resolution
  cycle.
- Add historical/logistic baselines alongside the option-implied prior and
  BLF-style posterior.
