# Next Implementation Plan

This phase makes the live forecasting system easier to operate while it waits
for the first real option expiries to resolve.

Current live ticker set:

```text
AAPL MSFT NVDA AMD SPY QQQ
```

Current live question generation rule:

- Fetch expiries from 0 to 14 days out.
- For each ticker/expiry, generate questions for up to 5 call strikes below
  current spot and up to 5 call strikes at or above current spot.
- Cap production generation at 5 questions per ticker per run, with a global
  30-question cap across the six-ticker set.
- Keep each hourly market snapshot as a distinct forecast instance, even when
  the plain-English question repeats.

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
- Use the expanded liquid ticker set cautiously and monitor Yahoo/OpenAI cost.
- Add a trained logistic live prior once enough resolved labels exist; the
  guarded historical empirical prior is already wired into prompts and traces.
- Extend `murphy analysis-report` with plots and grouped train/test split
  exports for ScalarLM experiments.
