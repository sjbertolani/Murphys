# murphy

BLF-inspired binary forecasting for ATM or near-the-money call options.

This is research software, not trading advice or a live trading system.

## Current State

The system now runs a live, no-leakage forecasting loop on GCP for hourly call-option predictions. It uses Yahoo Finance as the first market data provider, OpenAI for LLM probabilities, Cloud SQL as the transactional source of truth, BigQuery as an analytics mirror, GCS for dataset artifacts, and DuckDB for offline/local analysis.

The live question template is:

```text
Will the price of $TICKER be greater than $ATM_CALL_OPTION_STRIKE_PRICE on $DATE_OF_EXPIRY?
```

Important dataset rule: the same English question may repeat, but each hourly market snapshot is a distinct forecast instance. The identity is effectively ticker/option/strike/expiry plus forecast-hour information cutoff. This preserves changing option prices, spot, IV, volume/open interest, cached context, prompt, and LLM response over time.

Current live ticker set:

```text
AAPL MSFT NVDA AMD SPY QQQ
```

## Live GCP Deployment

Project:

```text
murphys-494519
```

Region:

```text
us-west1
```

Local `gcloud` note:

```bash
/usr/local/share/google-cloud-sdk/bin/gcloud auth login
/usr/local/share/google-cloud-sdk/bin/gcloud config set project murphys-494519
/usr/local/share/google-cloud-sdk/bin/gcloud config set run/region us-west1
```

On this machine the `gcloud` binary may not be on `PATH`; use the full path
`/usr/local/share/google-cloud-sdk/bin/gcloud`. The local browser auth flow
worked reliably. The `--no-launch-browser` remote verification-code flow hit
Google's "Access blocked: This app's request is invalid" error.

Main assets:

- Cloud SQL Postgres instance: `murphy-postgres`
- Artifact Registry image: `us-west1-docker.pkg.dev/murphys-494519/murphy/murphy:latest`
- GCS artifact bucket: `gs://murphys-494519-murphy-artifacts`
- BigQuery dataset: `murphy`
- Secrets: `openai-api-key`, `murphy-db-pass`

Enabled Cloud Scheduler jobs:

| Job | Schedule PT | Purpose |
| --- | --- | --- |
| `murphy-live-cycle-trading-hourly` | `30 6-12 * * 1-5` | Collect market data, generate hourly forecast instances, call OpenAI |
| `murphy-live-cycle-trading-close` | `0 13 * * 1-5` | Final close-time live cycle |
| `murphy-resolve-and-report-post-close` | `15 13 * * 1-5` | Collect expiry-date bars, resolve due predictions, print report |
| `murphy-bigquery-mirror-post-close` | `30 13 * * 1-5` | Mirror Cloud SQL live/audit tables to BigQuery |
| `murphy-daily-status-post-close` | `45 13 * * 1-5` | Print operational status and warnings |
| `murphy-scalar-sft-export-weekly` | `0 9 * * 6` | Export resolved ScalarLM SFT JSONL to GCS |

## No-Leakage Design

- Every forecast gets an `information_cutoff`.
- Prompts include only stored evidence at or before that cutoff.
- Market data, cached web/news context, and LLM calls are recorded in `external_call_cache`.
- Cache records include request/response payloads, timestamps, and response hashes.
- Resolution requires an underlying bar from the actual expiry date by default, preventing stale prior-close labeling.
- Evaluation reports include leakage checks and cache hash references.
- Prediction traces include market-implied prior, historical empirical prior when enough prior labels exist, raw LLM probability, and a BLF-style log-odds posterior update.

## Data Stores

Cloud SQL tables include:

- `underlying_bars`
- `option_chain_snapshots`
- `option_examples`
- `live_questions`
- `llm_responses`
- `live_resolutions`
- `daily_runs`
- `external_call_cache`

BigQuery mirrors the live/audit tables for analytics. DuckDB exports are available for offline work.

## Useful Commands

Run tests:

```bash
python3 -m pytest
```

Initialize local DuckDB:

```bash
murphy init-db --db data/murphy.duckdb
```

Run a local/live cycle:

```bash
murphy run-live-cycle \
  --provider yahoo \
  --backend duckdb \
  --tickers AAPL \
  --dry-run
```

Print an operational report:

```bash
murphy daily-status --backend cloud-sql --include-bigquery
```

Print an evaluation report:

```bash
murphy evaluation-report --backend cloud-sql --ticker AAPL
```

Write an offline analysis report:

```bash
murphy export-cloud-sql-to-duckdb --duckdb data/murphy_offline.duckdb
murphy analysis-report \
  --backend duckdb \
  --db data/murphy_offline.duckdb \
  --output data/offline_analysis_report.md
```

Mirror Cloud SQL to BigQuery:

```bash
murphy mirror-cloud-sql-to-bigquery
```

Export Cloud SQL to DuckDB:

```bash
murphy export-cloud-sql-to-duckdb --duckdb data/murphy_offline.duckdb
```

Export resolved ScalarLM SFT rows:

```bash
murphy export-scalar-sft-dataset \
  --backend cloud-sql \
  --output /tmp/scalar_sft_resolved.jsonl \
  --gcs-uri gs://murphys-494519-murphy-artifacts/scalar_sft/
```

## Verified So Far

- Full test suite passes: `38 passed`.
- Cloud Run `murphy-daily-status` executed successfully.
- Cloud Run `murphy-scalar-sft-export` executed successfully and uploaded an expected empty JSONL while there are no resolved labels yet.
- BigQuery mirror has been verified with live row counts.
- The current deployed generator creates at most one forecast per ticker/strike/expiry per forecast hour.
- Expanded scheduled run for `AAPL MSFT NVDA AMD SPY QQQ` completed successfully with 946 option snapshots, 30 cached news items, 30 questions, and 30 predictions.
- Cloud Monitoring alert policies exist for failed Cloud Run jobs and non-empty `daily-status` warnings.

## Docs

- Deployment details: [docs/gcp_deployment_plan.md](docs/gcp_deployment_plan.md)
- Market data notes: [docs/market_data_provider_options.md](docs/market_data_provider_options.md)
- Near-term implementation plan: [docs/next_implementation_plan.md](docs/next_implementation_plan.md)

## Next Work

Near-term priorities:

- Add notification channels to the GCP alert policies.
- Add a trained logistic live prior once enough resolved labels exist; the guarded historical empirical prior is already wired in.
- Extend `murphy analysis-report` with plots and grouped train/test split exports.
- Add training/evaluation splits that avoid leakage across correlated hourly rows from the same option contract.
