# Murphy

BLF-inspired binary forecasting for near-the-money call options.

This is research software. It is not trading advice, a trading strategy, or a
live trading system.

## Why This Exists

Murphy started as a one-week implementation experiment inspired by Kevin
Murphy's paper, "Agentic Forecasting using Sequential Bayesian Updating of
Linguistic Beliefs" (arXiv:2604.18576). The short version: Kevin Murphy, author
of the Probabilistic Machine Learning books, published the paper on Monday; I
saw it on Tuesday; Codex helped build this adapted implementation on Wednesday
and Thursday; by Friday it was running on GCP and making true blind predictions
on live option contracts.

The paper describes the Bayesian Linguistic Forecaster, or BLF: an agentic
forecasting loop that keeps a structured belief state, updates that belief with
tool observations, aggregates multiple trials, and calibrates the resulting
probabilities.

The original paper described the architecture but did not come with a complete
drop-in codebase for this options-specific use case. That used to be a serious
barrier: reimplementing a research system from prose meant weeks of scaffolding
before the interesting questions could even start. With Codex and modern LLMs,
that is no longer the hard part. The interesting work moves up a level: reading
the paper carefully, making the assumptions explicit, adapting the method to a
new domain, and building enough auditability to know whether the system is
actually forecasting or just leaking future information.

This repository is that adaptation for binary call-option forecasts.

## How This Deviates From The Paper

The paper evaluates BLF on forecasting benchmark questions. Murphy intentionally
deviates from that setting in two important ways:

- **True blind prediction.** Instead of only replaying historical questions, the
  live loop generates forecasts from market snapshots captured at prediction
  time. Labels are unknown when the LLM is called and are resolved later from
  stored underlying prices.
- **Cloud-native operation on GCP.** The system is designed to run as scheduled
  Google Cloud Run Jobs with Cloud SQL as the transactional source of truth,
  BigQuery as an analytics mirror, GCS for dataset artifacts, and Secret Manager
  for runtime credentials.

The core BLF ideas remain visible: structured prompts, strict information
cutoffs, persisted evidence, probability traces, market-implied priors,
calibration, and walk-forward evaluation.

## Forecasting Task

The live question template is:

```text
Will the price of $TICKER be greater than $ATM_CALL_OPTION_STRIKE_PRICE on $DATE_OF_EXPIRY?
```

The system treats each hourly market snapshot as a distinct forecast instance.
The same English question may appear more than once, but each instance has its
own option price, spot price, implied volatility, volume/open interest, cached
context, prompt, LLM response, and information cutoff.

Current default ticker set:

```text
AAPL MSFT NVDA AMD SPY QQQ
```

The live generator uses a short-term strike ladder:

- Expiries must be at least 12 hours away and no more than 14 days away.
- For each ticker and expiry, select up to 5 call strikes below spot.
- Select up to 5 call strikes at or above spot.
- Each selected strike becomes a binary question for that forecast timestamp.
- Production-style runs can cap both per-ticker and global question counts.

## Architecture

Local and cloud components share the same core package.

- **Collectors** fetch underlying bars and option-chain snapshots.
- **Question generation** turns option snapshots into binary events.
- **Predictors** call an OpenAI-compatible model or a deterministic dry-run
  predictor.
- **Repositories** persist the live/audit trail to DuckDB or Cloud SQL.
- **Evaluation commands** resolve labels, check leakage constraints, export
  datasets, and run walk-forward calibration reports.

Cloud deployment uses:

- Cloud Run Jobs for collection, prediction, resolution, reporting, and exports.
- Cloud Scheduler for trading-hour and post-close runs.
- Cloud SQL Postgres for live state.
- BigQuery for mirrored analytics tables.
- GCS for exported datasets and reports.
- Secret Manager for API keys and database passwords.

## No-Leakage Design

Murphy is built around a simple rule: every forecast must be explainable using
only information available at its `information_cutoff`.

- Every forecast stores an information cutoff.
- Prompts include only stored evidence at or before that cutoff.
- Market data, cached news context, prompt text, and LLM responses are persisted.
- Resolution requires underlying bars from the actual expiry date by default.
- Evaluation reports include leakage checks and cache hash references.
- Walk-forward reports split by contract group, not random rows, so repeated
  hourly snapshots for the same option contract do not cross train/test splits.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,agent,gcp,providers,training]"
```

For local development without cloud extras:

```bash
python -m pip install -e ".[dev,providers]"
```

## Configuration

Copy the example environment file and fill in local values:

```bash
cp .env.example .env
```

Never commit real credentials. `.env`, `.env.*`, `.env~`, `.gcloud/`, DuckDB
files, JSON/JSONL data exports, and generated reports under `data/` are ignored.

For GCP, use `deployment/env.gcp.example` as a template. Real values should live
in your shell, CI secrets, or Google Secret Manager.

Required runtime variables depend on the backend:

```bash
CLOUD_SQL_CONNECTION_NAME=your-gcp-project:us-west1:murphy-postgres
DB_NAME=murphy
DB_USER=murphy_app
DB_PASS=replace-me
PRIVATE_IP=false

BQ_PROJECT_ID=your-gcp-project
BQ_DATASET=murphy
BQ_LOCATION=US

OPENAI_API_KEY=replace-me
MARKET_DATA_API_KEY=replace-me
```

`MARKET_DATA_API_KEY` is reserved for providers that need one. The current
Yahoo/yfinance prototype does not use an official paid key.

## Useful Commands

Run tests:

```bash
python3 -m pytest
```

Initialize local DuckDB:

```bash
murphy init-db --db data/murphy.duckdb
```

Run a dry local live cycle:

```bash
murphy run-live-cycle \
  --provider yahoo \
  --backend duckdb \
  --tickers AAPL \
  --dry-run
```

Generate a live operational report:

```bash
murphy daily-status --backend cloud-sql --include-bigquery
```

Write an offline analysis report:

```bash
murphy offline-analysis \
  --duckdb data/murphy_offline.duckdb \
  --output data/offline_analysis_report.md \
  --test-fraction 0.2
```

Run a walk-forward calibration comparison:

```bash
murphy walk-forward-report \
  --backend cloud-sql \
  --n-folds 5 \
  --min-train-groups 20 \
  --output data/walk_forward_report.md
```

Score a one-off question with current live context:

```bash
murphy score-custom-question \
  --backend cloud-sql \
  --ticker AAPL \
  --strike 295 \
  --date 2026-06-15
```

Export resolved ScalarLM-style SFT rows:

```bash
murphy export-scalar-sft-dataset \
  --backend cloud-sql \
  --output /tmp/scalar_sft_resolved.jsonl \
  --gcs-uri gs://your-murphy-artifacts/scalar_sft/
```

## GCP Bootstrap

The deployment examples are intentionally generic. Set your own project and
region before running cloud commands:

```bash
export PROJECT_ID="your-gcp-project"
export REGION="us-west1"
export REPO="murphy"
export IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/murphy:latest"

PROJECT_ID="$PROJECT_ID" REGION="$REGION" bash deployment/bootstrap_gcp.sh
```

See [docs/gcp_deployment_plan.md](docs/gcp_deployment_plan.md) for the full
Cloud Run, Cloud Scheduler, Cloud SQL, BigQuery, and Secret Manager plan.

## Repository Hygiene

The current tree uses placeholders for credentials and project-specific cloud
resources. Before publishing a fork or pushing to a new remote:

- Run a secret scanner against both the working tree and git history.
- Keep real `.env` files and local cloud auth directories untracked.
- Rotate any API keys that were ever committed to a private remote.
- Consider rewriting history if private project IDs, bucket names, or resource
  names should not appear in public commit history.

## Docs

- Deployment details: [docs/gcp_deployment_plan.md](docs/gcp_deployment_plan.md)
- Market data notes: [docs/market_data_provider_options.md](docs/market_data_provider_options.md)
- Near-term implementation plan: [docs/next_implementation_plan.md](docs/next_implementation_plan.md)
- Public launch article draft: [docs/public_launch_article.md](docs/public_launch_article.md)
