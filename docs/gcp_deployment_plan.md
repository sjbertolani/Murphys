# GCP Deployment Plan

Goal: run a live forecasting loop that records option chains, generates questions such as:

> Will the price of $TICKER be greater than $ATM_CALL_OPTION_STRIKE_PRICE on $DATE_OF_EXPIRY?

Then it calls an LLM using only information available at forecast time, persists the response, waits until expiry or the resolution window, and resolves the label from stored underlying prices.

## Architecture

Use Cloud Run Jobs plus Cloud Scheduler for execution, Cloud SQL Postgres for transactional live state, BigQuery for analytics/event history, and DuckDB only for offline/local analysis exports.

Production roles:

- Cloud Run Jobs: run collectors, question generation, prediction, and resolution.
- Cloud Scheduler: trigger jobs multiple times per trading day.
- Cloud SQL for PostgreSQL: source of truth for live state and idempotent job operations.
- BigQuery: append-heavy analytics store for snapshots, responses, resolutions, and later dashboards.
- DuckDB: offline export target for local modeling and fast ad hoc analysis.
- Secret Manager: API keys and DB password.
- Artifact Registry: container images.
- Cloud Logging/Error Reporting: job observability.

## Why Not DuckDB In Cloud Run

Cloud Run's container filesystem is ephemeral. DuckDB is excellent for offline analytics, but it is not the right system of record for scheduled Cloud Run jobs with multiple invocations. Keep DuckDB as a reproducible export format:

```bash
murphy export-cloud-sql-to-duckdb --duckdb data/murphy_offline.duckdb
```

## Data Placement

Cloud SQL Postgres should hold:

- `underlying_bars`
- `option_chain_snapshots`
- `option_examples`
- `live_questions`
- `llm_responses`
- `live_resolutions`
- `daily_runs`

BigQuery should hold append-oriented copies:

- `option_chain_snapshots`
- `llm_responses`
- `live_resolutions`

DuckDB should hold:

- Cloud SQL exports for offline research.
- Feature matrices and model-evaluation snapshots.
- Experimental tables that do not need to serve live jobs.

## Runtime Flow

The daily container should eventually run four logical commands:

1. `collect-snapshots`
   - Pull current option chains and underlying bars from provider.
   - Write to Cloud SQL.
   - Append high-volume raw snapshots to BigQuery.

2. `generate-live-questions`
   - Select ATM/near-money calls from latest snapshot.
   - Insert unresolved examples into `option_examples`.
   - Insert question text into `live_questions`.

3. `predict-pending`
   - Build an LLM prompt from stored evidence with a hard cutoff timestamp.
   - Call the selected LLM.
   - Insert into `llm_responses`.
   - Append response metadata to BigQuery.

4. `resolve-live-questions`
   - Find due questions.
   - Use stored underlying bars at or before expiry.
   - Insert into `live_resolutions`.
   - Update labels in `option_examples`.
   - Append resolution event to BigQuery.

The code currently implements Yahoo/yfinance snapshot collection for command 1, local DuckDB
and Cloud SQL-backed versions of commands 1-4 through `run-live-cycle`.

## Environment Variables

Cloud SQL:

```bash
CLOUD_SQL_CONNECTION_NAME="project:region:instance"
DB_NAME="murphy"
DB_USER="murphy_app"
DB_PASS="..."
PRIVATE_IP="false"
```

BigQuery:

```bash
BQ_PROJECT_ID="your-project"
BQ_DATASET="murphy"
BQ_LOCATION="US"
```

Provider/LLM secrets should live in Secret Manager and be mounted as env vars:

```bash
OPENAI_API_KEY="..."
MARKET_DATA_API_KEY="..."
```

## Local Cloud Initialization

After authenticating with Application Default Credentials:

```bash
gcloud auth application-default login
murphy init-cloud --cloud-sql --bigquery
```

The code uses the Cloud SQL Python Connector for Postgres and the BigQuery Python client. Google documents Cloud SQL connectors as providing encryption and IAM-based authorization, and notes Postgres support via `pg8000` or `asyncpg`.

## Build and Push

Set variables:

```bash
export PROJECT_ID="your-gcp-project"
export REGION="us-west1"
export REPO="murphy"
export IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/murphy:latest"
```

The generic bootstrap script enables APIs, creates service accounts, grants
baseline IAM roles, and creates the Artifact Registry Docker repository. It
intentionally does not create Cloud SQL or Cloud Run jobs yet, because those are
the first cost-bearing steps.

```bash
PROJECT_ID="$PROJECT_ID" bash deployment/bootstrap_gcp.sh
```

Enable APIs:

```bash
gcloud services enable \
  artifactregistry.googleapis.com \
  bigquery.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  sqladmin.googleapis.com
```

Create Artifact Registry:

```bash
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker \
  --location="$REGION"
```

Build:

```bash
gcloud builds submit --tag "$IMAGE"
```

## Cloud SQL

Create a Postgres instance:

```bash
gcloud sql instances create murphy-postgres \
  --database-version=POSTGRES_16 \
  --region="$REGION" \
  --tier=db-f1-micro \
  --storage-size=10GB
```

This is intentionally small for a prototype. If Cloud SQL rejects `db-f1-micro` for the selected
Postgres edition in this project, use the smallest available shared-core or 1-vCPU tier shown by:

```bash
gcloud sql tiers list
```

Create DB/user:

```bash
gcloud sql databases create murphy --instance=murphy-postgres
gcloud sql users create murphy_app --instance=murphy-postgres --password="$DB_PASS"
```

Grant the Cloud Run service account permission to connect:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:murphy-runner@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/cloudsql.client"
```

## BigQuery

The app can initialize the dataset/tables:

```bash
murphy init-cloud --bigquery
```

Or create the dataset manually:

```bash
bq --location=US mk --dataset "$PROJECT_ID:murphy"
```

Grant the runtime service account:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:murphy-runner@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"
```

## Cloud Run Jobs

Create a job for cloud initialization:

```bash
gcloud run jobs create murphy-init-cloud \
  --image "$IMAGE" \
  --region "$REGION" \
  --service-account "murphy-runner@$PROJECT_ID.iam.gserviceaccount.com" \
  --command murphy \
  --args init-cloud,--cloud-sql,--bigquery \
  --set-env-vars CLOUD_SQL_CONNECTION_NAME="$PROJECT_ID:$REGION:murphy-postgres",DB_NAME=murphy,DB_USER=murphy_app,BQ_PROJECT_ID="$PROJECT_ID",BQ_DATASET=murphy
```

Create a job for hourly Yahoo snapshot collection:

```bash
gcloud run jobs create murphy-collect-yahoo \
  --image "$IMAGE" \
  --region "$REGION" \
  --service-account "murphy-runner@$PROJECT_ID.iam.gserviceaccount.com" \
  --command murphy \
  --args collect-snapshots,--provider,yahoo,--backend,cloud-sql,--tickers,AAPL,MSFT,NVDA,AMD,SPY,QQQ,--min-dte,0.5,--max-dte,14,--lookback-days,10 \
  --set-env-vars CLOUD_SQL_CONNECTION_NAME="$PROJECT_ID:$REGION:murphy-postgres",DB_NAME=murphy,DB_USER=murphy_app
```

Mount `DB_PASS` from Secret Manager before running this in production.

Create the full live-cycle job:

```bash
gcloud run jobs create murphy-live-cycle \
  --image "$IMAGE" \
  --region "$REGION" \
  --service-account "murphy-runner@$PROJECT_ID.iam.gserviceaccount.com" \
  --command murphy \
  --args run-live-cycle,--provider,yahoo,--backend,cloud-sql,--tickers,AAPL,MSFT,NVDA,AMD,SPY,QQQ,--min-dte,0.5,--max-dte,14,--lookback-days,10,--max-questions,30,--strike-window-size,5,--max-questions-per-ticker,5,--model,gpt-4.1-mini \
  --set-env-vars CLOUD_SQL_CONNECTION_NAME="$PROJECT_ID:$REGION:murphy-postgres",DB_NAME=murphy,DB_USER=murphy_app
```

For a smoke test without OpenAI calls:

```bash
gcloud run jobs execute murphy-live-cycle \
  --region "$REGION" \
  --args run-live-cycle,--provider,yahoo,--backend,cloud-sql,--tickers,AAPL,MSFT,NVDA,AMD,SPY,QQQ,--dry-run
```

Mount `DB_PASS` and `OPENAI_API_KEY` from Secret Manager before running real predictions.

## Scheduler

Run multiple times per trading day, Pacific time:

```bash
gcloud scheduler jobs create http murphy-generate-questions-open \
  --location "$REGION" \
  --schedule "35 6 * * 1-5" \
  --time-zone "America/Los_Angeles" \
  --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/murphy-generate-questions:run" \
  --http-method POST \
  --oauth-service-account-email "scheduler-invoker@$PROJECT_ID.iam.gserviceaccount.com"
```

Suggested first schedule:

- 06:35 PT: after market open, first liquid snapshot.
- Hourly during market hours: collect Yahoo snapshots.
- 09:30 PT: mid-session question generation/prediction.
- 12:30 PT: afternoon question generation/prediction.
- 14:15 PT and later: resolution and cleanup after the system's 21:00 UTC
  option-expiry due timestamp.

## LLM Cutoff Policy

For live forecasts:

- Generate at most one forecast per ticker/strike/expiry per forecast hour.
  Later hourly snapshots for the same option are intentionally new forecast
  instances because pricing and external context may have changed.
- `information_cutoff` is the snapshot timestamp.
- The prompt includes only stored option/price evidence whose timestamps are <= cutoff.
- If web/news search is used, it must run at forecast time and raw observations must be persisted.
- Do not use web search for historical replays unless you have a trustworthy historical archive.

## Immediate Next Build Step

Next production hardening steps:

- Mount `DB_PASS` and `OPENAI_API_KEY` from Secret Manager in Cloud Run jobs.
- Add BigQuery append calls to mirror snapshots/responses/resolutions.
- Add retry/backoff and timeout policy around Yahoo/yfinance and OpenAI.
- Add scoring/reporting jobs once resolved labels accumulate.
