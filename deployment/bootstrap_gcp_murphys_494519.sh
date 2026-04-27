#!/usr/bin/env bash
set -euo pipefail

# Project-specific bootstrap for Murphy live forecasting infrastructure.
# This script enables APIs and creates IAM/service-account scaffolding.
# It does not create Cloud SQL or Cloud Run jobs yet, because those can incur cost.

PROJECT_ID="${PROJECT_ID:-murphys-494519}"
REGION="${REGION:-us-west1}"
REPO="${REPO:-murphy}"
RUNTIME_SA="murphy-runner@${PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_SA="scheduler-invoker@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Using project: ${PROJECT_ID}"
echo "Using region:  ${REGION}"

gcloud config set project "${PROJECT_ID}"

gcloud services enable \
  artifactregistry.googleapis.com \
  bigquery.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com \
  iam.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  sqladmin.googleapis.com

if ! gcloud iam service-accounts describe "${RUNTIME_SA}" >/dev/null 2>&1; then
  gcloud iam service-accounts create murphy-runner \
    --display-name="Murphy Cloud Run Jobs"
fi

if ! gcloud iam service-accounts describe "${SCHEDULER_SA}" >/dev/null 2>&1; then
  gcloud iam service-accounts create scheduler-invoker \
    --display-name="Murphy Scheduler Invoker"
fi

for role in \
  roles/cloudsql.client \
  roles/bigquery.dataEditor \
  roles/secretmanager.secretAccessor \
  roles/artifactregistry.reader
do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role="${role}" \
    --quiet
done

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SCHEDULER_SA}" \
  --role="roles/run.invoker" \
  --quiet

if ! gcloud artifacts repositories describe "${REPO}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Murphy forecasting containers"
fi

echo
echo "Bootstrap complete."
echo "Next cost-bearing steps:"
echo "  1. Create Cloud SQL Postgres instance."
echo "  2. Create secrets for DB_PASS, LLM key, and market data key."
echo "  3. Build and deploy Cloud Run jobs."
