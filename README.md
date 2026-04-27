# murphy

BLF-inspired binary forecasting for ATM or near-the-money call options.

The first milestone is a reproducible DuckDB backtest:

1. Store option-chain snapshots and underlying bars.
2. Build strictly time-safe option examples.
3. Fit non-LLM probability baselines.
4. Add a Bayesian Linguistic Forecaster-style agent loop.
5. Aggregate, calibrate, and evaluate predictions.

This is research software, not trading advice or a live trading system.

## Live Cloud Shape

The intended production shape is:

- Cloud SQL Postgres as the live transactional source of truth.
- BigQuery for append-heavy analytics/event history.
- DuckDB for offline/local exports and model research.

Cloud env vars are listed in [.env.example](.env.example). Deployment notes live in
[docs/gcp_deployment_plan.md](docs/gcp_deployment_plan.md).

## Quick Smoke Test

```bash
python3 -m pytest
```

If dependencies are not installed yet, install in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```
# Murphys
