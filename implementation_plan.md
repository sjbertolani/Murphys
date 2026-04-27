# Python Implementation Plan

Goal: build a Python version of BLF adapted to binary prediction on ATM or near-the-money call options, using DuckDB for storage and leaving a clean path to ScalarLM post-training.

## Phase 0: Define the event

Start with one unambiguous label:

`label = 1 if underlying_close_at_expiration > strike else 0`

Scope constraints for v1:

- Calls only.
- ATM or near-the-money only, e.g. absolute moneyness `abs(S/K - 1) <= 0.03`.
- Liquid underlyings only.
- Forecast timestamp fixed, e.g. market close on trade date.
- Expiration horizon bucketed, e.g. 7-45 DTE.
- Research/backtesting only, not live trading.

## Phase 1: Project scaffold

Create a package like:

```text
murphy/
  pyproject.toml
  src/murphy/
    __init__.py
    db.py
    schemas.py
    options_data.py
    features.py
    priors.py
    agent/
      belief.py
      prompts.py
      runner.py
      tools.py
      aggregation.py
      calibration.py
    training/
      datasets.py
      scalarlm_client.py
      rewards.py
    eval.py
  tests/
```

Core dependencies:

```text
duckdb
polars
pyarrow
pydantic
numpy
scipy
scikit-learn
statsmodels
litellm or openai
```

Optional later:

```text
dynamax
rebayes
GPy or gpflow
scalarlm
```

## Phase 2: DuckDB schema

Tables:

- `underlying_bars`: symbol, timestamp, open, high, low, close, volume, adjusted fields.
- `option_chain_snapshots`: symbol, option_symbol, quote_timestamp, expiration, strike, call_put, bid, ask, mid, iv, delta, gamma, theta, vega, volume, open_interest.
- `option_examples`: example_id, symbol, option_symbol, forecast_timestamp, expiration, strike, spot, dte, moneyness, label, resolution_timestamp.
- `features`: example_id, feature_name, feature_value, feature_timestamp.
- `agent_trials`: trial_id, example_id, seed, model, started_at, raw_probability, status.
- `agent_steps`: trial_id, step_index, action_type, action_json, observation_ref, belief_json, probability.
- `forecasts`: example_id, method, raw_probability, aggregate_probability, calibrated_probability, created_at.
- `eval_results`: run_id, split, method, brier_score, log_loss, auc, calibration_error.

Use strict as-of joins:

```sql
WHERE feature_timestamp <= forecast_timestamp
```

Never compute features from data after the forecast timestamp.

## Phase 3: Non-LLM baselines first

Implement these before the full agent:

1. Constant baseline
   - Predict base rate by DTE/moneyness bucket.

2. Risk-neutral prior
   - Estimate probability from option-implied delta or Black-Scholes d2.
   - Store as `market_prior`.

3. Logistic regression / calibrated gradient boosting
   - Features: moneyness, DTE, IV, bid-ask spread, volume, open interest, realized volatility, recent returns, trend, market regime.
   - Fit with time-series splits.

4. Calibration
   - Platt scaling and isotonic calibration.
   - Grouped/hierarchical variant by symbol, sector, or DTE bucket.

This gives a sanity floor and avoids making the LLM responsible for basic numerical inference.

## Phase 4: BLF-style agent

Use the paper's loop:

1. Initialize belief: probability from `market_prior` or empirical bucket prior.
2. At each step, LLM emits:
   - action
   - updated belief JSON
3. Tool executes.
4. Store action, observation, and belief in DuckDB.
5. Stop on submit or after max steps.

Initial tools:

- `fetch_option_snapshot(example_id)`
- `fetch_underlying_history(symbol, forecast_timestamp, lookback_days)`
- `compute_volatility_features(example_id)`
- `compute_black_scholes_prior(example_id)`
- `fetch_event_calendar(symbol, forecast_timestamp)`
- `submit(probability, reasoning)`

Later tools:

- news/search with date filtering
- earnings transcript search
- macro regime lookup
- GP/state-space volatility model output
- `rebayes` online calibration state

Belief schema:

```json
{
  "probability": 0.52,
  "confidence": "low",
  "evidence_for": [],
  "evidence_against": [],
  "open_questions": [],
  "update_reasoning": ""
}
```

## Phase 5: Aggregation and calibration

For each example, run K independent trials, default K=5.

Aggregation:

- Convert probabilities to logits.
- Average logits.
- Apply sigmoid.
- Optionally tune shrinkage toward 0.5 using validation data.

Calibration:

- Fit Platt scaling on validation/backtest data.
- Add grouped intercept offsets where data is sufficient:
  - symbol
  - sector
  - DTE bucket
  - moneyness bucket
  - market regime

Metrics:

- Brier score
- log loss
- calibration curve / ECE
- ROC AUC as secondary
- economic backtest only after probability calibration is stable

## Phase 6: ScalarLM post-training path

Keep the agent model interface OpenAI-compatible so ScalarLM can replace hosted APIs.

ScalarLM use cases:

1. SFT classifier model
   - Input: structured option prompt with features, prior, and date-limited evidence.
   - Output: strict JSON with probability, confidence, and short rationale.

2. Belief-update model
   - Input: prior belief + one tool observation.
   - Output: updated belief JSON.
   - This directly trains the BLF component instead of a monolithic forecaster.

3. Outcome-based post-training
   - Use resolved examples.
   - Reward/logging based on proper scoring rules: negative log loss or Brier reward.
   - Requires custom ScalarLM `ml/` training loop if moving beyond simple input/output SFT.

Suggested dataset row:

```json
{
  "input": "Forecast whether this ATM call resolves ITM. Use only data through 2025-01-17 16:00:00 ET. ...",
  "output": "{\"probability\": 0.57, \"label\": 1}"
}
```

Important: train on time-ordered splits only. Do not randomly split option data, because market regimes leak across nearby dates.

## Phase 7: Evaluation protocol

Use walk-forward backtesting:

1. Train on dates <= T.
2. Calibrate on a later validation window.
3. Test on dates after validation.
4. Roll forward and repeat.

Report:

- Baseline vs LLM-agent vs ScalarLM-post-trained model.
- Calibration before and after Platt/hierarchical calibration.
- Performance by symbol, DTE bucket, moneyness bucket, IV regime, and liquidity bucket.
- Sensitivity to transaction-cost assumptions if economic utility is evaluated.

## Phase 8: Recommended first milestone

The first useful milestone is not ScalarLM. It is a reproducible DuckDB backtest with:

- 10-50 liquid tickers.
- ATM/near-money calls.
- One label definition.
- Feature pipeline.
- Three non-LLM baselines.
- BLF-style agent with numerical tools only.
- K=5 logit aggregation.
- Platt calibration.

Once this works, ScalarLM has clean training data and a meaningful target.

