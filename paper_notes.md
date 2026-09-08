# Agentic Forecasting Paper Notes

Paper: Kevin Murphy, "Agentic Forecasting using Sequential Bayesian Updating of Linguistic Beliefs", arXiv:2604.18576v3, 2026-05-04.

Local PDFs:
- `2604.18576v2.pdf`: previous local copy.
- `2604.18576v3.pdf`: current arXiv copy.

Version comparison: see [docs/paper_v2_v3_comparison.md](docs/paper_v2_v3_comparison.md).

## One-paragraph summary

The paper introduces BLF, the Bayesian Linguistic Forecaster: an agentic binary forecasting system that combines iterative tool use, a structured natural-language belief state, multi-trial aggregation, and hierarchical calibration. Instead of dumping all retrieved text into context, BLF asks the LLM to maintain a semi-structured belief object at each step: probability, confidence, evidence for and against, and open questions. It runs several independent trials, aggregates probabilities in logit space, and then applies Platt scaling with source-specific offsets. On 400 ForecastBench backtesting questions, BLF beats the reported top public methods and the crowd/empirical baseline, with v3 emphasizing mixed-effects/paired analysis, cross-LLM experiments, and more careful interpretation of which components matter by base LLM and crowd-prior setting.

## Main ideas

1. Linguistic belief state
   - Maintained at every agent step.
   - Contains probability, confidence, pro/con evidence, open questions, and update reasoning.
   - Evidence items are expected to cite sources.
   - The belief state is "Bayesian-style" rather than strict Bayesian inference; the LLM performs the update and may not satisfy formal Bayesian consistency.
   - Its practical role is to organize, compress, and scaffold reasoning.

2. Sequential tool loop
   - The LLM chooses one action and updates belief in the same generation.
   - Actions include web search, reading selected search files, URL lookup, source-specific time-series tools, Wikipedia snapshot fetching, and submit.
   - Max steps are usually 10; if no submit happens, the last belief probability is forced as the answer.

3. Leakage control for backtesting
   - Search date filtering.
   - LLM-based result leak filtering.
   - Date-clamped data tools.
   - URL blocking for resolution sources.
   - The paper reports a 1.5% residual undetected search leakage rate in a post-hoc audit.

4. Source tools and priors
   - Market questions can use market/crowd estimates.
   - Dataset questions can use empirical source/subtype priors.
   - Time-series tools provide date-limited history and optional model estimates.
   - DBnomics uses a bypass KNN estimator because the LLM struggled with raw temperature data.

5. Multi-trial aggregation
   - Run K independent trials, default K=5.
   - Aggregate with logit-space mean by default.
   - Optional shrinkage pulls toward an empirical or uniform prior when trial logit variance is high.
   - Shrinkage helped smaller/noisier AIBQ2 settings but not ForecastBench, where the best shrinkage setting reduced to plain logit averaging.

6. Calibration
   - Platt scaling maps raw probabilities through sigmoid(a * logit(p) + b).
   - ForecastBench uses hierarchical Platt scaling with per-source intercept offsets.
   - This prevents global calibration from over-shrinking sources with highly skewed base rates.
   - v3 clarifies that hierarchical calibration is especially important in zero-shot/empirical-prior settings, while calibration has smaller marginal effect on the full BLF agent.

## Useful implementation details

- Belief state JSON should be a first-class object stored after each step, not just a prompt artifact.
- Search should save full documents separately and show snippets first; selected documents are summarized by a cheaper model.
- Search and tool outputs should be persisted, because auditability is part of the method.
- Tool availability should depend on question source/type.
- DB-backed trace storage will matter immediately: question, trial, step, action, observation metadata, belief state, raw probability, aggregate probability, calibrated probability, and final label.
- Use paired evaluation and bootstrap confidence intervals when comparing variants; raw leaderboard comparisons confound question difficulty.

## Mapping to call option prediction

For call options, the natural binary event is:

> Will the underlying close above the call strike at the forecast horizon or expiration?

For ATM or near-the-money calls, this becomes a probabilistic classification problem near the decision boundary. BLF's belief-state architecture can wrap a more numerical option pipeline:

- "Question" = option contract + forecast timestamp + horizon/expiration + binary resolution rule.
- "Evidence" = price/volume/open interest, implied volatility, realized volatility, underlying returns, earnings/events, macro/news, market regime, and option-chain context.
- "Tools" = DuckDB-backed feature retrieval, option-chain snapshot lookup, volatility models, Black-Scholes-derived priors, event calendar lookup, and web/news search if allowed.
- "Crowd prior" analogue = option-implied probability or risk-neutral probability adjusted cautiously; treat it as a strong but biased prior, not ground truth.

## Cautions

- The paper is about forecasting benchmark questions, not trading profitability. Accurate probabilities can still lose money after spreads, fees, liquidity, and risk premia.
- For options, evaluate both probability accuracy and economic utility. Brier/log loss alone is not enough.
- Strict temporal joins are non-negotiable. Options data is especially prone to leakage through revised OHLCV, earnings calendars, end-of-day option chains, and survivorship bias.
