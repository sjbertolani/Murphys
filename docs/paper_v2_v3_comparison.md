# Murphy BLF Paper v2-to-v3 Comparison

Compared files:

- Local previous copy: `2604.18576v2.pdf`
- Downloaded current copy: `2604.18576v3.pdf`
- arXiv: <https://arxiv.org/abs/2604.18576>

arXiv reports v3 was revised on 2026-05-04. The arXiv comments say v3 fixes an error in baseline numbers, leaves the main-method results unaffected, adds cross-LLM experiments, and improves presentation.

## High-Level Changes

v3 is not a new algorithm, but it is a meaningful revision of the empirical story.

- The core BLF stack remains: structured linguistic belief state, multi-trial aggregation, hierarchical calibration, and leakage-controlled backtesting.
- The presentation is more careful about calling the updates "Bayesian-style" rather than formal Bayesian inference.
- Headline claims are softened and made more statistically precise. v2 said BLF significantly beats all top methods across all types; v3 says BLF has the best point estimates, but some pairwise gaps are not statistically significant in lower-power market/tranche comparisons.
- v3 adds cross-LLM analysis across Pro-3.1, Flash-3.1, Kimi-K2.5, GPT-5, and Sonnet-4.6.
- v3 makes the no-LLM crowd/empirical-prior baseline more central and corrects some baseline numbers.

## Abstract And Framing

v2 described the belief update as approximate sequential Bayesian inference. v3 explicitly says the belief slots mirror a Bayesian update form, but the actual update is an LLM forward pass and may not satisfy formal Bayesian consistency.

Implementation implication: our traces should avoid implying the LLM update is mathematically exact Bayes. It is better to label these as BLF-style or Bayesian-style belief states, with explicit probabilities and evidence fields.

## Main Results

v2 Table 1 headline:

- BLF + crowd + empirical prior + calibration: BI `85.2` market, `62.4` data, `73.8` all.
- Crowd+emp baseline: BI `81.5` market, `58.3` data, `69.9` all.
- v2 said BLF significantly outperformed all external methods on both market and dataset questions.

v3 Table 1 headline:

- BLF (Pro): BI `83.8` market, `62.7` data, `73.3` all.
- Baseline no LLM crowd+emp: BI `81.5` market, `58.3` data, `69.9` all.
- BLF beats the baseline overall by `+3.4` BI with significance.
- Some external comparisons are now softer:
  - Cassi overall gap is significant.
  - GPT-5 overall gap is significant.
  - Foresight overall gap is significant on its covered tranche.
  - Grok is not statistically distinguishable from BLF on its tranche-A coverage, despite BLF's better point estimate.
  - Market-only significance is lower-powered; v3 says the only market gap reaching significance is against Foresight.

Implementation implication: our analysis reports should prioritize paired/walk-forward comparisons and confidence intervals rather than only point estimates. This supports our current move toward walk-forward reports and Platt calibration selection.

## Ablations

v2 emphasized a rank ordering of harmful removals:

- zero-shot hurts most
- no search hurts
- batch/non-sequential search hurts
- removing belief state hurts
- weaker base models hurt
- no tools is mostly neutral overall but hurts dataset questions

v3 reframes ablations around a cumulative build-up from a strong "NoBel" baseline:

- NoBel = search-enabled sequential text accumulation, no structured belief state, no shrink-prior aggregation, no calibration.
- BLF-full = belief state + shrink-prior aggregation + hierarchical calibration on top of NoBel.
- The contribution of BLF components varies substantially by base LLM and by whether the crowd anchor is present.

Implementation implication: our next paper-reconstruction step should compare against a stronger no-belief baseline, not just raw LLM vs posterior. We should add an offline experiment where the same cached evidence is fed to:

- raw one-shot LLM
- sequential search/evidence accumulation without belief state
- structured belief state
- structured belief state plus multi-trial aggregation
- calibrated final probability

## Cross-LLM Experiments

v3 adds a substantial cross-LLM analysis. Key reported pattern:

- BLF helps all tested base LLMs, but the amount differs.
- Kimi-K2.5 benefits strongly and is described as close to Pro while being cheaper/open-weights.
- Sonnet and GPT-5 see weaker BLF gains over the NoBel baseline in some settings.
- v3 includes a belief-submit consistency check: submitted probabilities match final belief probabilities, so models are at least reading the final belief slot at submit time.
- v3 suggests some models may update `belief.p` less reliably from tool evidence, motivating mid-loop intervention tests.

Implementation implication: we should not assume OpenAI models respond to BLF scaffolding the same way Gemini Pro did. For our project, this supports keeping raw LLM, fixed posterior, Platt LLM, and learned ensemble side by side, and later testing multiple OpenAI models or open models before deciding what to train with ScalarLM.

## Aggregation

v2 already said K=5 and logit-space averaging matter, with optional LOO-tuned shrinkage toward `0.5`.

v3 clarifies:

- aggregation is logit-space averaging by default
- hierarchical shrinkage can be toward an empirical or uniform prior
- on ForecastBench, shrinkage effectively reduces to logit averaging
- shrinkage is more useful in smaller/noisier settings such as AIBQ2

Implementation implication: our single-call custom scorer is still incomplete relative to the paper. We should add K independent trials per question, logit-space aggregation, and then Platt calibration. For the live options setting, a smaller K such as 3 may be a cost-conscious first step.

## Calibration

v3 keeps hierarchical Platt calibration, but clarifies where it matters most:

- full BLF is already fairly strong; calibration has limited marginal effect there
- hierarchical calibration is crucial in zero-shot/empirical-prior settings because global Platt can over-shrink source-specific extremes

Implementation implication: our current custom scorer selecting Platt-calibrated raw LLM probability is consistent with the latest walk-forward evidence in our data, but it is not the same as the paper's full hierarchical calibration. For the options domain, the analogous hierarchy should probably include ticker, DTE bucket, moneyness bucket, and maybe forecast-hour/market-regime bucket.

## Backtesting And Leakage

v3 preserves the leakage claim: four-layer date-leakage defense with post-hoc audit showing about `1.5%` residual leakage.

Implementation implication: our cached market/news/LLM records and `information_cutoff` design remain aligned with the paper. We should add a leakage audit report that samples cached web/news items and verifies publish timestamps, cache capture timestamps, and resolution dates.

## What Changed For Our Roadmap

Highest-priority changes to align with v3:

1. Rename some internal/docs wording from exact Bayesian update to BLF-style or Bayesian-style update where the LLM is doing the update.
2. Add a NoBel-style baseline: sequential evidence accumulation without structured belief state.
3. Add K-trial custom and scheduled forecasting with logit aggregation.
4. Add hierarchical calibration buckets for options rather than only global Platt/ensemble calibration.
5. Add confidence intervals and paired comparisons to walk-forward reports.
6. Consider cross-model experiments before committing to one LLM or ScalarLM training target.

## Bottom Line

v3 strengthens the paper as an empirical study but reduces some over-broad claims from v2. The main method is unchanged, but the evidence now says component value depends on model choice and prior setting. For our options project, the most relevant v3 lesson is: keep calibration and baselines empirical, test the harness against strong no-belief alternatives, and do not assume the same BLF scaffolding gains transfer identically across LLMs or market domains.
