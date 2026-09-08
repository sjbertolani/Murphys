# Rebuilding A Forecasting Paper Into A Live Options Lab With Codex

On Monday, Kevin Murphy published an interesting paper.

That sentence undersells it a little. Murphy is one of the people whose work
helped shape modern machine learning practice, and his Probabilistic Machine
Learning books are sitting on a lot of shelves for good reason. So when he
published "Agentic Forecasting using Sequential Bayesian Updating of Linguistic
Beliefs," it was worth paying attention.

On Tuesday, I saw the paper on Twitter.

On Wednesday and Thursday, I had Codex build a working implementation with a
few modifications of my own.

By Friday, it was running on Google Cloud Platform and making true blind
predictions on live option contracts.

That is the part I keep coming back to. Not just that an LLM helped write code.
That has become almost normal. The strange and exciting part is how quickly the
distance collapsed between reading a research idea and operating a real system
that tests a variant of it.

Research papers often end with a familiar kind of cliffhanger. The idea is
clear. The diagrams are persuasive. The results make you want to try it
immediately. Then you look for the code and discover that the code is not the
artifact being shared.

That used to be the end of the weekend project.

This repository started from Murphy's paper. It introduces BLF, the Bayesian
Linguistic Forecaster: an agentic forecasting system that asks a language model
to maintain a structured belief state, update that belief after tool use, run
multiple independent trials, aggregate probabilities in logit space, and
calibrate the final result.

It is a very natural idea once you see it. Instead of treating the model as a
one-shot oracle, treat it as a forecaster with a notebook. It has a current
probability, confidence, evidence for and against, and unresolved questions. It
can go look things up. It can change its mind. The final answer is not just a
number, but the last state of a recorded reasoning process.

The paper did not include code. It also, naturally, did not include the thing I
wanted to build next: true blind prediction on live option contracts. A few
years ago, that gap would have been the main project. You would spend most of
your energy reconstructing scaffolding from prose: schemas, runners, storage,
prompts, calibration reports, deployment scripts, and all the unglamorous
plumbing that has to exist before the first serious experiment can run.

With Codex and modern LLMs, that is no longer the bottleneck.

The bottleneck is judgment.

What exactly counts as the forecast? What information was available at the time
the prediction was made? What is the label? How do we store enough evidence to
audit ourselves later? How do we avoid building a beautiful historical backtest
that quietly sees the future?

Those became the interesting questions.

## From ForecastBench To Call Options

The BLF paper evaluates forecasting benchmark questions. Murphy adapts the same
core idea to a narrower financial question:

```text
Will the price of $TICKER be greater than $STRIKE on $EXPIRY_DATE?
```

For a call option, that is the question of whether the underlying finishes in
the money. Near-the-money contracts are especially interesting because they sit
near the decision boundary. The system is not asking an LLM to predict a distant
tail event from vibes. It is asking for a calibrated probability on a hard,
well-defined binary outcome, with market-implied information available as a
prior.

That matters. Options already encode a crowd-like probability through prices,
volatility, and moneyness. The LLM should not replace that signal. It should sit
around it, interpret context, expose uncertainty, and produce an auditable
probability trace that can later be judged by proper scoring rules.

Murphy currently focuses on liquid names like AAPL, MSFT, NVDA, AMD, SPY, and
QQQ. The live generator looks at short-dated expiries, builds a ladder of call
strikes around spot, and turns each selected strike into a binary question.

The important detail is that the unit of prediction is not just the English
question. It is the market snapshot at prediction time.

The same sentence might appear again an hour later, but it is a different
forecast. Spot moved. The option mid changed. Implied volatility changed. Volume
and open interest may have changed. News context may have changed. The prompt
and the LLM response belong to that specific information cutoff.

That is the start of a real live forecasting dataset.

## The Big Deviation: True Blind Prediction

Historical replay is useful, but it is also dangerous. If you are not careful,
you can accidentally give the system end-of-day data, revised data, post-event
articles, stale labels, or train/test splits that leak market regime information
across nearby contracts.

Murphy is built around the opposite constraint: make predictions before the
answer is known.

The live loop collects option-chain snapshots and underlying bars, generates
questions, calls the model, stores the prompt and response, and waits. Resolution
happens later, after the expiry-date underlying price is available. Evaluation
uses those stored records rather than reconstructing the past from whatever a
data provider happens to return today.

That one design choice changes the feel of the project. It becomes less like
"can I reproduce a leaderboard number?" and more like "can I run a small
forecasting observatory that tells the truth about its own inputs?"

Every forecast gets an information cutoff. Prompts are built only from stored
evidence available at or before that cutoff. External context is cached. LLM
responses are persisted. Reports include leakage checks. Walk-forward analysis
splits by contract group so repeated hourly rows from the same option contract
do not end up on both sides of the train/test boundary.

This is not glamorous. It is the part that makes the glamour worth anything.

## The Second Deviation: Run It On GCP

The paper is about a method. This repository is also about operating the method.

Murphy runs locally with DuckDB, but the live version is designed for Google
Cloud Platform:

- Cloud Run Jobs execute collection, prediction, resolution, reporting, and
  dataset export commands.
- Cloud Scheduler triggers those jobs during trading hours and after market
  close.
- Cloud SQL Postgres stores the transactional live state.
- BigQuery mirrors tables for analytics.
- GCS stores exported datasets and reports.
- Secret Manager holds API keys and database passwords.

That stack is intentionally ordinary. The goal is not to invent infrastructure.
The goal is to make the forecasting process repeatable, observable, and ordinary
enough that the research questions can stay interesting.

The first data provider is Yahoo/yfinance because it is good enough to start
collecting live snapshots immediately. That choice is provisional. The provider
interface is abstracted so a more serious options feed can replace it later
without changing the database model.

## What Codex Changed

The underrated thing about using Codex on a project like this is not that it
writes code quickly. It is that it collapses the distance between reading and
building.

That Wednesday/Thursday sprint changed the rhythm of the work. I could read a
section of the paper, decide how I wanted to adapt it, and then turn that
decision into a schema, CLI command, test, report, or deployment note while the
design was still fresh. The implementation became a conversation with the paper
instead of a long detour away from it.

That does not remove the need to understand the method. If anything, it makes
the need sharper. Codex can scaffold the runner, but it cannot decide what the
label should mean. It can wire a report, but it cannot decide whether a split is
honest. It can implement calibration, but it cannot tell you that your market
prior is economically biased unless you ask the right question.

The power is in moving faster through the mechanical layers so that more human
attention can go to the assumptions.

## What Is In The Repo

The repo includes:

- DuckDB and Cloud SQL repositories.
- Option-chain and underlying-bar collection.
- Live question generation.
- OpenAI-compatible prediction calls.
- Deterministic dry-run prediction for smoke tests.
- Prompt and response persistence.
- Resolution workflows.
- Offline analysis reports.
- Walk-forward calibration comparisons.
- ScalarLM-style dataset exports.
- GCP deployment notes and bootstrap scripts.

It is still research software. It is not a trading bot. It is not investment
advice. It is a lab for asking whether BLF-style structured belief updates can
produce useful, calibrated probabilities in a noisy market domain where leakage
control matters.

## Why Publish It

The reason to publish this is not that it is finished. The reason is that the
old boundary between "paper idea" and "working research system" has moved.

A paper without code is no longer a locked door. It is a specification, a set of
claims, and an invitation to build carefully.

Codex and modern LLMs make the first implementation cheaper. They do not make
the experiment automatically good. The job is still to adapt, test, audit,
measure, and be honest about what the system knew at the moment it made each
prediction.

That is what Murphy is trying to do: take an agentic forecasting idea from a
paper, move it into a true blind live setting, and leave behind a public codebase
that makes the assumptions visible enough for other people to challenge them.
