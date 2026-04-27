# Reference Implementations and Packages

## Exact paper implementation

I did not find a public GitHub repository for Murphy's BLF paper or "Bayesian Linguistic Forecaster" as of this search. The paper appears to describe the system in enough detail to reimplement the main architecture.

Searched terms included:

- "Agentic Forecasting using Sequential Bayesian Updating of Linguistic Beliefs" GitHub
- "Bayesian Linguistic Forecaster" GitHub
- "2604.18576" GitHub
- "Kevin Murphy Bayesian Linguistic Forecaster GitHub"

## Closest forecasting-agent references

1. ForecastBench
   - Repo: https://github.com/forecastingresearch/forecastbench
   - Why it matters: official benchmark pipeline for question generation, datasets, scoring, and leaderboard methodology.
   - Use for: dataset schema inspiration, scoring, submission/evaluation logic, benchmark hygiene.

2. ForecastBench datasets
   - Repo linked from ForecastBench README: https://github.com/forecastingresearch/forecastbench-datasets
   - Use for: historical leaderboard predictions and benchmark data where licensing permits.

3. Metaculus forecasting-tools
   - Repo: https://github.com/Metaculus/forecasting-tools
   - Why it matters: Python framework for Metaculus forecasting bots, API wrappers, benchmarking, and prebuilt bot classes.
   - Use for: bot scaffolding, tournament integration, Metaculus API models, and benchmarking utilities.

4. Metaculus bot template
   - Repo: https://github.com/Metaculus/metac-bot-template
   - Use for: minimal, practical bot skeleton.

5. mini-prophet
   - Paper cites: https://github.com/ai-prophet/mini-prophet
   - Why it matters: cited as an example of text accumulation rather than BLF-style structured belief state.
   - Use for: contrast/reference only, not as the target architecture.

## Kevin Patrick Murphy / probml ecosystem

1. probml organization
   - https://github.com/probml
   - Contains Kevin Murphy's book/software ecosystem.

2. pyprobml
   - https://github.com/probml/pyprobml
   - Python code for the Probabilistic Machine Learning books.
   - Use for: reference implementations of probabilistic modeling, calibration, Bayesian inference examples.

3. dynamax
   - https://github.com/probml/dynamax
   - JAX package for probabilistic state-space modeling.
   - Use for: hidden Markov models, state-space models, Kalman-style models, volatility/regime modeling if we choose a JAX path.

4. sts-jax
   - https://github.com/probml/sts-jax
   - Structural time series in JAX.
   - Use for: baseline time-series models and Bayesian structural time-series ideas.

5. rebayes
   - https://github.com/probml/rebayes
   - Recursive Bayesian estimation / online inference.
   - Use for: online updating of model parameters or calibration, aligned with the paper's sequential Bayesian flavor.

## Neil Lawrence / SheffieldML lineage

1. GPy
   - https://github.com/SheffieldML/GPy
   - Gaussian process framework in Python from SheffieldML.
   - Use for: GP regression/classification baselines, volatility surfaces, nonparametric uncertainty modeling.

2. GPflow
   - https://github.com/GPflow/GPflow
   - TensorFlow Gaussian process library originally created by Lawrence-group students/collaborators James Hensman and Alexander Matthews.
   - Use for: modern GP models if TensorFlow is acceptable.

For this project, I would only use GPy/GPflow if we have a clear GP modeling need. For the first implementation, DuckDB + scikit-learn/statsmodels + a light LLM agent is likely faster and more robust.

## ScalarLM

1. Docs
   - https://www.scalarlm.com/
   - ScalarLM is an open-source stack for closed-loop LLM experimentation: inference and post-training in one deployment.

2. Source
   - https://github.com/tensorwavecloud/ScalarLM

3. Key capabilities relevant here
   - OpenAI-compatible inference endpoint.
   - Training endpoint for post-training jobs.
   - vLLM for inference.
   - Megatron-LM for training.
   - Hugging Face Hub integration.
   - Custom training code can be supplied via a local `ml/` directory.

## Database and data stack

1. DuckDB
   - Use as the local analytics database for options, underlying bars, features, forecasts, and evaluations.

2. Recommended Python packages
   - `duckdb`, `ibis-framework` or direct SQL, `polars`, `pandas`, `pyarrow`
   - `numpy`, `scipy`, `scikit-learn`, `statsmodels`
   - `pydantic` for schemas
   - `litellm` or OpenAI-compatible client for model abstraction
   - `yfinance` only for prototyping; serious option work needs a reliable historical options vendor
   - `mibian` or custom Black-Scholes utilities; keep formulas transparent and tested

