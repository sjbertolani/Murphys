from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np

from murphy.metrics import brier_score, expected_calibration_error, log_loss
from murphy.priors import clamp_probability, risk_neutral_call_itm_probability


DEFAULT_FEATURES = [
    "dte",
    "implied_vol",
    "log_open_interest",
    "log_volume",
    "moneyness",
    "realized_vol_20",
    "spread_fraction",
    "trailing_return_5",
    "trailing_return_20",
]


@dataclass(frozen=True)
class BaselineResult:
    method: str
    n_train: int
    n_test: int
    brier_score: float
    log_loss: float
    calibration_error: float
    positive_rate: float


def run_baselines(
    db_path: str = "data/murphy.duckdb",
    test_fraction: float = 0.3,
    calibration_fraction: float = 0.2,
    persist: bool = True,
) -> list[BaselineResult]:
    """Train/evaluate simple probability baselines on labeled option examples."""
    import duckdb

    con = duckdb.connect(db_path)
    try:
        df = load_feature_frame(con)
        if len(df) < 20:
            raise ValueError("Need at least 20 labeled examples to run baselines")

        train_idx, test_idx = chronological_split(df["forecast_timestamp"].to_numpy(), test_fraction)
        fit_idx, calibration_idx = chronological_split(
            df["forecast_timestamp"].to_numpy()[train_idx],
            calibration_fraction,
        )
        fit_idx = train_idx[fit_idx]
        calibration_idx = train_idx[calibration_idx]

        y_train = df["label"].to_numpy(dtype=int)[fit_idx]
        y_calibration = df["label"].to_numpy(dtype=int)[calibration_idx]
        y_test = df["label"].to_numpy(dtype=int)[test_idx]

        predictions: dict[str, np.ndarray] = {}
        predictions["empirical_base_rate"] = empirical_base_rate_predictions(y_train, len(test_idx))

        market_prior_cal = market_prior_predictions(df.iloc[calibration_idx])
        market_prior = market_prior_predictions(df.iloc[test_idx])
        if market_prior is not None:
            predictions["black_scholes_prior"] = market_prior
            calibrated = platt_calibrate_predictions(market_prior_cal, y_calibration, market_prior)
            if calibrated is not None:
                predictions["black_scholes_prior_platt"] = calibrated

        logistic_cal, logistic = logistic_predictions(
            df,
            fit_idx,
            calibration_idx,
            test_idx,
            DEFAULT_FEATURES,
        )
        if logistic is not None:
            predictions["logistic_regression"] = logistic
            calibrated = platt_calibrate_predictions(logistic_cal, y_calibration, logistic)
            if calibrated is not None:
                predictions["logistic_regression_platt"] = calibrated

        results: list[BaselineResult] = []
        run_id = str(uuid4())
        created_at = datetime.now(timezone.utc)
        example_ids = df["example_id"].to_numpy()[test_idx]

        for method, probs in predictions.items():
            probs = np.array([clamp_probability(float(p), eps=0.01) for p in probs])
            result = BaselineResult(
                method=method,
                n_train=len(fit_idx),
                n_test=len(test_idx),
                brier_score=brier_score(probs.tolist(), y_test.tolist()),
                log_loss=log_loss(probs.tolist(), y_test.tolist()),
                calibration_error=expected_calibration_error(probs.tolist(), y_test.tolist(), bins=10),
                positive_rate=float(y_test.mean()),
            )
            results.append(result)
            if persist:
                persist_forecasts(con, example_ids, method, probs, created_at)
                persist_eval_result(con, run_id, "chronological_test", result, created_at)

        return results
    finally:
        con.close()


def load_feature_frame(con: Any):
    query = """
    SELECT *
    FROM (
      SELECT
        e.example_id,
        e.symbol,
        e.forecast_timestamp,
        e.expiration,
        e.strike,
        e.spot,
        e.dte,
        e.moneyness,
        e.label,
        f.feature_name,
        f.feature_value
      FROM option_examples e
      LEFT JOIN features f
        ON f.example_id = e.example_id
       AND f.feature_timestamp <= e.forecast_timestamp
      WHERE e.label IS NOT NULL
    )
    PIVOT (max(feature_value) FOR feature_name IN (
      'ask',
      'bid',
      'implied_vol',
      'log_open_interest',
      'log_volume',
      'mid',
      'realized_vol_20',
      'spread_fraction',
      'trailing_return_5',
      'trailing_return_20'
    ))
    ORDER BY forecast_timestamp, symbol, expiration, strike
    """
    df = con.execute(query).df()
    # DuckDB's PIVOT returns quoted column names in some versions.
    df.columns = [str(column).strip("'") for column in df.columns]
    return df


def chronological_split(timestamps: np.ndarray, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    order = np.argsort(timestamps)
    split = max(1, min(len(order) - 1, int(round(len(order) * (1.0 - test_fraction)))))
    return order[:split], order[split:]


def empirical_base_rate_predictions(y_train: np.ndarray, n_test: int) -> np.ndarray:
    # Laplace smoothing.
    probability = (float(y_train.sum()) + 1.0) / (len(y_train) + 2.0)
    return np.full(n_test, probability, dtype=float)


def market_prior_predictions(df):
    if "implied_vol" not in df.columns:
        return None

    probs = []
    for row in df.itertuples(index=False):
        iv = getattr(row, "implied_vol", None)
        if iv is None or not np.isfinite(iv) or iv <= 0:
            probs.append(np.nan)
            continue
        probs.append(
            risk_neutral_call_itm_probability(
                spot=float(row.spot),
                strike=float(row.strike),
                dte=float(row.dte),
                volatility=float(iv),
            )
        )
    values = np.asarray(probs, dtype=float)
    if np.isnan(values).all():
        return None
    fill = float(np.nanmean(values))
    return np.where(np.isnan(values), fill, values)


def logistic_predictions(
    df,
    fit_idx: np.ndarray,
    calibration_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_names: list[str],
):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    available = [feature for feature in feature_names if feature in df.columns]
    if not available:
        return None

    x = df[available].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    if len(np.unique(y[fit_idx])) < 2:
        return None, None

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(x[fit_idx], y[fit_idx])
    return model.predict_proba(x[calibration_idx])[:, 1], model.predict_proba(x[test_idx])[:, 1]


def platt_calibrate_predictions(
    calibration_probabilities: np.ndarray | None,
    calibration_labels: np.ndarray,
    test_probabilities: np.ndarray,
) -> np.ndarray | None:
    from sklearn.linear_model import LogisticRegression

    if calibration_probabilities is None:
        return None
    if len(calibration_probabilities) != len(calibration_labels):
        raise ValueError("calibration probabilities and labels must have same length")
    if len(np.unique(calibration_labels)) < 2:
        return None

    x_cal = np.array([_safe_logit(p) for p in calibration_probabilities], dtype=float).reshape(-1, 1)
    x_test = np.array([_safe_logit(p) for p in test_probabilities], dtype=float).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(x_cal, calibration_labels)
    return calibrator.predict_proba(x_test)[:, 1]


def _safe_logit(probability: float) -> float:
    p = clamp_probability(float(probability), eps=0.01)
    return float(np.log(p / (1.0 - p)))


def persist_forecasts(
    con: Any,
    example_ids: np.ndarray,
    method: str,
    probabilities: np.ndarray,
    created_at: datetime,
) -> None:
    rows = [
        (str(example_id), method, float(probability), None, None, created_at)
        for example_id, probability in zip(example_ids, probabilities, strict=True)
    ]
    con.executemany(
        """
        INSERT OR REPLACE INTO forecasts
          (example_id, method, raw_probability, aggregate_probability, calibrated_probability, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def persist_eval_result(
    con: Any,
    run_id: str,
    split: str,
    result: BaselineResult,
    created_at: datetime,
) -> None:
    con.execute(
        """
        INSERT INTO eval_results
          (run_id, split, method, brier_score, log_loss, auc, calibration_error, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            split,
            result.method,
            result.brier_score,
            result.log_loss,
            None,
            result.calibration_error,
            created_at,
        ],
    )
