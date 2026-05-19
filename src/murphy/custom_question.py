from __future__ import annotations

from datetime import UTC, datetime, time
from typing import Any

from murphy.live import LiveQuestion, call_option_question_text, live_llm_prompt
from murphy.priors import bayesian_binary_update, clamp_probability, delta_as_probability
from murphy.priors import risk_neutral_call_itm_probability
from murphy.schemas import OptionRight, OptionSnapshot
from murphy.training.datasets import contract_group_key
from murphy.walk_forward import _fit_logit_ensemble, _fit_platt
from murphy.web_context import NewsContextItem, news_items_payload


WALK_FORWARD_SHADOW_CANDIDATE = "platt_llm_probability"
LEARNED_ENSEMBLE_CANDIDATE = "learned_logit_ensemble"
CURRENT_PRIMARY_CANDIDATE = "fixed_blf_posterior"


def score_custom_question(
    repository,
    market_provider,
    predictor,
    *,
    symbol: str,
    strike: float,
    target_date: str | datetime,
    model: str,
    news_provider=None,
    lookback_days: int = 5,
    news_limit: int = 5,
    min_train_groups: int = 100,
    report_limit: int = 10000,
    max_expiry_gap_days: int = 7,
    include_news_context: bool = True,
    as_of: datetime | None = None,
    record_external_calls: bool = True,
) -> dict[str, Any]:
    as_of = _normalize_datetime(as_of or datetime.now(UTC))
    normalized_symbol = symbol.strip().upper()
    question_expiration = _parse_target_date(target_date)
    if question_expiration <= as_of:
        raise ValueError("target_date must be in the future")

    bars = market_provider.fetch_underlying_bars(
        [normalized_symbol],
        lookback_days=lookback_days,
        as_of=as_of,
    )
    if not bars:
        raise ValueError(f"no underlying bars available for {normalized_symbol}")

    question_dte = (question_expiration - as_of).total_seconds() / 86400.0
    snapshots = market_provider.fetch_option_chain_snapshots(
        [normalized_symbol],
        min_dte=0.0,
        max_dte=max(question_dte + max_expiry_gap_days, 1.0),
        as_of=as_of,
    )
    if record_external_calls and hasattr(repository, "record_external_call"):
        repository.record_external_call(
            provider=market_provider.name,
            call_type="custom_question_market_snapshot",
            request_payload={
                "symbol": normalized_symbol,
                "strike": strike,
                "target_date": question_expiration.isoformat(),
                "lookback_days": lookback_days,
                "max_expiry_gap_days": max_expiry_gap_days,
            },
            response_payload={
                "underlying_bars": [bar.model_dump(mode="json") for bar in bars],
                "option_snapshots": [snapshot.model_dump(mode="json") for snapshot in snapshots],
                "counts": {
                    "underlying_bars": len(bars),
                    "option_snapshots": len(snapshots),
                },
            },
            captured_at=as_of,
            information_cutoff=as_of,
            source_timestamp=as_of,
        )

    proxy_snapshot = _select_proxy_snapshot(
        snapshots,
        symbol=normalized_symbol,
        target_expiration=question_expiration,
        requested_strike=float(strike),
        max_expiry_gap_days=max_expiry_gap_days,
    )
    latest_bar = max(bars, key=lambda bar: bar.timestamp)
    spot = float(proxy_snapshot.spot or latest_bar.close)
    moneyness = spot / float(strike) - 1.0
    market_prior = _market_prior_from_proxy_snapshot(
        spot=spot,
        strike=float(strike),
        question_dte=question_dte,
        implied_volatility=proxy_snapshot.implied_volatility,
        delta=proxy_snapshot.delta,
    )

    evaluation = repository.evaluation_report(
        ticker=None,
        limit=report_limit,
        include_unresolved=False,
    )
    resolved_rows = _eligible_resolved_rows(evaluation.get("predictions", []), as_of=as_of)
    historical_prior, historical_scope, historical_count = _historical_empirical_prior_from_rows(
        resolved_rows,
        symbol=normalized_symbol,
        dte=question_dte,
        moneyness=moneyness,
    )
    combined_prior = (
        market_prior
        if historical_prior is None
        else bayesian_binary_update(market_prior, historical_prior, signal_weight=0.5)
    )

    news_items: list[NewsContextItem] = []
    if include_news_context and news_provider is not None:
        fetched_items = news_provider.fetch([normalized_symbol], limit_per_ticker=news_limit)
        news_items = [
            item
            for item in fetched_items
            if item.published_at is None or _normalize_datetime(item.published_at) <= as_of
        ]
        if record_external_calls and hasattr(repository, "record_external_call"):
            repository.record_external_call(
                provider=news_provider.name,
                call_type="custom_question_news_context",
                request_payload={
                    "symbol": normalized_symbol,
                    "limit_per_ticker": news_limit,
                    "target_date": question_expiration.isoformat(),
                },
                response_payload=news_items_payload(news_items),
                captured_at=as_of,
                information_cutoff=as_of,
                source_timestamp=as_of,
            )

    evidence_block = _custom_evidence_block(
        symbol=normalized_symbol,
        question_expiration=question_expiration,
        requested_strike=float(strike),
        proxy_snapshot=proxy_snapshot,
        spot=spot,
        question_dte=question_dte,
        moneyness=moneyness,
        bars=bars,
        news_items=news_items,
        prior_components={
            "market_implied_prior_probability": market_prior,
            "historical_empirical_prior_probability": historical_prior,
            "historical_empirical_scope": historical_scope,
            "historical_empirical_count": historical_count,
            "combined_prior_probability": combined_prior,
        },
    )
    question = LiveQuestion(
        question_id="custom-question",
        example_id="custom-question",
        symbol=normalized_symbol,
        strike=float(strike),
        expiration=question_expiration,
        forecast_timestamp=as_of,
        information_cutoff=as_of,
        question_text=call_option_question_text(normalized_symbol, float(strike), question_expiration),
    )
    prompt = live_llm_prompt(question, evidence_block)
    raw_probability, reasoning, raw_response = predictor.predict(prompt)
    raw_probability = clamp_probability(float(raw_probability), eps=0.01)
    fixed_posterior_probability = bayesian_binary_update(
        combined_prior,
        raw_probability,
        signal_weight=0.75,
    )
    if record_external_calls and hasattr(repository, "record_external_call"):
        repository.record_external_call(
            provider="openai" if model != "dry-run" else "deterministic",
            call_type="custom_question_llm_prediction",
            request_payload={
                "symbol": normalized_symbol,
                "strike": strike,
                "target_date": question_expiration.isoformat(),
                "model": model,
                "prompt": prompt,
            },
            response_payload={
                "probability": raw_probability,
                "reasoning": reasoning,
            },
            response_text=raw_response,
            captured_at=as_of,
            information_cutoff=as_of,
            source_timestamp=as_of,
        )

    platt_probability, platt_metadata = _platt_llm_probability(
        resolved_rows,
        raw_probability=raw_probability,
        min_train_groups=min_train_groups,
    )
    learned_probability, learned_metadata = _learned_ensemble_probability(
        resolved_rows,
        raw_probability=raw_probability,
        fixed_posterior_probability=fixed_posterior_probability,
        min_train_groups=min_train_groups,
    )
    if platt_probability is not None:
        selected_method = WALK_FORWARD_SHADOW_CANDIDATE
        selected_probability = platt_probability
        calibration_metadata = platt_metadata
        selection_reason = (
            "Using the latest selected walk-forward calibration candidate because enough "
            "prior resolved contract groups are available."
        )
    elif learned_probability is not None:
        selected_method = LEARNED_ENSEMBLE_CANDIDATE
        selected_probability = learned_probability
        calibration_metadata = learned_metadata
        selection_reason = (
            "Falling back to the learned logit ensemble because Platt LLM calibration "
            "was not available."
        )
    else:
        selected_method = CURRENT_PRIMARY_CANDIDATE
        selected_probability = fixed_posterior_probability
        calibration_metadata = platt_metadata
        selection_reason = (
            "Falling back to the fixed BLF posterior because the learned ensemble "
            "and Platt LLM calibration guardrails were not met."
        )

    return {
        "question_text": question.question_text,
        "forecast_timestamp": as_of,
        "information_cutoff": as_of,
        "symbol": normalized_symbol,
        "requested_strike": float(strike),
        "target_date": question_expiration.date().isoformat(),
        "question_dte": question_dte,
        "prompt": prompt,
        "evidence_block": evidence_block,
        "proxy_option": {
            "option_symbol": proxy_snapshot.option_symbol,
            "expiration": proxy_snapshot.expiration.isoformat(),
            "strike": proxy_snapshot.strike,
            "expiry_gap_days": abs(
                (proxy_snapshot.expiration - question_expiration).total_seconds()
            )
            / 86400.0,
            "bid": proxy_snapshot.bid,
            "ask": proxy_snapshot.ask,
            "mid": proxy_snapshot.mid,
            "implied_volatility": proxy_snapshot.implied_volatility,
            "delta": proxy_snapshot.delta,
            "volume": proxy_snapshot.volume,
            "open_interest": proxy_snapshot.open_interest,
        },
        "prior_components": {
            "market_implied_prior_probability": market_prior,
            "historical_empirical_prior_probability": historical_prior,
            "historical_empirical_scope": historical_scope,
            "historical_empirical_count": historical_count,
            "combined_prior_probability": combined_prior,
        },
        "llm_response": {
            "model": model,
            "probability": raw_probability,
            "reasoning": reasoning,
            "raw_response": raw_response,
        },
        "fixed_blf_posterior_probability": fixed_posterior_probability,
        "platt_llm_probability": platt_probability,
        "learned_logit_ensemble_probability": learned_probability,
        "calibration": calibration_metadata,
        "calibration_candidates": {
            "platt_llm_probability": platt_metadata,
            "learned_logit_ensemble": learned_metadata,
        },
        "selected_probability": selected_probability,
        "selected_method": selected_method,
        "selected_prediction_at_0_5": int(selected_probability >= 0.5),
        "selection_reason": selection_reason,
        "walk_forward_candidates": {
            "shadow_candidate": WALK_FORWARD_SHADOW_CANDIDATE,
            "comparison_candidate": LEARNED_ENSEMBLE_CANDIDATE,
            "current_primary_candidate": CURRENT_PRIMARY_CANDIDATE,
        },
        "recent_bars_count": len(bars),
        "news_items_count": len(news_items),
    }


def _platt_llm_probability(
    resolved_rows: list[dict[str, Any]],
    *,
    raw_probability: float,
    min_train_groups: int,
) -> tuple[float | None, dict[str, Any]]:
    prior_contract_groups = len({row["contract_group_key"] for row in resolved_rows})
    metadata = {
        "method": WALK_FORWARD_SHADOW_CANDIDATE,
        "prior_contract_groups": prior_contract_groups,
        "min_train_groups_required": min_train_groups,
        "used": False,
        "fallback_reason": None,
    }
    if prior_contract_groups < min_train_groups:
        metadata["fallback_reason"] = "not_enough_prior_contract_groups"
        return None, metadata

    labels = [int(row["label"]) for row in resolved_rows]
    train_llm = [_probability(row["probability"]) for row in resolved_rows]
    predictor = _fit_platt(train_llm, labels)
    if predictor is None:
        metadata["fallback_reason"] = "insufficient_label_variation"
        return None, metadata

    calibrated_probability = float(predictor([raw_probability])[0])
    metadata["used"] = True
    return calibrated_probability, metadata


def _learned_ensemble_probability(
    resolved_rows: list[dict[str, Any]],
    *,
    raw_probability: float,
    fixed_posterior_probability: float,
    min_train_groups: int,
) -> tuple[float | None, dict[str, Any]]:
    prior_contract_groups = len({row["contract_group_key"] for row in resolved_rows})
    metadata = {
        "method": LEARNED_ENSEMBLE_CANDIDATE,
        "prior_contract_groups": prior_contract_groups,
        "min_train_groups_required": min_train_groups,
        "used": False,
        "fallback_reason": None,
    }
    if prior_contract_groups < min_train_groups:
        metadata["fallback_reason"] = "not_enough_prior_contract_groups"
        return None, metadata

    labels = [int(row["label"]) for row in resolved_rows]
    train_llm = [_probability(row["probability"]) for row in resolved_rows]
    train_posterior = [_probability(row["posterior_probability"]) for row in resolved_rows]
    predictor = _fit_logit_ensemble(train_llm, train_posterior, labels)
    if predictor is None:
        metadata["fallback_reason"] = "insufficient_label_variation"
        return None, metadata

    learned_probability = float(
        predictor([raw_probability], [fixed_posterior_probability])[0]
    )
    metadata["used"] = True
    return learned_probability, metadata


def _eligible_resolved_rows(predictions: list[dict[str, Any]], *, as_of: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in predictions:
        if item.get("label") is None:
            continue
        if item.get("probability") is None or item.get("posterior_probability") is None:
            continue
        if not all(item.get("leakage_checks", {}).values()):
            continue
        forecast_timestamp = _maybe_parse_datetime(item.get("forecast_timestamp"))
        if forecast_timestamp is None or forecast_timestamp >= as_of:
            continue
        rows.append(
            {
                **item,
                "forecast_timestamp": forecast_timestamp,
                "contract_group_key": contract_group_key(item),
            }
        )
    return sorted(rows, key=lambda row: (row["forecast_timestamp"], row["contract_group_key"]))


def _historical_empirical_prior_from_rows(
    resolved_rows: list[dict[str, Any]],
    *,
    symbol: str,
    dte: float,
    moneyness: float,
) -> tuple[float | None, str | None, int]:
    scoped = [
        row
        for row in resolved_rows
        if str(row.get("symbol") or "").upper() == symbol.upper()
        and row.get("dte") is not None
        and row.get("moneyness") is not None
        and abs(float(row["moneyness"]) - moneyness) <= 0.02
        and abs(float(row["dte"]) - dte) <= 7.0
    ]
    prior = _smoothed_historical_prior(scoped, min_count=20)
    if prior is not None:
        return prior, "symbol_moneyness_dte", len(scoped)

    symbol_wide = [
        row for row in resolved_rows if str(row.get("symbol") or "").upper() == symbol.upper()
    ]
    prior = _smoothed_historical_prior(symbol_wide, min_count=20)
    if prior is not None:
        return prior, "symbol", len(symbol_wide)

    prior = _smoothed_historical_prior(resolved_rows, min_count=50)
    if prior is not None:
        return prior, "global", len(resolved_rows)
    return None, None, 0


def _smoothed_historical_prior(rows: list[dict[str, Any]], *, min_count: int) -> float | None:
    total = len(rows)
    if total < min_count:
        return None
    successes = sum(int(row["label"]) for row in rows)
    return float((successes + 1.0) / (total + 2.0))


def _select_proxy_snapshot(
    snapshots: list[OptionSnapshot],
    *,
    symbol: str,
    target_expiration: datetime,
    requested_strike: float,
    max_expiry_gap_days: int,
) -> OptionSnapshot:
    candidates = [
        snapshot
        for snapshot in snapshots
        if snapshot.symbol.upper() == symbol.upper() and snapshot.right == OptionRight.CALL
    ]
    if not candidates:
        raise ValueError(f"no call option snapshots available for {symbol}")

    ranked = sorted(
        candidates,
        key=lambda snapshot: (
            _expiry_gap_days(snapshot.expiration, target_expiration) > max_expiry_gap_days,
            snapshot.expiration < target_expiration,
            _expiry_gap_days(snapshot.expiration, target_expiration),
            abs(float(snapshot.strike) - requested_strike),
            -(float(snapshot.open_interest) if snapshot.open_interest is not None else 0.0),
            -(float(snapshot.volume) if snapshot.volume is not None else 0.0),
        ),
    )
    selected = ranked[0]
    gap_days = _expiry_gap_days(selected.expiration, target_expiration)
    if gap_days > max_expiry_gap_days:
        raise ValueError(
            f"nearest listed option expiry is {gap_days:.2f} days away from the target date"
        )
    return selected


def _custom_evidence_block(
    *,
    symbol: str,
    question_expiration: datetime,
    requested_strike: float,
    proxy_snapshot: OptionSnapshot,
    spot: float,
    question_dte: float,
    moneyness: float,
    bars,
    news_items: list[NewsContextItem],
    prior_components: dict[str, Any],
) -> str:
    bar_lines = [
        "recent_underlying_bars_most_recent_first:",
        *[
            (
                f"- {bar.timestamp.isoformat()}: open={bar.open} high={bar.high} "
                f"low={bar.low} close={bar.close} volume={bar.volume}"
            )
            for bar in sorted(bars, key=lambda item: item.timestamp, reverse=True)
        ],
    ]
    news_lines = ["cached_web_news_context:"]
    for item in news_items:
        news_lines.append(
            "- "
            f"{item.published_at.isoformat() if item.published_at is not None else 'unknown_time'} "
            f"{item.title} source={item.source} link={item.link}"
        )
    if len(news_lines) == 1:
        news_lines.append("- none_cached_for_this_forecast")

    return "\n".join(
        [
            f"symbol: {symbol}",
            f"question_target_date: {question_expiration.isoformat()}",
            f"requested_strike: {requested_strike}",
            f"proxy_option_symbol: {proxy_snapshot.option_symbol}",
            f"proxy_option_expiration: {proxy_snapshot.expiration.isoformat()}",
            f"proxy_option_strike: {proxy_snapshot.strike}",
            f"spot_at_forecast: {spot}",
            f"days_to_question_target: {question_dte:.3f}",
            f"moneyness_spot_over_strike_minus_one: {moneyness:.6f}",
            f"proxy_bid: {proxy_snapshot.bid}",
            f"proxy_ask: {proxy_snapshot.ask}",
            f"proxy_mid: {proxy_snapshot.mid}",
            f"proxy_implied_volatility: {proxy_snapshot.implied_volatility}",
            f"proxy_delta: {proxy_snapshot.delta}",
            (
                "market_implied_prior_probability: "
                f"{float(prior_components['market_implied_prior_probability']):.6f}"
            ),
            (
                "historical_empirical_prior_probability: "
                f"{prior_components['historical_empirical_prior_probability']}"
            ),
            f"historical_empirical_scope: {prior_components['historical_empirical_scope']}",
            f"historical_empirical_count: {prior_components['historical_empirical_count']}",
            (
                "combined_prior_probability: "
                f"{float(prior_components['combined_prior_probability']):.6f}"
            ),
            f"proxy_volume: {proxy_snapshot.volume}",
            f"proxy_open_interest: {proxy_snapshot.open_interest}",
            *bar_lines,
            *news_lines,
        ]
    )


def _market_prior_from_proxy_snapshot(
    *,
    spot: float,
    strike: float,
    question_dte: float,
    implied_volatility: float | None,
    delta: float | None,
) -> float:
    try:
        if implied_volatility is not None and float(implied_volatility) > 0:
            return risk_neutral_call_itm_probability(
                spot=float(spot),
                strike=float(strike),
                dte=float(question_dte),
                volatility=float(implied_volatility),
            )
    except (TypeError, ValueError):
        pass
    try:
        delta_probability = delta_as_probability(None if delta is None else float(delta))
    except (TypeError, ValueError):
        delta_probability = None
    return 0.5 if delta_probability is None else float(delta_probability)


def _expiry_gap_days(expiration: datetime, target_expiration: datetime) -> float:
    return abs((expiration - target_expiration).total_seconds()) / 86400.0


def _parse_target_date(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        normalized = _normalize_datetime(value)
        return datetime.combine(normalized.date(), time(hour=21), tzinfo=UTC)
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%m-%d-%y", "%m/%d/%y", "%m-%d-%Y", "%m/%d/%Y"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        return datetime.combine(parsed.date(), time(hour=21), tzinfo=UTC)
    raise ValueError(f"unsupported target_date format: {value}")


def _maybe_parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, str):
        try:
            return _normalize_datetime(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    return None


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _probability(value: Any) -> float:
    return clamp_probability(float(value), eps=0.01)
