from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from murphy.analysis_report import build_analysis_report, render_markdown_report
from murphy.baselines import run_baselines
from murphy.bigquery import (
    bigquery_table_freshness,
    initialize_bigquery_dataset,
    mirror_cloud_sql_to_bigquery,
)
from murphy.cloud_config import load_cloud_config_from_env
from murphy.cloud_sql import initialize_cloud_sql
from murphy.cloud_storage import upload_file_to_gcs
from murphy.db import MurphyDb
from murphy.external_options import (
    DEFAULT_OPTIONS_DB,
    ingest_examples_from_existing_options_db,
    materialize_features_from_existing_options_db,
)
from murphy.live import (
    export_pending_llm_prompts_jsonl,
    generate_live_questions_from_snapshots,
    resolve_due_live_questions,
)
from murphy.market_data.collect import collect_market_snapshots, provider_from_name
from murphy.offline_export import export_cloud_sql_to_duckdb
from murphy.operational_status import build_operational_status_report
from murphy.predict import DeterministicPredictor, OpenAiPredictor, predict_pending
from murphy.repository import CloudSqlRepository, DuckDbRepository
from murphy.training.datasets import (
    scalar_sft_dataset_from_report,
    scalar_sft_split_datasets_from_report,
    write_jsonl,
)
from murphy.web_context import YahooFinanceNewsProvider, news_items_payload


def main() -> None:
    parser = argparse.ArgumentParser(prog="murphy")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init-db", help="Initialize the Murphy DuckDB schema.")
    init_parser.add_argument("--db", default="data/murphy.duckdb")

    ingest_parser = subparsers.add_parser(
        "ingest-existing-options",
        help="Ingest labeled call examples from the existing options project DB.",
    )
    ingest_parser.add_argument("--source-db", default=str(DEFAULT_OPTIONS_DB))
    ingest_parser.add_argument("--target-db", default="data/murphy.duckdb")
    ingest_parser.add_argument("--max-abs-moneyness", type=float, default=0.03)
    ingest_parser.add_argument("--min-dte", type=int, default=7)
    ingest_parser.add_argument("--max-dte", type=int, default=45)
    ingest_parser.add_argument("--limit", type=int, default=None)

    feature_parser = subparsers.add_parser(
        "materialize-features",
        help="Materialize feature rows for existing Murphy option examples.",
    )
    feature_parser.add_argument("--source-db", default=str(DEFAULT_OPTIONS_DB))
    feature_parser.add_argument("--target-db", default="data/murphy.duckdb")
    feature_parser.add_argument("--no-replace", action="store_true")

    baseline_parser = subparsers.add_parser(
        "run-baselines",
        help="Run empirical, market-prior, and logistic baselines.",
    )
    baseline_parser.add_argument("--db", default="data/murphy.duckdb")
    baseline_parser.add_argument("--test-fraction", type=float, default=0.3)
    baseline_parser.add_argument("--calibration-fraction", type=float, default=0.2)
    baseline_parser.add_argument("--no-persist", action="store_true")

    live_parser = subparsers.add_parser(
        "generate-live-questions",
        help="Generate unresolved live questions from the latest option-chain snapshot.",
    )
    live_parser.add_argument("--db", default="data/murphy.duckdb")
    live_parser.add_argument("--max-abs-moneyness", type=float, default=0.03)
    live_parser.add_argument("--min-dte", type=float, default=0.0)
    live_parser.add_argument("--max-dte", type=float, default=14.0)
    live_parser.add_argument("--max-questions", type=int, default=50)
    live_parser.add_argument("--min-open-interest", type=float, default=1.0)
    live_parser.add_argument("--min-volume", type=float, default=0.0)
    live_parser.add_argument("--strike-window-size", type=int, default=5)
    live_parser.add_argument("--max-questions-per-ticker", type=int, default=None)

    prompt_parser = subparsers.add_parser(
        "export-pending-prompts",
        help="Export pending live questions as LLM prompt JSONL.",
    )
    prompt_parser.add_argument("--db", default="data/murphy.duckdb")
    prompt_parser.add_argument("--output", default="data/pending_llm_prompts.jsonl")
    prompt_parser.add_argument("--limit", type=int, default=None)

    resolve_parser = subparsers.add_parser(
        "resolve-live-questions",
        help="Resolve due live questions from stored underlying bars.",
    )
    resolve_parser.add_argument("--db", default="data/murphy.duckdb")
    resolve_parser.add_argument("--allow-prior-close", action="store_true")

    resolve_report_parser = subparsers.add_parser(
        "resolve-and-report",
        help="Collect underlying bars, resolve due questions, and print an evaluation report.",
    )
    resolve_report_parser.add_argument("--provider", default="yahoo", choices=["yahoo", "yfinance"])
    resolve_report_parser.add_argument("--backend", default="duckdb", choices=["duckdb", "cloud-sql"])
    resolve_report_parser.add_argument("--db", default="data/murphy.duckdb")
    resolve_report_parser.add_argument("--tickers", nargs="+", required=True)
    resolve_report_parser.add_argument("--lookback-days", type=int, default=10)
    resolve_report_parser.add_argument("--report-limit", type=int, default=50)
    resolve_report_parser.add_argument("--allow-prior-close", action="store_true")

    collect_parser = subparsers.add_parser(
        "collect-snapshots",
        help="Collect underlying bars and option-chain snapshots from a market data provider.",
    )
    collect_parser.add_argument("--provider", default="yahoo", choices=["yahoo", "yfinance"])
    collect_parser.add_argument("--backend", default="duckdb", choices=["duckdb", "cloud-sql"])
    collect_parser.add_argument("--db", default="data/murphy.duckdb")
    collect_parser.add_argument("--tickers", nargs="+", required=True)
    collect_parser.add_argument("--min-dte", type=int, default=0)
    collect_parser.add_argument("--max-dte", type=int, default=14)
    collect_parser.add_argument("--lookback-days", type=int, default=10)

    predict_parser = subparsers.add_parser(
        "predict-pending",
        help="Call OpenAI for pending live questions and store probabilities.",
    )
    predict_parser.add_argument("--backend", default="duckdb", choices=["duckdb", "cloud-sql"])
    predict_parser.add_argument("--db", default="data/murphy.duckdb")
    predict_parser.add_argument("--model", default="gpt-4.1-mini")
    predict_parser.add_argument("--limit", type=int, default=None)
    predict_parser.add_argument("--dry-run", action="store_true")

    cycle_parser = subparsers.add_parser(
        "run-live-cycle",
        help="Collect snapshots, generate questions, predict pending, and resolve due questions.",
    )
    cycle_parser.add_argument("--provider", default="yahoo", choices=["yahoo", "yfinance"])
    cycle_parser.add_argument("--backend", default="duckdb", choices=["duckdb", "cloud-sql"])
    cycle_parser.add_argument("--db", default="data/murphy.duckdb")
    cycle_parser.add_argument("--tickers", nargs="+", required=True)
    cycle_parser.add_argument("--min-dte", type=int, default=0)
    cycle_parser.add_argument("--max-dte", type=int, default=14)
    cycle_parser.add_argument("--lookback-days", type=int, default=10)
    cycle_parser.add_argument("--max-questions", type=int, default=50)
    cycle_parser.add_argument("--strike-window-size", type=int, default=5)
    cycle_parser.add_argument("--max-questions-per-ticker", type=int, default=None)
    cycle_parser.add_argument("--model", default="gpt-4.1-mini")
    cycle_parser.add_argument("--prediction-limit", type=int, default=None)
    cycle_parser.add_argument("--dry-run", action="store_true")
    cycle_parser.add_argument("--no-news-context", action="store_true")
    cycle_parser.add_argument("--news-limit-per-ticker", type=int, default=5)
    cycle_parser.add_argument("--summary-ticker", default=None)

    summary_parser = subparsers.add_parser(
        "summarize-live-ticker",
        help="Print a JSON trace of recent live questions, responses, and belief steps.",
    )
    summary_parser.add_argument("--backend", default="duckdb", choices=["duckdb", "cloud-sql"])
    summary_parser.add_argument("--db", default="data/murphy.duckdb")
    summary_parser.add_argument("--ticker", required=True)
    summary_parser.add_argument("--limit", type=int, default=5)

    eval_report_parser = subparsers.add_parser(
        "evaluation-report",
        help="Print a JSON audit/evaluation report for live predictions.",
    )
    eval_report_parser.add_argument("--backend", default="duckdb", choices=["duckdb", "cloud-sql"])
    eval_report_parser.add_argument("--db", default="data/murphy.duckdb")
    eval_report_parser.add_argument("--ticker", default=None)
    eval_report_parser.add_argument("--limit", type=int, default=50)
    eval_report_parser.add_argument("--resolved-only", action="store_true")

    analysis_parser = subparsers.add_parser(
        "analysis-report",
        help="Write an offline analysis report for cached live predictions.",
    )
    analysis_parser.add_argument("--backend", default="duckdb", choices=["duckdb", "cloud-sql"])
    analysis_parser.add_argument("--db", default="data/murphy.duckdb")
    analysis_parser.add_argument("--ticker", default=None)
    analysis_parser.add_argument("--limit", type=int, default=10000)
    analysis_parser.add_argument("--resolved-only", action="store_true")
    analysis_parser.add_argument("--test-fraction", type=float, default=0.2)
    analysis_parser.add_argument("--validation-fraction", type=float, default=0.0)
    analysis_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    analysis_parser.add_argument("--output", default="data/offline_analysis_report.md")

    offline_analysis_parser = subparsers.add_parser(
        "offline-analysis",
        help="Export Cloud SQL to DuckDB and write the offline analysis report.",
    )
    offline_analysis_parser.add_argument("--duckdb", default="data/murphy_offline.duckdb")
    offline_analysis_parser.add_argument("--ticker", default=None)
    offline_analysis_parser.add_argument("--limit", type=int, default=10000)
    offline_analysis_parser.add_argument("--resolved-only", action="store_true")
    offline_analysis_parser.add_argument("--test-fraction", type=float, default=0.2)
    offline_analysis_parser.add_argument("--validation-fraction", type=float, default=0.0)
    offline_analysis_parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    offline_analysis_parser.add_argument("--output", default="data/offline_analysis_report.md")

    scalar_export_parser = subparsers.add_parser(
        "export-scalar-sft-dataset",
        help="Export resolved, leakage-checked live predictions as ScalarLM SFT JSONL.",
    )
    scalar_export_parser.add_argument("--backend", default="duckdb", choices=["duckdb", "cloud-sql"])
    scalar_export_parser.add_argument("--db", default="data/murphy.duckdb")
    scalar_export_parser.add_argument("--ticker", default=None)
    scalar_export_parser.add_argument("--limit", type=int, default=10000)
    scalar_export_parser.add_argument("--output", default="data/scalar_sft_resolved.jsonl")
    scalar_export_parser.add_argument("--gcs-uri", default=None)
    scalar_export_parser.add_argument("--allow-leakage-check-failures", action="store_true")

    scalar_split_parser = subparsers.add_parser(
        "export-scalar-sft-splits",
        help="Export resolved ScalarLM SFT JSONL files with leakage-safe grouped splits.",
    )
    scalar_split_parser.add_argument("--backend", default="duckdb", choices=["duckdb", "cloud-sql"])
    scalar_split_parser.add_argument("--db", default="data/murphy.duckdb")
    scalar_split_parser.add_argument("--ticker", default=None)
    scalar_split_parser.add_argument("--limit", type=int, default=10000)
    scalar_split_parser.add_argument("--output-dir", default="data/scalar_sft_splits")
    scalar_split_parser.add_argument("--test-fraction", type=float, default=0.2)
    scalar_split_parser.add_argument("--validation-fraction", type=float, default=0.0)
    scalar_split_parser.add_argument("--gcs-uri", default=None)
    scalar_split_parser.add_argument("--allow-leakage-check-failures", action="store_true")

    status_parser = subparsers.add_parser(
        "daily-status",
        help="Print an operational JSON status report with warnings.",
    )
    status_parser.add_argument("--backend", default="duckdb", choices=["duckdb", "cloud-sql"])
    status_parser.add_argument("--db", default="data/murphy.duckdb")
    status_parser.add_argument("--include-bigquery", action="store_true")
    status_parser.add_argument("--max-snapshot-age-hours", type=float, default=6.0)
    status_parser.add_argument("--max-bigquery-age-hours", type=float, default=24.0)

    cloud_parser = subparsers.add_parser(
        "init-cloud",
        help="Initialize configured Cloud SQL and/or BigQuery resources.",
    )
    cloud_parser.add_argument("--cloud-sql", action="store_true")
    cloud_parser.add_argument("--bigquery", action="store_true")

    export_parser = subparsers.add_parser(
        "export-cloud-sql-to-duckdb",
        help="Export Cloud SQL live tables to a local DuckDB file for offline analytics.",
    )
    export_parser.add_argument("--duckdb", default="data/murphy_offline.duckdb")

    bq_export_parser = subparsers.add_parser(
        "mirror-cloud-sql-to-bigquery",
        help="Mirror Cloud SQL live/audit tables into BigQuery for analytics.",
    )
    bq_export_parser.add_argument("--tables", nargs="+", default=None)
    bq_export_parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows instead of replacing each BigQuery table.",
    )

    args = parser.parse_args()

    if args.command == "init-db":
        db = MurphyDb(Path(args.db))
        db.initialize()
        db.close()
        print(f"initialized {args.db}")
        return

    if args.command == "generate-live-questions":
        questions = generate_live_questions_from_snapshots(
            db_path=args.db,
            max_abs_moneyness=args.max_abs_moneyness,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
            max_questions=args.max_questions,
            min_open_interest=args.min_open_interest,
            min_volume=args.min_volume,
            strike_window_size=args.strike_window_size,
            max_questions_per_ticker=args.max_questions_per_ticker,
        )
        print(f"generated {len(questions)} live questions")
        for question in questions[:5]:
            print(f"{question.question_id} {question.question_text}")
        return

    if args.command == "export-pending-prompts":
        count = export_pending_llm_prompts_jsonl(
            db_path=args.db,
            output_path=args.output,
            limit=args.limit,
        )
        print(f"exported {count} prompts to {args.output}")
        return

    if args.command == "resolve-live-questions":
        count = resolve_due_live_questions(
            db_path=args.db,
            require_expiration_date=not args.allow_prior_close,
        )
        print(f"resolved {count} live questions")
        return

    if args.command == "resolve-and-report":
        repository = _repository_for_backend(args.backend, args.db)
        try:
            bar_count = _collect_resolution_bars_into_repository(
                repository=repository,
                provider_name=args.provider,
                tickers=args.tickers,
                lookback_days=args.lookback_days,
            )
            resolved = repository.resolve_due_live_questions(
                require_expiration_date=not args.allow_prior_close,
            )
            report = repository.evaluation_report(
                ticker=args.tickers[0] if len(args.tickers) == 1 else None,
                limit=args.report_limit,
                include_unresolved=True,
            )
        finally:
            repository.close()
        print(f"resolve-and-report provider={args.provider} bars={bar_count} resolved={resolved}")
        print(json.dumps(report, default=str, indent=2, sort_keys=True))
        return

    if args.command == "collect-snapshots":
        repository = _repository_for_backend(args.backend, args.db)
        try:
            bar_count, snapshot_count, result = _collect_into_repository(
                repository=repository,
                provider_name=args.provider,
                tickers=args.tickers,
                min_dte=args.min_dte,
                max_dte=args.max_dte,
                lookback_days=args.lookback_days,
            )
        finally:
            repository.close()
        print(
            f"collected provider={result.provider} bars={bar_count} "
            f"option_snapshots={snapshot_count} at={result.collected_at.isoformat()}"
        )
        return

    if args.command == "predict-pending":
        repository = _repository_for_backend(args.backend, args.db)
        try:
            predictor = (
                DeterministicPredictor(model="dry-run")
                if args.dry_run
                else OpenAiPredictor(model=args.model)
            )
            model = "dry-run" if args.dry_run else args.model
            results = predict_pending(repository, predictor, model=model, limit=args.limit)
        finally:
            repository.close()
        print(f"predicted {len(results)} pending questions")
        return

    if args.command == "run-live-cycle":
        repository = _repository_for_backend(args.backend, args.db)
        try:
            bar_count, snapshot_count, result = _collect_into_repository(
                repository=repository,
                provider_name=args.provider,
                tickers=args.tickers,
                min_dte=args.min_dte,
                max_dte=args.max_dte,
                lookback_days=args.lookback_days,
            )
            news_count = 0
            if not args.no_news_context:
                try:
                    news_count = _collect_news_context_into_repository(
                        repository=repository,
                        tickers=args.tickers,
                        information_cutoff=result.collected_at,
                        limit_per_ticker=args.news_limit_per_ticker,
                    )
                except Exception as exc:  # noqa: BLE001 - news is advisory context, not the core job.
                    normalized_tickers = sorted(
                        {ticker.strip().upper() for ticker in args.tickers if ticker.strip()}
                    )
                    news_error = f"{type(exc).__name__}: {exc}"
                    repository.record_external_call(
                        provider="yahoo_finance_news",
                        call_type="web_news_context_error",
                        request_payload={
                            "tickers": normalized_tickers,
                            "limit_per_ticker": args.news_limit_per_ticker,
                        },
                        response_payload={"error": news_error},
                        captured_at=datetime.now(UTC),
                        information_cutoff=result.collected_at,
                        source_timestamp=result.collected_at,
                    )
                    print(f"news_context_error={news_error}")
            questions = repository.generate_live_questions(
                min_dte=args.min_dte,
                max_dte=args.max_dte,
                max_questions=args.max_questions,
                strike_window_size=args.strike_window_size,
                max_questions_per_ticker=args.max_questions_per_ticker,
            )
            predictor = (
                DeterministicPredictor(model="dry-run")
                if args.dry_run
                else OpenAiPredictor(model=args.model)
            )
            model = "dry-run" if args.dry_run else args.model
            predictions = predict_pending(
                repository,
                predictor,
                model=model,
                limit=args.prediction_limit,
            )
            resolved = repository.resolve_due_live_questions()
            summary = (
                repository.live_ticker_summary(args.summary_ticker, limit=5)
                if args.summary_ticker
                else None
            )
        finally:
            repository.close()
        print(
            f"live-cycle provider={result.provider} bars={bar_count} "
            f"option_snapshots={snapshot_count} questions={len(questions)} "
            f"news_items={news_count} predictions={len(predictions)} resolved={resolved}"
        )
        if summary is not None:
            print("single_ticker_trace_json=")
            print(json.dumps(summary, default=str, indent=2, sort_keys=True))
        return

    if args.command == "summarize-live-ticker":
        repository = _repository_for_backend(args.backend, args.db)
        try:
            summary = repository.live_ticker_summary(args.ticker, limit=args.limit)
        finally:
            repository.close()
        print(json.dumps(summary, default=str, indent=2, sort_keys=True))
        return

    if args.command == "evaluation-report":
        repository = _repository_for_backend(args.backend, args.db)
        try:
            report = repository.evaluation_report(
                ticker=args.ticker,
                limit=args.limit,
                include_unresolved=not args.resolved_only,
                test_fraction=args.test_fraction,
                validation_fraction=args.validation_fraction,
            )
        finally:
            repository.close()
        print(json.dumps(report, default=str, indent=2, sort_keys=True))
        return

    if args.command == "analysis-report":
        repository = _repository_for_backend(args.backend, args.db)
        try:
            report = build_analysis_report(
                repository,
                ticker=args.ticker,
                limit=args.limit,
                include_unresolved=not args.resolved_only,
            )
        finally:
            repository.close()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "json":
            output_path.write_text(
                json.dumps(report, default=str, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            output_path.write_text(render_markdown_report(report), encoding="utf-8")
        print(f"wrote analysis report to {output_path}")
        return

    if args.command == "offline-analysis":
        config = load_cloud_config_from_env()
        if config.cloud_sql is None:
            raise ValueError("Cloud SQL env vars are not configured")
        counts = export_cloud_sql_to_duckdb(config.cloud_sql, args.duckdb)
        repository = DuckDbRepository(args.duckdb)
        try:
            report = build_analysis_report(
                repository,
                ticker=args.ticker,
                limit=args.limit,
                include_unresolved=not args.resolved_only,
                test_fraction=args.test_fraction,
                validation_fraction=args.validation_fraction,
            )
        finally:
            repository.close()
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "json":
            output_path.write_text(
                json.dumps(report, default=str, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            output_path.write_text(render_markdown_report(report), encoding="utf-8")
        print("exported Cloud SQL tables to DuckDB:")
        for table, count in counts.items():
            print(f"{table}: {count}")
        print(f"wrote analysis report to {output_path}")
        return

    if args.command == "export-scalar-sft-dataset":
        repository = _repository_for_backend(args.backend, args.db)
        try:
            report = repository.evaluation_report(
                ticker=args.ticker,
                limit=args.limit,
                include_unresolved=False,
            )
        finally:
            repository.close()
        rows = scalar_sft_dataset_from_report(
            report,
            require_leakage_checks=not args.allow_leakage_check_failures,
        )
        count = write_jsonl(rows, args.output)
        print(f"exported {count} ScalarLM SFT rows to {args.output}")
        if args.gcs_uri:
            uploaded_uri = upload_file_to_gcs(args.output, args.gcs_uri)
            print(f"uploaded ScalarLM SFT rows to {uploaded_uri}")
        return

    if args.command == "export-scalar-sft-splits":
        repository = _repository_for_backend(args.backend, args.db)
        try:
            report = repository.evaluation_report(
                ticker=args.ticker,
                limit=args.limit,
                include_unresolved=False,
            )
        finally:
            repository.close()
        split_rows, manifest = scalar_sft_split_datasets_from_report(
            report,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
            require_leakage_checks=not args.allow_leakage_check_failures,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written_files = []
        counts = {}
        for split, rows in sorted(split_rows.items()):
            output_path = output_dir / f"{split}.jsonl"
            counts[split] = write_jsonl(rows, output_path)
            written_files.append(output_path)
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({**manifest, "files": counts}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written_files.append(manifest_path)
        print(f"exported grouped ScalarLM split rows to {output_dir}")
        print(json.dumps({**manifest, "files": counts}, indent=2, sort_keys=True))
        if args.gcs_uri:
            uploaded = [upload_file_to_gcs(path, args.gcs_uri) for path in written_files]
            print("uploaded grouped ScalarLM split files:")
            for uri in uploaded:
                print(uri)
        return

    if args.command == "daily-status":
        repository = _repository_for_backend(args.backend, args.db)
        try:
            repository_status = repository.operational_status()
        finally:
            repository.close()
        bigquery_status = None
        if args.include_bigquery:
            config = load_cloud_config_from_env()
            if config.bigquery is None:
                raise ValueError("BigQuery env vars are not configured")
            bigquery_status = bigquery_table_freshness(config.bigquery)
        report = build_operational_status_report(
            repository_status,
            bigquery_status=bigquery_status,
            max_snapshot_age_hours=args.max_snapshot_age_hours,
            max_bigquery_age_hours=args.max_bigquery_age_hours,
        )
        print(json.dumps(report, default=str, indent=2, sort_keys=True))
        if report["warnings"]:
            codes = ",".join(warning["code"] for warning in report["warnings"])
            print(f"MURPHY_DAILY_STATUS_WARNINGS count={len(report['warnings'])} codes={codes}")
        return

    if args.command == "init-cloud":
        config = load_cloud_config_from_env()
        did_anything = False
        if args.cloud_sql:
            if config.cloud_sql is None:
                raise ValueError("Cloud SQL env vars are not configured")
            initialize_cloud_sql(config.cloud_sql)
            print("initialized Cloud SQL schema")
            did_anything = True
        if args.bigquery:
            if config.bigquery is None:
                raise ValueError("BigQuery env vars are not configured")
            initialize_bigquery_dataset(config.bigquery)
            print("initialized BigQuery dataset/tables")
            did_anything = True
        if not did_anything:
            raise ValueError("Choose --cloud-sql and/or --bigquery")
        return

    if args.command == "export-cloud-sql-to-duckdb":
        config = load_cloud_config_from_env()
        if config.cloud_sql is None:
            raise ValueError("Cloud SQL env vars are not configured")
        counts = export_cloud_sql_to_duckdb(config.cloud_sql, args.duckdb)
        for table, count in counts.items():
            print(f"{table}: {count}")
        return

    if args.command == "mirror-cloud-sql-to-bigquery":
        config = load_cloud_config_from_env()
        if config.cloud_sql is None:
            raise ValueError("Cloud SQL env vars are not configured")
        if config.bigquery is None:
            raise ValueError("BigQuery env vars are not configured")
        counts = mirror_cloud_sql_to_bigquery(
            config.cloud_sql,
            config.bigquery,
            tables=args.tables,
            write_disposition="WRITE_APPEND" if args.append else "WRITE_TRUNCATE",
        )
        for table, count in counts.items():
            print(f"{table}: {count}")
        return

    if args.command == "ingest-existing-options":
        count = ingest_examples_from_existing_options_db(
            source_db=args.source_db,
            target_db=args.target_db,
            max_abs_moneyness=args.max_abs_moneyness,
            min_dte=args.min_dte,
            max_dte=args.max_dte,
            limit=args.limit,
        )
        print(f"ingested {count} option examples into {args.target_db}")
        return

    if args.command == "materialize-features":
        count = materialize_features_from_existing_options_db(
            source_db=args.source_db,
            target_db=args.target_db,
            replace=not args.no_replace,
        )
        print(f"materialized {count} feature rows into {args.target_db}")
        return

    if args.command == "run-baselines":
        results = run_baselines(
            db_path=args.db,
            test_fraction=args.test_fraction,
            calibration_fraction=args.calibration_fraction,
            persist=not args.no_persist,
        )
        for result in results:
            print(
                f"{result.method}: n_train={result.n_train} n_test={result.n_test} "
                f"brier={result.brier_score:.4f} log_loss={result.log_loss:.4f} "
                f"ece={result.calibration_error:.4f} test_pos_rate={result.positive_rate:.4f}"
            )
        return

    raise AssertionError(f"unhandled command {args.command}")


def _repository_for_backend(backend: str, db_path: str):
    if backend == "duckdb":
        return DuckDbRepository(db_path)
    if backend == "cloud-sql":
        config = load_cloud_config_from_env()
        if config.cloud_sql is None:
            raise ValueError("Cloud SQL env vars are not configured")
        return CloudSqlRepository(config.cloud_sql)
    raise ValueError(f"Unsupported backend: {backend}")


def _collect_into_repository(
    repository,
    provider_name: str,
    tickers: list[str],
    min_dte: int,
    max_dte: int,
    lookback_days: int,
):
    provider = provider_from_name(provider_name)
    bars, snapshots, result = collect_market_snapshots(
        provider=provider,
        tickers=tickers,
        min_dte=min_dte,
        max_dte=max_dte,
        lookback_days=lookback_days,
    )
    repository.record_external_call(
        provider=result.provider,
        call_type="market_snapshot",
        request_payload={
            "tickers": sorted({ticker.upper() for ticker in tickers}),
            "min_dte": min_dte,
            "max_dte": max_dte,
            "lookback_days": lookback_days,
        },
        response_payload={
            "underlying_bars": [bar.model_dump(mode="json") for bar in bars],
            "option_snapshots": [snapshot.model_dump(mode="json") for snapshot in snapshots],
            "counts": {
                "underlying_bars": len(bars),
                "option_snapshots": len(snapshots),
            },
        },
        captured_at=result.collected_at,
        information_cutoff=result.collected_at,
        source_timestamp=result.collected_at,
    )
    bar_count = repository.insert_underlying_bars(bars)
    snapshot_count = repository.insert_option_snapshots(snapshots)
    repository.record_collection_result(result)
    return bar_count, snapshot_count, result


def _collect_resolution_bars_into_repository(
    repository,
    provider_name: str,
    tickers: list[str],
    lookback_days: int,
) -> int:
    provider = provider_from_name(provider_name)
    normalized_tickers = sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})
    bars = provider.fetch_underlying_bars(
        normalized_tickers,
        lookback_days=lookback_days,
    )
    captured_at = max((bar.timestamp for bar in bars), default=None)
    repository.record_external_call(
        provider=provider.name,
        call_type="resolution_underlying_bars",
        request_payload={
            "tickers": normalized_tickers,
            "lookback_days": lookback_days,
        },
        response_payload={
            "underlying_bars": [bar.model_dump(mode="json") for bar in bars],
            "counts": {"underlying_bars": len(bars)},
        },
        captured_at=captured_at,
        information_cutoff=captured_at,
        source_timestamp=captured_at,
    )
    return repository.insert_underlying_bars(bars)


def _collect_news_context_into_repository(
    repository,
    tickers: list[str],
    information_cutoff,
    limit_per_ticker: int,
) -> int:
    provider = YahooFinanceNewsProvider()
    normalized_tickers = sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})
    items = provider.fetch(normalized_tickers, limit_per_ticker=limit_per_ticker)
    repository.record_external_call(
        provider=provider.name,
        call_type="web_news_context",
        request_payload={
            "tickers": normalized_tickers,
            "limit_per_ticker": limit_per_ticker,
        },
        response_payload=news_items_payload(items),
        captured_at=datetime.now(UTC),
        information_cutoff=information_cutoff,
        source_timestamp=information_cutoff,
    )
    return len(items)


if __name__ == "__main__":
    main()
