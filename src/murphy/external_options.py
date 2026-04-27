from __future__ import annotations

from pathlib import Path


DEFAULT_OPTIONS_DB = Path("/Users/steveb/Desktop/projects/options/data/options.db")

FEATURE_NAMES = [
    "ask",
    "bid",
    "dte",
    "implied_vol",
    "log_open_interest",
    "log_volume",
    "mid",
    "moneyness",
    "realized_vol_20",
    "spread_fraction",
    "spot",
    "trailing_return_5",
    "trailing_return_20",
]


def ingest_examples_from_existing_options_db(
    source_db: str | Path = DEFAULT_OPTIONS_DB,
    target_db: str | Path = "data/murphy.duckdb",
    max_abs_moneyness: float = 0.03,
    min_dte: int = 7,
    max_dte: int = 45,
    limit: int | None = None,
) -> int:
    """Create labeled call-option examples from the existing options project DB.

    Expected source schema comes from `/Users/steveb/Desktop/projects/options/data/db.py`:
    `prices(ticker, date, close, ...)` and
    `option_chains(ticker, snapshot_date, expiry, strike, flag, ... )`.

    Labels are `1` when the close at or before expiry is greater than strike.
    Spot is the close at or before the snapshot date. This keeps the join temporal.
    """
    source_path = Path(source_db)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    from murphy.db import MurphyDb

    target = MurphyDb(target_db)
    target.initialize()
    con = target.conn
    source_sql = str(source_path).replace("'", "''")
    con.execute(f"ATTACH '{source_sql}' AS source_options (READ_ONLY)")

    limit_sql = "" if limit is None else f"LIMIT {int(limit)}"
    query = f"""
    INSERT OR REPLACE INTO option_examples
    WITH calls AS (
      SELECT
        oc.ticker,
        oc.snapshot_date,
        oc.expiry,
        oc.strike,
        oc.bid,
        oc.ask,
        oc.last_price,
        oc.implied_vol,
        oc.volume,
        oc.open_interest,
        (
          SELECT p.close
          FROM source_options.prices p
          WHERE p.ticker = oc.ticker
            AND p.date <= oc.snapshot_date
          ORDER BY p.date DESC
          LIMIT 1
        ) AS spot,
        (
          SELECT p.close
          FROM source_options.prices p
          WHERE p.ticker = oc.ticker
            AND p.date <= oc.expiry
          ORDER BY p.date DESC
          LIMIT 1
        ) AS expiration_close
      FROM source_options.option_chains oc
      WHERE lower(oc.flag) = 'c'
    ),
    filtered AS (
      SELECT
        ticker,
        ticker || '-' || strftime(expiry, '%Y%m%d') || '-C-' || CAST(strike AS VARCHAR)
          AS option_symbol,
        CAST(snapshot_date AS TIMESTAMP) AS forecast_timestamp,
        CAST(expiry AS TIMESTAMP) AS expiration,
        strike,
        spot,
        date_diff('day', snapshot_date, expiry) AS dte,
        spot / strike - 1.0 AS moneyness,
        CASE WHEN expiration_close > strike THEN 1 ELSE 0 END AS label,
        CAST(expiry AS TIMESTAMP) AS resolution_timestamp
      FROM calls
      WHERE spot IS NOT NULL
        AND expiration_close IS NOT NULL
        AND strike > 0
        AND abs(spot / strike - 1.0) <= ?
        AND date_diff('day', snapshot_date, expiry) BETWEEN ? AND ?
      ORDER BY snapshot_date, ticker, expiry, strike
      {limit_sql}
    )
    SELECT
      ticker || ':' || option_symbol || ':' || strftime(forecast_timestamp, '%Y%m%dT%H%M%S')
        AS example_id,
      ticker AS symbol,
      option_symbol,
      forecast_timestamp,
      expiration,
      strike,
      spot,
      dte,
      moneyness,
      label,
      resolution_timestamp
    FROM filtered
    """
    before = con.execute("SELECT count(*) FROM option_examples").fetchone()[0]
    con.execute(query, [max_abs_moneyness, min_dte, max_dte])
    after = con.execute("SELECT count(*) FROM option_examples").fetchone()[0]
    con.execute("DETACH source_options")
    target.close()
    return int(after - before)


def summarize_existing_options_db(source_db: str | Path = DEFAULT_OPTIONS_DB) -> dict[str, int]:
    """Return table counts for the existing options DB."""
    import duckdb

    con = duckdb.connect(str(source_db), read_only=True)
    try:
        tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        return {
            table: con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in tables
        }
    finally:
        con.close()


def materialize_features_from_existing_options_db(
    source_db: str | Path = DEFAULT_OPTIONS_DB,
    target_db: str | Path = "data/murphy.duckdb",
    replace: bool = True,
) -> int:
    """Populate `features` for existing option examples using the source options DB."""
    source_path = Path(source_db)
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    from murphy.db import MurphyDb

    target = MurphyDb(target_db)
    target.initialize()
    con = target.conn
    source_sql = str(source_path).replace("'", "''")
    con.execute(f"ATTACH '{source_sql}' AS source_options (READ_ONLY)")

    if replace:
        placeholders = ", ".join(["?"] * len(FEATURE_NAMES))
        con.execute(
            f"DELETE FROM features WHERE feature_name IN ({placeholders})",
            FEATURE_NAMES,
        )

    query = """
    INSERT OR REPLACE INTO features
    WITH base AS (
      SELECT
        e.example_id,
        e.symbol,
        e.forecast_timestamp,
        e.expiration,
        e.strike,
        e.spot,
        e.dte,
        e.moneyness,
        oc.bid,
        oc.ask,
        oc.last_price,
        oc.implied_vol,
        oc.volume,
        oc.open_interest,
        (
          SELECT p.close
          FROM source_options.prices p
          WHERE p.ticker = e.symbol
            AND p.date < CAST(e.forecast_timestamp AS DATE)
          ORDER BY p.date DESC
          LIMIT 1 OFFSET 4
        ) AS close_5,
        (
          SELECT p.close
          FROM source_options.prices p
          WHERE p.ticker = e.symbol
            AND p.date < CAST(e.forecast_timestamp AS DATE)
          ORDER BY p.date DESC
          LIMIT 1 OFFSET 19
        ) AS close_20,
        (
          SELECT stddev_samp(ln(p.close / prev_close))
          FROM (
            SELECT
              p.close,
              lag(p.close) OVER (ORDER BY p.date) AS prev_close
            FROM source_options.prices p
            WHERE p.ticker = e.symbol
              AND p.date <= CAST(e.forecast_timestamp AS DATE)
            ORDER BY p.date DESC
            LIMIT 21
          ) p
          WHERE prev_close IS NOT NULL
            AND p.close > 0
            AND prev_close > 0
        ) * sqrt(252.0) AS realized_vol_20
      FROM option_examples e
      LEFT JOIN source_options.option_chains oc
        ON oc.ticker = e.symbol
       AND oc.snapshot_date = CAST(e.forecast_timestamp AS DATE)
       AND oc.expiry = CAST(e.expiration AS DATE)
       AND oc.strike = e.strike
       AND lower(oc.flag) = 'c'
      WHERE e.label IS NOT NULL
    ),
    wide AS (
      SELECT
        *,
        CASE
          WHEN bid IS NOT NULL AND ask IS NOT NULL THEN (bid + ask) / 2.0
          ELSE last_price
        END AS mid,
        CASE
          WHEN bid IS NOT NULL AND ask IS NOT NULL AND (bid + ask) > 0
            THEN (ask - bid) / ((bid + ask) / 2.0)
          ELSE NULL
        END AS spread_fraction,
        CASE WHEN close_5 > 0 THEN spot / close_5 - 1.0 ELSE NULL END AS trailing_return_5,
        CASE WHEN close_20 > 0 THEN spot / close_20 - 1.0 ELSE NULL END AS trailing_return_20,
        CASE WHEN volume IS NOT NULL THEN ln(1.0 + volume) ELSE NULL END AS log_volume,
        CASE WHEN open_interest IS NOT NULL THEN ln(1.0 + open_interest) ELSE NULL END
          AS log_open_interest
      FROM base
    ),
    feature_rows AS (
      SELECT example_id, 'ask' AS feature_name, ask AS feature_value, forecast_timestamp FROM wide
      UNION ALL SELECT example_id, 'bid', bid, forecast_timestamp FROM wide
      UNION ALL SELECT example_id, 'dte', dte, forecast_timestamp FROM wide
      UNION ALL SELECT example_id, 'implied_vol', implied_vol, forecast_timestamp FROM wide
      UNION ALL SELECT example_id, 'log_open_interest', log_open_interest, forecast_timestamp FROM wide
      UNION ALL SELECT example_id, 'log_volume', log_volume, forecast_timestamp FROM wide
      UNION ALL SELECT example_id, 'mid', mid, forecast_timestamp FROM wide
      UNION ALL SELECT example_id, 'moneyness', moneyness, forecast_timestamp FROM wide
      UNION ALL SELECT example_id, 'realized_vol_20', realized_vol_20, forecast_timestamp FROM wide
      UNION ALL SELECT example_id, 'spread_fraction', spread_fraction, forecast_timestamp FROM wide
      UNION ALL SELECT example_id, 'spot', spot, forecast_timestamp FROM wide
      UNION ALL SELECT example_id, 'trailing_return_5', trailing_return_5, forecast_timestamp FROM wide
      UNION ALL SELECT example_id, 'trailing_return_20', trailing_return_20, forecast_timestamp FROM wide
    )
    SELECT example_id, feature_name, feature_value, forecast_timestamp
    FROM feature_rows
    WHERE feature_value IS NOT NULL
      AND isfinite(feature_value)
    """
    before = con.execute("SELECT count(*) FROM features").fetchone()[0]
    con.execute(query)
    after = con.execute("SELECT count(*) FROM features").fetchone()[0]
    con.execute("DETACH source_options")
    target.close()
    return int(after - before)
