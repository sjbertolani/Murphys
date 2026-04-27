# Market Data Provider Options

We need live or near-live option-chain snapshots with enough fields to generate and resolve questions:

- underlying ticker
- quote timestamp
- expiration
- strike
- call/put
- bid/ask/mid or last
- implied volatility
- delta/greeks if available
- volume and open interest
- underlying spot or a reliable equity quote

## Practical Choices

### Tradier

Pros:

- Options-first brokerage API.
- Official docs say real-time equities/options data is available to Tradier Brokerage account holders.
- Option Greeks and volatility are included courtesy of ORATS, with Greeks updated hourly.

Cons:

- Real-time data requires a Tradier Brokerage account.
- Sandbox data is delayed.
- Best if we are comfortable creating/using a brokerage account.

Fit: good low-friction choice if you want brokerage-linked real-time data and possibly later paper/live trading integration.

### ThetaData

Pros:

- Retail-friendly options data pricing.
- Pricing page lists real-time access, unlimited requests, low-latency snapshots, and option-chain snapshots on the Standard options plan.
- Better fit for collecting snapshots several times daily without a brokerage account.

Cons:

- Need to verify deployment shape; some ThetaData workflows use a local terminal/client.
- Provider-specific adapter work required.

Fit: probably the best first research provider if we want live chain collection without tying to a broker.

### ORATS

Pros:

- Rich option-specific data, cleaned/smoothed values, IV metrics, historical volatility, earnings, and proprietary indicators.
- Delayed, live, and intraday tiers.

Cons:

- More expensive than ThetaData for live/intraday.
- May be more data than the first iteration needs.

Fit: strong if we want better engineered option features and are willing to pay for them.

### Polygon

Pros:

- Very polished APIs and infrastructure.
- Options docs describe broad U.S. options coverage, quotes, trades, snapshots, Greeks, IV, and open interest.

Cons:

- Business options plans are expensive.
- Individual options pricing/entitlements need careful checking before choosing it.

Fit: great API ergonomics, but likely not the cheapest starting point for options.

## Recommendation

For this project, I would choose in this order:

1. **Yahoo/yfinance now** so we can begin collecting a no-leakage live dataset immediately.
2. **Alpha Vantage** if we want a low-cost API key with official options endpoints and historical chains.
3. **ThetaData Standard options plan** if we want an independent market-data feed for research.
4. **Tradier** if you are willing to use a brokerage account and may want trading integration later.
5. **ORATS** if richer option features matter more than cost.
6. **Polygon** if budget is not the constraint and API polish matters most.

The first deployed collector uses Yahoo/yfinance. The tradeoff is that Yahoo is not an official
SLA-backed API, so every snapshot used for forecasting must be persisted. If Yahoo changes behavior
or rate-limits us, we can swap providers without changing the database schema.

For implementation, the collector should be provider-abstracted:

```text
MarketDataProvider
  fetch_underlying_bars(tickers, start, end)
  fetch_option_chain_snapshot(ticker, timestamp)
```

That keeps the live forecasting database stable even if we swap providers.
