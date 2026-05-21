# Phase 8 Crypto Hourly Trend Signals

Phase 8 adds a research-first lab for checking whether common crypto trading
claims are technically sound against `buy_hold`. The first tracked surface is
`BTC-USD` on hourly bars, with a focused visual inspection path for sub-hour
bars.

This is not a paper-trading or live-money feature. It exists to falsify or
support explicit indicator rules before any execution loop is designed.

## Tracked Config

The tracked config is:

- `configs/experiment.crypto_hourly_trend.yaml`
- `configs/experiment.crypto_15m_signal_week.yaml`
- `configs/experiment.crypto_15m_patterns_week.yaml`
- `configs/experiment.crypto_15m_patterns_day.yaml`
- `configs/experiment.crypto_5m_patterns_day.yaml`
- `configs/experiment.crypto_pattern_exit_tuned_2024_ytd.yaml`
- `configs/experiment.crypto_pattern_meta_label_2024_ytd.yaml`
- `configs/experiment.crypto_pattern_meta_tuned_2024_ytd.yaml`
- `configs/experiment.crypto_ts_ml_2024_ytd.yaml`
- `configs/experiment.crypto_indicator_ml_tuned_6h_2024_ytd.yaml`
- `configs/experiment.crypto_indicator_ml_tuned_12h_2024_ytd.yaml`
- `configs/experiment.crypto_indicator_ml_tuned_24h_2024_ytd.yaml`

It pins the first research path to:

- `data.symbols: [BTC-USD]`
- `data.interval: "1h"`
- `portfolio.ranking.rebalance_frequency: "bar"`
- `evaluation.periods_per_year: 8760`
- `evaluation.benchmark_strategy: "buy_hold"`
- `paper.enabled: false`

## Binance-Style Fee Baseline

The backtest engine applies costs as:

```text
cost_return = turnover * (bps_per_trade / 10_000)
```

For a long/cash strategy, entering from cash has turnover `1.0` and exiting to
cash has turnover `1.0`. Binance's current Spot/Margin fee table shows regular
users at `0.100% / 0.100%` maker/taker, and `0.07500% / 0.07500%` when the BNB
25% fee discount is applied.

That maps to:

- `10.0` bps per turnover unit for regular spot maker/taker.
- `7.5` bps per turnover unit for the BNB-discounted regular tier.
- a buy-then-sell round trip costs about `20` bps at the regular spot tier
  because it is two turnover events.

The crypto research configs therefore use `portfolio.costs.bps_per_trade: 10`
as the baseline and include `7.5`, `20`, `40`, and `75` bps in sensitivity
analysis. Higher rows are stress tests for slippage, spreads, worse routing, or
non-spot execution assumptions; they are not the baseline Binance spot fee.

Reference: [Binance Spot Trading Fee Rate](https://www.binance.com/en/fee/trading).

The 15m signal-week config adds:

- `data.interval: "15m"`
- `evaluation.focus_start` and `evaluation.focus_end`
- `evaluation.visualize_signals: true`

Supported intraday research intervals are `1h`, `45m`, `30m`, `15m`, `5m`,
and `1m`.
Each interval has a crypto 24/7 annualization default, so the config can omit
`evaluation.periods_per_year` unless a run needs an explicit override.

## Indicator Stack

The indicator-stack baseline emits a normal `WeightsFrame`, so it uses the
existing backtest, analytics, benchmark-relative, plotting, and Markdown report
surfaces.

The first rule family is explicit and deterministic:

- EMA fast above EMA slow
- RSI inside a configured range
- MACD above its signal line
- Bollinger breakout or mean-reversion confirmation
- volume above a rolling average threshold
- optional rolling VWAP confirmation

The strategy is long/cash only. A bar is long when the configured minimum number
of confirmations is met at the signal bar close. The weight becomes effective at
the next available bar open.

This maps the IG strategy taxonomy into deterministic tests: HODL is
`buy_hold`, trend trading is moving-average/RSI momentum, and breakout trading
uses Bollinger, volume, RSI, and MACD confirmations. Hedging stays out of scope
because this phase is long/cash research, not short or derivatives execution.

Reference: [IG bitcoin trading strategies and tips](https://www.ig.com/en-ch/trading-strategies/best-bitcoin-trading-strategies-and-tips-190813).

## Chart-Pattern Stack

The chart-pattern baseline is a separate long/cash strategy named
`chart_patterns`. It exists to test common chart-pattern claims without
subjective drawing or hindsight. Every pattern uses trailing bars only, signals
at completed bar close, and becomes tradable at the next available bar.

The deterministic pattern stack covers the repeatedly cited patterns from
Changelly, Binance Square, TradingView, and BeInCrypto:

- ascending triangle breakout
- descending triangle breakdown
- symmetrical triangle breakout
- bullish and bearish rectangle breakouts
- head-and-shoulders breakdown
- inverse head-and-shoulders breakout
- double bottom breakout
- double top breakdown
- triple bottom breakout
- triple top breakdown
- falling wedge breakout
- rising wedge breakdown
- bull flag breakout
- bear flag breakdown
- pennant breakout
- cup-and-handle breakout
- ascending channel continuation
- descending channel breakdown
- megaphone breakout
- diamond breakdown

Long entries require at least the configured number of bullish pattern
detections and zero bearish detections on the same signal bar. Bearish patterns
therefore act as cash gates in Phase 8 because this research track is long/cash,
not short or derivatives execution.

For review, the repo also includes a synthetic 20-pattern gallery generator:

```bash
PYTHONPATH=src python scripts/generate_pattern_gallery.py
```

The generator writes:

- `artifacts/pattern-gallery/synthetic_pattern_gallery.csv`
- `artifacts/pattern-gallery/synthetic_pattern_gallery.png`

Green gallery lines represent patterns that map to executable `chart_patterns`
detector columns.

References:

- [Changelly crypto chart patterns](https://changelly.com/blog/crypto-chart-patterns/)
- [Binance Square chart patterns](https://www.binance.com/en/square/post/24738155290281)
- [TradingView top key patterns cheat sheet](https://www.tradingview.com/chart/BTCUSDT/6puZ59qW-TOP-20-Key-Patterns-cheat-sheet/)
- [BeInCrypto crypto trading patterns](https://beincrypto.com/learn/crypto-trading-patterns/)

## Timing Rules

For `rebalance_frequency: "bar"`:

- every completed bar can be a signal bar
- execution starts at the next available bar
- the final bar cannot create a new effective allocation because no future bar
  exists

This preserves the existing daily ETF semantics for `D`, `W-FRI`, and other
pandas period frequencies.

## Visual Inspection

When `evaluation.visualize_signals` is true and `indicator_stack` is enabled,
the backtest report persists:

- `indicator_diagnostics.csv`
- `signal_price_overlay.png`
- `signal_confirmations.png`
- `signal_performance_focus.png`

The diagnostics file records the signal timestamp, effective next-bar date,
symbol, close, EMA values, RSI, MACD values, Bollinger bands, rolling volume
average, VWAP, confirmation booleans, confirmation count, and target weight.
The plots are static report artifacts for a configured focus window; they are
not a dashboard or execution tool.

When `chart_patterns` is enabled, the report also persists:

- `pattern_diagnostics.csv`
- `pattern_price_overlay.png`
- `pattern_detections.png`
- `pattern_detection_windows.png`
- `pattern_performance_focus.png`

The day-level config keeps the full-week data window for detector lookback but
focuses visual inspection on `2026-04-22`. The detection-window plot shows the
nearby 15m bars around each hit, marks the signal bar, and names the exact
pattern columns that fired. The 5m day-level config uses the same focus date
with smaller bars and a scaled lookback window for finer-grained inspection.

For interval meta-analysis across the same week, run:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_crypto_pattern_interval_meta.py
```

This writes `artifacts/runs/crypto_pattern_interval_meta/interval_meta_summary.csv`
and `interval_meta_report.md` for `1m`, `5m`, `15m`, `30m`, `45m`, and `1h`.
The `1m` and `5m` panels use native provider data when available; the wider
intraday panels are deterministic OHLCV resamples from the native 5m panel.
For a different window, pass dates and an output directory explicitly:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_crypto_pattern_interval_meta.py `
  --start-date 2026-04-01 `
  --end-date 2026-05-01 `
  --output-root artifacts/runs/crypto_pattern_interval_meta_april `
  --experiment-suffix april_meta `
  --intervals 1m 5m 15m 30m 45m 1h
```

The `1m` path downloads in seven-day chunks because the upstream provider
limits one-minute history per request. If the requested end date is beyond the
provider's latest available bar, the generated panel stops at the latest bar
returned by the provider.
For lower-turnover exploration, use a slower grid such as:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_crypto_pattern_interval_meta.py `
  --start-date 2026-04-01 `
  --end-date 2026-05-01 `
  --output-root artifacts/runs/crypto_pattern_interval_meta_april_slow `
  --experiment-suffix april_slow_meta `
  --intervals 1h 2h 4h 6h 12h
```

When all requested intervals are `1h` or slower, the script uses native `1h`
data as the base panel before resampling. This avoids relying on short-retention
`5m` data for longer YTD-style windows.
For multi-year crypto runs that exceed Yahoo intraday retention, use the Binance
public klines source:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_crypto_pattern_interval_meta.py `
  --source binance `
  --start-date 2024-01-01 `
  --end-date 2026-05-01 `
  --output-root artifacts/runs/crypto_pattern_interval_meta_2024_ytd_slow `
  --experiment-suffix 2024_ytd_slow_meta `
  --intervals 1h 2h 4h 8h 12h 1d
```

For the cost-aware exit overlay, the tracked tuned config uses a high-confidence
meta-label gate and reports a threshold sweep:

```powershell
$env:PYTHONPATH = "src"
python -m marketlab.cli backtest --config configs/experiment.crypto_pattern_exit_tuned_2024_ytd.yaml
```

The sweep includes net return, drawdown, turnover, cost drag, exit count, cash
bar count, and average exposure. Treat high thresholds carefully: rows with very
few exits can look close to buy-and-hold because the overlay mostly abstained,
not because the chart-pattern evidence became stronger.

For nested meta-label tuning and partial-exposure research, run:

```powershell
$env:PYTHONPATH = "src"
python -m marketlab.cli backtest --config configs/experiment.crypto_pattern_meta_tuned_2024_ytd.yaml
```

This config writes `pattern_meta_tuning_candidates.csv`,
`pattern_meta_tuning_selections.csv`, `pattern_partial_exposure_diagnostics.csv`,
and `pattern_partial_threshold_sweep.csv`. It remains research-only and keeps
paper execution disabled. Short-capable pattern strategies are intentionally
deferred until funding, borrow, slippage, and execution-cost assumptions are
defined.

## Phase 8.6 Time-Series ML

Phase 8.6 adds a direct hourly time-series ML lane that predicts bar-level
long/cash exposure for `BTC-USD` instead of only filtering rare bearish pattern
exit candidates. It remains research-only and uses trailing numeric features:
lagged returns, volatility, moving-average ratios and slopes, RSI, MACD,
Bollinger z-score, volume z-score, and UTC hour/day cyclic features.

Run the tracked comparison with:

```powershell
$env:PYTHONPATH = "src"
python -m marketlab.cli run-experiment --config configs/experiment.crypto_ts_ml_2024_ytd.yaml
```

The run writes `ml_strategy_threshold_sweep.csv` alongside the regular
`strategy_summary.csv`, `benchmark_relative.csv`, and `report.md`. The sweep
tests each configured ML model across `[0.50, 0.52, 0.55, 0.58, 0.60]` and
marks a pass only when the strategy beats `buy_hold` after costs, has at least
five exposure changes, and has average exposure below `0.995`. Pattern/meta
strategies stay in the same report as comparison baselines, not production
promotions.

## Phase 8.7 Indicator-ML Tuning

Phase 8.7 keeps the direct hourly ML lane research-only, but adds deterministic
indicator-stack features to the model input and tunes model/threshold choices
inside each walk-forward fold. The indicator features reuse the configured
`baselines.indicator_stack` parameters and include EMA spread, RSI, MACD
histogram, Bollinger position, volume state, confirmation booleans, and
confirmation count.

Run the 6h, 12h, and 24h target comparisons with:

```powershell
$env:PYTHONPATH = "src"
python -m marketlab.cli run-experiment --config configs/experiment.crypto_indicator_ml_tuned_6h_2024_ytd.yaml
python -m marketlab.cli run-experiment --config configs/experiment.crypto_indicator_ml_tuned_12h_2024_ytd.yaml
python -m marketlab.cli run-experiment --config configs/experiment.crypto_indicator_ml_tuned_24h_2024_ytd.yaml
```

Each run writes `ml_strategy_tuning_candidates.csv` and
`ml_strategy_tuning_selections.csv`. Candidate selection uses only the
validation tail inside each outer training fold; the selected model is then
refit on the full outer training fold before scoring the outer test fold. The
selected strategy is reported as `ml_indicator_tuned__long_only__cash`.

The Phase 8.7 gate is stricter than pure forecast accuracy: a candidate must
beat `buy_hold` after costs, improve either Sharpe-like score or max drawdown,
and satisfy the configured exposure-activity guardrails. A pass remains
research evidence, not permission to enable crypto paper trading.

## Acceptance Gate

The report includes a `Trend-Signal Acceptance Gate` section when
`indicator_stack` and `buy_hold` are both present.

The gate passes only when the indicator stack beats buy-and-hold on net
cumulative return and also improves either the Sharpe-like metric or max
drawdown. Until reviewed research runs pass that gate, crypto shadow-paper work
stays blocked.

## Shadow-Paper Boundary

Future crypto shadow-paper work must define provider and broker ports separately
from the existing Alpaca ETF paper services. The Phase 7 paper bot remains daily,
single-ETF, long/cash, and Alpaca-paper specific.

BTC paper is tracked separately from this generic crypto research lane. Its
config is `configs/experiment.btc_paper_daily.yaml`, its compose file is
`docker/compose.btc-paper.yml`, and it must remain isolated from the QQQ/VOO
paper inbox and artifacts.
