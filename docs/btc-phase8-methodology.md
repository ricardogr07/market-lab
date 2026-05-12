# BTC Phase 8 Methodology

This document is the canonical reference for the BTC Phase 8 research lane. The
goal is not to predict every BTC bar. The goal is to find a repeatable,
cost-aware allocation rule that can beat BTC buy-and-hold across multiple
regimes before any isolated Phase 9 paper deployment is allowed.

## 1. Data And Timing

The Phase 8 BTC config uses one BTC symbol on 4h bars. Local research prefers a
prepared panel cache when it exists because Yahoo/yfinance can reject long
historical intraday windows. The canonical panel columns are the MarketLab
panel contract: timestamp, open/high/low/close, volume, adjusted close, and
adjusted open/high/low.

Signals are formed only from completed bars. A signal at bar close maps to a
target weight on the next available bar, so the backtest does not trade on a
price that was used to form the same signal. Costs are applied as:

```text
cost_return = bps_per_trade * sum(abs(weight_change)) / 10000
```

Missing exposure is cash with zero return. BTC buy-and-hold stays fully long
through the out-of-sample window. Phase 8 includes two partial BTC/cash
benchmark families at 25%, 50%, and 75% BTC exposure:

- Static benchmarks set the initial BTC target weight and then allow exposure
  to drift with BTC performance.
- Constant-rebalanced benchmarks reset BTC exposure on each configured
  rebalance bar, so exposure comparisons include realistic turnover and cost
  drag.

These benchmarks prevent a mostly-cash strategy from passing only by avoiding a
BTC drawdown.

## 2. Features

BTC Phase 8 combines standard MarketLab features, crypto time-series features,
indicator-stack features, and regime features.

The crypto feature set includes returns, realized volatility, moving-average
distance and slope, RSI, MACD, Bollinger position, volume state, hour-of-day,
and day-of-week. Regime features add trend state, realized-volatility
percentile, drawdown from rolling high, volume shock, and a risk-off flag.

All features are computed from information available at or before the completed
signal bar. Forward returns are never used as model inputs.

## 3. Target Label

The original target is directional. For each signal row, MarketLab finds the next
effective bar, then measures BTC forward return over the configured horizon. In
the BTC Phase 8 config, `target.horizon_days: 6` is interpreted by the current
intraday target code as six 4h bars.

The model label is:

```text
target = 1 if forward_return > 0 else 0
```

The model score is the estimated probability that the forward return will be
positive:

```text
score = P(target = 1 | past features)
```

The allocation-utility research path uses a different supervised target:

```yaml
target:
  horizon_days: 14
  type: "allocation_utility"
```

For each row, MarketLab evaluates the future utility of 0%, 25%, 50%, and 100%
BTC over the horizon. Utility combines forward return, forward drawdown,
forward realized volatility, and an entry-cost penalty. Return and entry cost
stay linear in exposure, while risk uses a convex exposure penalty so partial
tiers can be the best label:

```text
utility =
  weight * forward_return
  - drawdown_penalty * weight^risk_penalty_power * abs(forward_drawdown)
  - volatility_penalty * weight^risk_penalty_power * forward_realized_volatility
  - entry_cost * weight
```

The daily long-history allocation-utility config uses
`allocation_utility_risk_penalty_power: 2.0`. The selected class is the tier
with the highest utility, with ties broken toward lower exposure. The
model score is expected allocation:

```text
score = sum(prob_tier * tier_weight)
```

## 4. Walk-Forward Training

The outer walk-forward split defines the real out-of-sample test windows. Each
fold trains only on rows whose target end date is known before the fold label
cutoff. The embargo removes recent rows that could leak overlapping forward
returns into training.

Inside each outer fold, the tuning step holds out a validation tail from the
training slice. Candidate models are selected on that validation tail only. The
selected candidate is then refit for the outer test fold using the selected
rolling training window inside the label-safe train slice.

Rolling training candidates make the lookback explicit:

```yaml
rolling_train_bars_grid: [540, 1095, 1620]
```

For each value, the candidate trains only on the latest N eligible signal bars.
This lets the research compare shorter adaptive memory against longer regime
history without using future data.

## 5. Models And Explainability

BTC Phase 8 currently evaluates the configured classification models:
logistic regression, L1 logistic regression, random forest, extra trees,
gradient boosting, and histogram gradient boosting.

The directional model output used for trading is the probability score, not a
raw model class prediction. The allocation-utility model output is expected
allocation from the predicted class probabilities. Phase 8 now tunes named
allocation-utility profiles inside the validation slice, applies training-slice
class weighting for rare partial labels, and uses sigmoid calibration only when
the current training slice has enough class support.

Target labels alone are not enough. A run can create 25% and 50% labels but
still predict mostly cash or full BTC. The predicted-tier support gate checks
the selected OOS predictions and blocks Phase 9 unless the traded expected
allocation actually uses the configured partial tiers.

Logistic models are explainable through signed feature coefficients after
scaling. Tree models are explainable through feature importance and through
fold-level diagnostics such as score histograms, calibration bins, threshold
diagnostics, validation candidate results, utility profile selection, class
weighting mode, and calibration status.

The research artifacts to inspect are:

- `model_summary.csv` and `fold_summary.csv` for predictive diagnostics.
- `calibration_diagnostics.csv` and `score_histograms.csv` for score behavior.
- `threshold_diagnostics.csv` for forward returns by probability threshold.
- `ml_strategy_tuning_candidates.csv` for model, threshold, hold, hysteresis,
  turnover, and validation outcome.
- `ml_strategy_tuning_selections.csv` for the selected candidate per fold.
- `allocation_target_diagnostics.csv` for utility-label distributions by fold
  and regime.
- `allocation_probability_diagnostics.csv` for predicted tier probabilities,
  nearest traded tier, predicted partial-tier support, expected allocation,
  realized utility, and realized return.
- `feature_importance.csv` for selected-fold coefficients or tree importances.
- `phase8_run_summary.csv` for the deterministic post-run summary of failed
  strict-gate rows, selected-fold coverage, candidate rejection reasons,
  target-tier support, predicted-tier support, benchmark deltas, and
  regime-slice active returns.

## 6. Long Or Cash Decision

In directional `tiered` mode, the selected model score maps to one of four
allowed BTC exposures:

```text
score < min_threshold       -> 0%
score < half_threshold      -> 25%
score < full_threshold      -> 50%
score >= full_threshold     -> 100%
```

The default tier grid is tuned from configured threshold sets such as:

```yaml
tier_threshold_sets:
  - [0.50, 0.55, 0.62]
  - [0.52, 0.58, 0.65]
  - [0.55, 0.60, 0.68]
```

If `crypto_regime_risk_off` is active, positive exposure is capped at 25%.
Risk-off de-risking can reduce exposure immediately even during a holding
period.

In `direct_tiered` allocation-utility mode, expected allocation is mapped to
the nearest allowed tier directly. The same risk-off cap, minimum hold,
hysteresis, turnover budget, validation-only candidate selection, and strict
gate then apply.

Candidate selection for the allocation-utility path is benchmark-relative. A
candidate must beat every configured validation benchmark, currently
`buy_hold`, `btc_rebalanced_25`, `btc_rebalanced_50`, and `btc_rebalanced_75`,
using validation-window net cumulative return only. Selection then maximizes
the weakest active return across those benchmarks, with buy-hold excess return,
drawdown delta, Sharpe-like delta, and lower turnover used as tie-breakers.

Minimum holding periods reduce churn. When a minimum hold is active, the
allocator keeps the prior tier until enough signal bars have elapsed since the
last change. The exception is an immediate risk-off move to a lower exposure.

The current longer-hold hysteresis experiments use:

```yaml
min_holding_period_bars_grid: [12, 18, 24, 36, 54]
hysteresis_margin_grid: [0.0, 0.02, 0.04, 0.06]
```

Hysteresis makes tier changes harder than tier retention. If the current tier is
50% and the full-exposure threshold is 0.62, a 0.03 margin requires a score of
at least 0.65 to increase to 100%. If the current tier is 100%, the same margin
allows the model to keep 100% until the score falls below 0.59. This reduces
probability-threshold whipsaw without allowing the LLM or runtime to resize
trades.

The validation-tuned regime participation policies are candidate parameters,
not paper-trading overrides. They only set minimum exposure floors for the
current completed-bar regime and retain the 25% risk-off cap:

```yaml
regime_participation_policies:
  - name: "model_only"
    bull_floor: 0.0
    sideways_floor: 0.0
    bear_floor: 0.0
    risk_off_cap: 0.25
  - name: "bull50_sideways25"
    bull_floor: 0.50
    sideways_floor: 0.25
    bear_floor: 0.0
    risk_off_cap: 0.25
  - name: "bull100_sideways25"
    bull_floor: 1.0
    sideways_floor: 0.25
    bear_floor: 0.0
    risk_off_cap: 0.25
  - name: "bull100_sideways50_bear25"
    bull_floor: 1.0
    sideways_floor: 0.50
    bear_floor: 0.25
    risk_off_cap: 0.25
```

Allowed strategy target tiers remain `0%`, `25%`, `50%`, and `100%`. The `75%`
exposure level exists only in the static and rebalanced BTC/cash benchmark
families.

Lower-cadence BTC configs reuse the same strict gate with bars that represent
larger decision intervals:

- `btc_phase8_regime_allocation_12h_hysteresis`: every 12h bar, 730 periods per
  year, 2-bar target horizon.
- `btc_phase8_regime_allocation_1d_hysteresis`: every 1d bar, 365 periods per
  year, 1-bar target horizon.
- `btc_phase8_regime_allocation_1d_long_history`: daily BTC-USD from
  2015-01-01 through 2026-05-07, using 3-year train windows and 6-month OOS
  windows.
- `btc_phase8_allocation_utility_1d_long_history`: the same daily long-history
  panel with direct tier labels over a 14-bar allocation-utility horizon.

## 7. Cost-Aware Candidate Selection

Candidate selection is portfolio-based, not just forecast-based. Each candidate
combines:

- model family
- rolling training window
- tier thresholds
- minimum holding period
- hysteresis margin

The candidate is backtested on the validation tail after configured BTC trading
costs. Net excess return versus buy-and-hold is the primary validation
objective. Drawdown improvement, Sharpe-like improvement, and lower annualized
turnover are tie-breakers. Candidate selection does not use outer test-fold
returns or outer test-fold exposure.

The configured annualized turnover budget is:

```yaml
max_annualized_turnover: 24.0
```

This budget rejects candidates that can only work by changing exposure too
often. It also protects the strict gate from accepting an overfit strategy that
looks acceptable before realistic BTC cost drag.

After a run, regenerate the deterministic summary without training models:

```bash
python scripts/run_marketlab.py phase8-summary --run-dir artifacts/runs/<experiment>/<run-id>
```

The same summary is embedded into `report.md` when Phase 8 artifacts are
generated.

## 8. Strict Research Gate

Phase 8 passes only if the final selected strategy passes all strict gate rows:

- Net cumulative return beats BTC buy-and-hold over the full OOS window.
- Net cumulative return also beats static BTC/cash benchmarks at 25%, 50%, and
  75% BTC exposure.
- Net cumulative return also beats constant-rebalanced BTC/cash benchmarks at
  25%, 50%, and 75% BTC exposure.
- Sharpe-like score matches or improves versus buy-and-hold.
- Max drawdown matches or improves versus buy-and-hold.
- Average exposure stays between the configured lower and upper bounds.
- Allocation-utility target labels have meaningful support for the required
  partial tiers. The current balanced gate requires 25% and 50% labels to each
  represent at least 5% of all labels and to appear in at least 60% of
  train/validation folds.
- Selected OOS predictions have meaningful partial-tier support. The current
  predicted gate requires 25% and 50% traded tiers to each represent at least
  3% of selected OOS predictions and to appear in at least 50% of selected
  folds.
- Net active return passes the 35 bps cost gate versus every required
  benchmark.
- The strategy remains acceptable at 50 bps versus every required benchmark.
- Active return is positive in enough regime slices.
- At least 75% of walk-forward folds select a valid candidate.
- Annualized turnover stays under the configured turnover budget.

If any row fails, Phase 9 remains blocked. In that case BTC can still produce
signals and reports, but the model is not approved for isolated paper trading.

## 9. Regime-State Objective Candidate

Regime-state is now implemented as a separate Phase 8 research config, not as a
replacement for the allocation-utility path. It derives utility tiers first,
then maps `0% -> risk_off`, `25%/50% -> reduced`, and `100% -> risk_on`.
State probabilities are converted back into expected allocation with
`risk_off=0%`, `reduced=50%`, and `risk_on=100%`; the existing direct-tier
mapping can still trade `0/25/50/100%` after risk caps, holding periods, and
hysteresis. This keeps the state objective interpretable while preserving the
same strict Phase 8 gate.

## 10. Phase 9 Boundary

Phase 9 is an isolated paper experiment only after Phase 8 passes. It uses a
separate Alpaca paper account, separate `.env.btc-paper`, separate LLM keys,
separate Telegram credentials, separate containers, and separate artifact
paths. The LLM may approve or reject a persisted BTC proposal, but it cannot
invent trades or resize the target allocation.
