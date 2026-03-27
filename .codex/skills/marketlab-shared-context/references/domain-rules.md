# Domain Rules

- Use `yfinance` as the only external market-data provider for Sprint 1.
- Prefer prepared panel cache if available.
- Build weekly signals from the last market close in each `W-FRI` period.
- Execute rebalances on the next market open.
- Adjust open/high/low using `adj_factor = adj_close / close`.
- Apply costs as `bps_per_trade * sum(abs(weight_change)) / 10000`.
- Treat missing exposure as cash earning zero.

## Strategy Semantics

- `buy_hold` is long-only equal weight.
- `sma` is long-only equal weight across symbols whose fast moving average exceeds the slow moving average on the weekly signal date.
