# Data Contracts

## Config

- `ExperimentConfig` is a dataclass tree loaded from YAML.
- Paths resolve from repo root when the config lives under `configs/`, otherwise from the config file parent.

## MarketPanel

- Long-format pandas frame.
- Sorted by `symbol`, then `timestamp`.
- Required columns:
  - `symbol`
  - `timestamp`
  - `open`
  - `high`
  - `low`
  - `close`
  - `volume`
  - `adj_close`
  - `adj_factor`
  - `adj_open`
  - `adj_high`
  - `adj_low`

## WeightsFrame

- Columns:
  - `strategy`
  - `effective_date`
  - `symbol`
  - `weight`
- Effective weights apply at the next market open.
- Emit a full symbol set on rebalance dates when a strategy must explicitly zero positions.

## PerformanceFrame

- Columns:
  - `date`
  - `strategy`
  - `gross_return`
  - `net_return`
  - `turnover`
  - `equity`

## Artifacts

- Metrics CSV
- Performance CSV
- Markdown report
- Cumulative return plot
- Drawdown plot
