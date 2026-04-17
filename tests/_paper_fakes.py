from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from marketlab.config import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeaturesConfig,
    ModelSpec,
    PaperConfig,
    PaperNotificationsConfig,
    PortfolioConfig,
    RankingConfig,
    TargetConfig,
    TelegramNotificationsConfig,
    WalkForwardConfig,
)


def build_paper_history_frame(
    *,
    start_date: str = "2022-01-03",
    end_date: str = "2026-04-10",
) -> pd.DataFrame:
    trading_dates = pd.bdate_range(start_date, end_date)
    rows: list[dict[str, object]] = []
    close_price = 100.0

    for index, timestamp in enumerate(trading_dates):
        open_price = close_price
        cycle = index % 4
        if cycle in {0, 1}:
            close_price = open_price + 0.9
        else:
            close_price = max(10.0, open_price - 0.7)
        rows.append(
            {
                "Date": timestamp,
                "Open": round(open_price, 4),
                "High": round(max(open_price, close_price) + 0.2, 4),
                "Low": round(min(open_price, close_price) - 0.2, 4),
                "Close": round(close_price, 4),
                "Adj Close": round(close_price, 4),
                "Volume": 1_000_000 + index,
            }
        )

    return pd.DataFrame(rows)


def build_phase7_paper_config(
    base_dir: Path,
    *,
    execution_mode: str = "agent_approval",
    symbol: str = "VOO",
    agent_backend: str = "deterministic_consensus",
    agent_model: str = "",
    telegram_enabled: bool = False,
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="phase7_paper_fixture",
        base_dir=base_dir,
        data=DataConfig(
            symbols=[symbol],
            start_date="2022-01-03",
            end_date="2026-04-10",
            interval="1d",
            cache_dir="artifacts/data-voo-paper-daily",
            prepared_panel_filename="panel.csv",
        ),
        features=FeaturesConfig(
            return_windows=[5, 10, 20, 40],
            ma_windows=[10, 20, 50],
            vol_windows=[10, 20],
            momentum_window=20,
        ),
        target=TargetConfig(horizon_days=1, type="direction"),
        portfolio=PortfolioConfig(
            ranking=RankingConfig(
                long_n=1,
                short_n=1,
                rebalance_frequency="D",
                weighting="equal",
                mode="long_only",
                min_score_threshold=0.55,
                cash_when_underfilled=True,
            )
        ),
        models=[
            ModelSpec("logistic_regression"),
            ModelSpec("logistic_l1"),
            ModelSpec("random_forest"),
            ModelSpec("extra_trees"),
            ModelSpec("gradient_boosting"),
            ModelSpec("hist_gradient_boosting"),
        ],
        evaluation=EvaluationConfig(
            walk_forward=WalkForwardConfig(
                train_years=3,
                test_months=1,
                step_months=1,
                min_train_rows=200,
                min_test_rows=15,
                min_train_positive_rate=0.05,
                min_test_positive_rate=0.05,
                embargo_periods=1,
            )
        ),
        paper=PaperConfig(
            enabled=True,
            execution_mode=execution_mode,
            agent_backend=agent_backend,
            agent_model=agent_model,
            poll_interval_seconds=1,
            approval_inbox_dir="artifacts/paper/inbox",
            state_dir="artifacts/paper/state",
            notifications=PaperNotificationsConfig(
                telegram=TelegramNotificationsConfig(enabled=telegram_enabled)
            ),
        ),
    )


class FakeAlpacaProvider:
    def __init__(self, frame: pd.DataFrame | None = None, *, symbol: str = "VOO") -> None:
        self.frame = frame if frame is not None else build_paper_history_frame()
        self.symbol = symbol

    def download_symbol_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> pd.DataFrame:
        if symbol != self.symbol:
            raise ValueError(f"Unexpected symbol: {symbol}")
        if interval != "1d":
            raise ValueError(f"Unexpected interval: {interval}")
        return self.frame.copy()


class FakeAlpacaBroker:
    def __init__(
        self,
        *,
        trading_days: Sequence[date] | None = None,
        equity: float = 10_000.0,
        buying_power: float | None = None,
        cash: float | None = None,
        current_qty: float = 0.0,
        market_price: float = 100.0,
        order_status: str = "accepted",
        symbol: str = "VOO",
    ) -> None:
        if trading_days is None:
            trading_days = tuple(
                pd.bdate_range("2026-04-10", "2026-04-24").date
            )
        self.trading_days = list(trading_days)
        self.equity = equity
        self.buying_power = buying_power if buying_power is not None else equity
        self.cash = cash if cash is not None else self.buying_power
        self.current_qty = current_qty
        self.market_price = market_price
        self.order_status = order_status
        self.symbol = symbol
        self.submitted_orders: list[dict[str, object]] = []

    def get_calendar(self, *, start_date: date, end_date: date) -> list[dict[str, object]]:
        return [
            {"date": trading_day.isoformat()}
            for trading_day in self.trading_days
            if start_date <= trading_day <= end_date
        ]

    def get_account(self) -> dict[str, object]:
        return {
            "account_number": "PA123456",
            "equity": f"{self.equity:.2f}",
            "buying_power": f"{self.buying_power:.2f}",
            "cash": f"{self.cash:.2f}",
            "status": "ACTIVE",
        }

    def get_position(self, symbol: str) -> dict[str, object] | None:
        if symbol != self.symbol or self.current_qty == 0.0:
            return None
        return {
            "symbol": symbol,
            "qty": f"{self.current_qty:.6f}",
            "market_value": f"{self.current_qty * self.market_price:.2f}",
        }

    def submit_fractional_day_market_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,
        client_order_id: str,
    ) -> dict[str, object]:
        order = {
            "id": f"order-{len(self.submitted_orders) + 1}",
            "symbol": symbol,
            "qty": f"{qty:.6f}",
            "side": side,
            "client_order_id": client_order_id,
            "status": self.order_status,
        }
        self.submitted_orders.append(order)
        if side == "buy":
            self.current_qty += qty
        else:
            self.current_qty = max(0.0, self.current_qty - qty)
        return order

    def submit_notional_day_market_order(
        self,
        *,
        symbol: str,
        notional: float,
        side: str,
        client_order_id: str,
    ) -> dict[str, object]:
        order = {
            "id": f"order-{len(self.submitted_orders) + 1}",
            "symbol": symbol,
            "notional": f"{notional:.2f}",
            "side": side,
            "client_order_id": client_order_id,
            "status": self.order_status,
        }
        self.submitted_orders.append(order)
        if side == "buy" and self.market_price > 0:
            self.current_qty += notional / self.market_price
            self.buying_power = max(0.0, self.buying_power - notional)
            self.cash = max(0.0, self.cash - notional)
        return order

    def get_order(self, order_id: str) -> dict[str, object]:
        return {
            "id": order_id,
            "status": self.order_status,
            "client_order_id": self.submitted_orders[-1]["client_order_id"],
        }


def write_phase7_paper_config(
    path: Path,
    *,
    execution_mode: str = "agent_approval",
    symbol: str = "VOO",
    agent_backend: str = "deterministic_consensus",
    agent_model: str = "",
    telegram_enabled: bool = False,
) -> Path:
    payload = {
        "experiment_name": "phase7_paper_fixture",
        "data": {
            "symbols": [symbol],
            "start_date": "2022-01-03",
            "end_date": "2026-04-10",
            "interval": "1d",
            "cache_dir": "artifacts/data-voo-paper-daily",
            "prepared_panel_filename": "panel.csv",
        },
        "features": {
            "return_windows": [5, 10, 20, 40],
            "ma_windows": [10, 20, 50],
            "vol_windows": [10, 20],
            "momentum_window": 20,
        },
        "target": {
            "horizon_days": 1,
            "type": "direction",
        },
        "portfolio": {
            "ranking": {
                "long_n": 1,
                "short_n": 1,
                "rebalance_frequency": "D",
                "weighting": "equal",
                "mode": "long_only",
                "min_score_threshold": 0.55,
                "cash_when_underfilled": True,
            }
        },
        "models": [
            {"name": "logistic_regression"},
            {"name": "logistic_l1"},
            {"name": "random_forest"},
            {"name": "extra_trees"},
            {"name": "gradient_boosting"},
            {"name": "hist_gradient_boosting"},
        ],
        "evaluation": {
            "walk_forward": {
                "train_years": 3,
                "test_months": 1,
                "step_months": 1,
                "min_train_rows": 200,
                "min_test_rows": 15,
                "min_train_positive_rate": 0.05,
                "min_test_positive_rate": 0.05,
                "embargo_periods": 1,
            }
        },
        "paper": {
            "enabled": True,
            "data_provider": "alpaca",
            "broker": "alpaca",
            "execution_mode": execution_mode,
            "agent_backend": agent_backend,
            "agent_model": agent_model,
            "agent_timeout_seconds": 30,
            "agent_fallback_backend": "deterministic_consensus",
            "consensus_min_long_votes": 4,
            "schedule_timezone": "America/New_York",
            "decision_time": "16:10",
            "submission_time": "19:05",
            "order_type": "day_market",
            "position_sizing": "full_equity_fractional",
            "approval_inbox_dir": "artifacts/paper/inbox",
            "state_dir": "artifacts/paper/state",
            "poll_interval_seconds": 1,
            "notifications": {
                "telegram": {
                    "enabled": telegram_enabled,
                }
            },
        },
        "artifacts": {
            "output_dir": "artifacts/runs",
            "save_predictions": True,
            "save_metrics_csv": True,
            "save_report_md": True,
            "save_plots": False,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
