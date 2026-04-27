from __future__ import annotations

from datetime import UTC, date, datetime
from datetime import time as dt_time
from zoneinfo import ZoneInfo

import pandas as pd

from marketlab.config import ExperimentConfig
from marketlab.models import supported_model_names

APPROVAL_PENDING = "pending"
APPROVAL_APPROVED = "approved"
APPROVAL_REJECTED = "rejected"
APPROVAL_NOT_REQUIRED = "not_required"
SUBMISSION_PENDING = "pending"
SUBMISSION_SUBMITTED = "submitted"
SUBMISSION_NOOP = "no_trade_required"
SUBMISSION_SKIPPED = "skipped"
CONSENSUS_POLICY = "consensus_vote"
TERMINAL_ORDER_STATUSES = {"canceled", "expired", "filled", "rejected"}
FAILED_ORDER_STATUSES = {"canceled", "expired", "rejected"}
BUY_NOTIONAL_BUFFER_RATIO = 0.99
ALPACA_MIN_NOTIONAL_ORDER = 1.0


def _now_utc(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _local_now(config: ExperimentConfig, now: datetime | None = None) -> datetime:
    return _now_utc(now).astimezone(ZoneInfo(config.paper.schedule_timezone))


def _clock_value(clock_text: str) -> dt_time:
    hour, minute = (int(part) for part in clock_text.split(":"))
    return dt_time(hour=hour, minute=minute)


def _iso_date(value: pd.Timestamp | datetime | date | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, datetime):
        return value.date().isoformat()
    return value.isoformat()


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _client_order_id(proposal_id: str, *, retry_suffix: str = "") -> str:
    base = f"marketlab-{proposal_id}".replace("_", "-")
    if retry_suffix == "":
        return base[:48]
    suffix = f"-{retry_suffix}".replace("_", "-")
    max_base_length = max(1, 48 - len(suffix))
    return f"{base[:max_base_length]}{suffix}"


def _position_market_value(position: dict[str, object] | None, *, reference_price: float) -> float:
    if position is None:
        return 0.0
    market_value = _safe_float(position.get("market_value"), default=float("nan"))
    if pd.notna(market_value):
        return abs(market_value)
    return abs(_safe_float(position.get("qty"))) * reference_price


def _buy_order_notional(
    *,
    equity: float,
    buying_power: float,
    current_market_value: float,
    target_weight: float,
) -> tuple[float, float]:
    desired_notional = max(equity * target_weight * BUY_NOTIONAL_BUFFER_RATIO, 0.0)
    available_notional = max(buying_power * BUY_NOTIONAL_BUFFER_RATIO, 0.0)
    order_notional = min(max(desired_notional - current_market_value, 0.0), available_notional)
    return desired_notional, order_notional


def _rounded_notional(value: float) -> float:
    return float(f"{max(value, 0.0):.2f}")


def _paper_symbol(config: ExperimentConfig) -> str:
    symbols = [str(symbol).strip() for symbol in config.data.symbols if str(symbol).strip() != ""]
    if len(symbols) != 1:
        raise RuntimeError(
            "Phase 7 paper trading requires exactly one configured symbol in data.symbols."
        )
    return symbols[0]


def _paper_model_names(config: ExperimentConfig) -> list[str]:
    configured = [spec.name for spec in config.models]
    if not configured:
        raise RuntimeError("Phase 7.1 paper trading requires configured models.")
    if len(set(configured)) != len(configured):
        raise RuntimeError("Phase 7.1 paper trading does not allow duplicate model names.")

    required_models = set(supported_model_names())
    configured_set = set(configured)
    if configured_set != required_models or len(configured) != len(required_models):
        required = ", ".join(sorted(required_models))
        raise RuntimeError(
            "Phase 7.1 paper trading requires config.models to include each supported "
            f"direction model exactly once: {required}."
        )
    if config.paper.consensus_min_long_votes > len(configured):
        raise RuntimeError(
            "paper.consensus_min_long_votes cannot exceed the number of configured paper models."
        )
    return configured


def validate_paper_trading_config(config: ExperimentConfig) -> None:
    if not config.paper.enabled:
        raise RuntimeError("paper.enabled must be true for Phase 7 paper-trading commands.")
    _paper_symbol(config)
    _paper_model_names(config)
    if config.data.interval != "1d":
        raise RuntimeError("Phase 7 paper trading requires data.interval='1d'.")
    if config.target.type != "direction":
        raise RuntimeError("Phase 7 paper trading requires target.type='direction'.")
    if config.target.horizon_days != 1:
        raise RuntimeError("Phase 7 paper trading requires target.horizon_days=1.")
    if config.portfolio.ranking.rebalance_frequency != "D":
        raise RuntimeError("Phase 7 paper trading requires portfolio.ranking.rebalance_frequency='D'.")
    if config.portfolio.ranking.mode != "long_only":
        raise RuntimeError("Phase 7 paper trading requires portfolio.ranking.mode='long_only'.")
    if config.portfolio.ranking.long_n != 1 or config.portfolio.ranking.short_n != 1:
        raise RuntimeError("Phase 7 paper trading requires long_n=1 and short_n=1.")
