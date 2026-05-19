from __future__ import annotations

import pandas as pd
import pytest

from marketlab.config import ExperimentConfig, RegimeParticipationPolicyConfig
from marketlab.pipeline import (
    _apply_allocation_score_policy,
    _deterministic_regime_fallback_weights_for_rows,
    _latest_training_rows,
    _predicted_tier_support,
    _select_ml_strategy_candidate,
    _select_ml_strategy_candidate_for_policy,
    _strict_research_gate,
)


def test_strict_research_gate_requires_cost_regime_and_exposure_checks() -> None:
    config = ExperimentConfig()
    config.evaluation.strict_research_gate.enabled = True
    config.evaluation.strict_research_gate.strategy_name = "ml_indicator_tuned__long_only__cash"
    config.evaluation.strict_research_gate.benchmark_strategy = "buy_hold"

    strategy_summary = pd.DataFrame(
        [
            {
                "strategy": "buy_hold",
                "cumulative_return": 0.20,
                "sharpe_like": 1.0,
                "max_drawdown": -0.30,
                "avg_gross_exposure": 1.0,
            },
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "cumulative_return": 0.28,
                "sharpe_like": 1.1,
                "max_drawdown": -0.25,
                "avg_gross_exposure": 0.50,
            },
        ]
    )
    cost_sensitivity = pd.DataFrame(
        [
            {
                "strategy": "buy_hold",
                "bps_per_trade": 35.0,
                "cumulative_return": 0.20,
            },
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "bps_per_trade": 35.0,
                "cumulative_return": 0.26,
            },
            {
                "strategy": "buy_hold",
                "bps_per_trade": 50.0,
                "cumulative_return": 0.20,
            },
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "bps_per_trade": 50.0,
                "cumulative_return": 0.24,
            },
        ]
    )
    regime_slices = pd.DataFrame(
        {
            "slice_name": ["bull", "bear", "sideways"],
            "active_return": [0.02, 0.01, 0.03],
        }
    )
    selections = pd.DataFrame(
        {
            "fold_id": [1, 2],
            "selection_status": ["selected", "selected"],
        }
    )

    gate = _strict_research_gate(
        config=config,
        strategy_summary=strategy_summary,
        cost_sensitivity=cost_sensitivity,
        regime_slices=regime_slices,
        ml_strategy_tuning_selections=selections,
    )

    overall = gate.loc[gate["condition"] == "overall"].iloc[0]
    assert bool(overall["passed"]) is True
    assert bool(
        gate.loc[gate["condition"] == "selected_walk_forward_fold_fraction", "passed"].iloc[0]
    ) is True


def test_allocation_score_policy_promotes_only_runtime_bull_rows() -> None:
    rows = pd.DataFrame(
        {
            "crypto_regime_risk_off": [0, 1, 0, 0],
            "crypto_regime_trend_state": [1, 1, 0, -1],
        },
        index=[10, 11, 12, 13],
    )
    scores = pd.Series([0.42, 0.42, 0.42, 0.42], index=rows.index)
    probabilities = pd.DataFrame(
        {
            "prob_tier_0": [0.15, 0.10, 0.05, 0.05],
            "prob_tier_100": [0.25, 0.30, 0.40, 0.40],
        },
        index=rows.index,
    )

    final_scores, triggered = _apply_allocation_score_policy(
        rows=rows,
        score_series=scores,
        probability_frame=probabilities,
        allocation_score_policy="bull_prob100_threshold",
        prob100_threshold=0.20,
    )

    assert final_scores.tolist() == [1.0, 0.42, 0.42, 0.42]
    assert triggered.tolist() == [True, False, False, False]


def test_allocation_score_policy_keeps_expected_score_when_threshold_fails() -> None:
    rows = pd.DataFrame(
        {
            "crypto_regime_risk_off": [0, 0],
            "crypto_regime_trend_state": [1, 1],
        }
    )
    scores = pd.Series([0.42, 0.45])
    probabilities = pd.DataFrame(
        {
            "prob_tier_0": [0.30, 0.10],
            "prob_tier_100": [0.25, 0.19],
        }
    )

    final_scores, triggered = _apply_allocation_score_policy(
        rows=rows,
        score_series=scores,
        probability_frame=probabilities,
        allocation_score_policy="bull_prob100_threshold",
        prob100_threshold=0.20,
    )

    assert final_scores.tolist() == [0.42, 0.45]
    assert triggered.tolist() == [False, False]


def test_deterministic_regime_fallback_weights_map_test_regime_labels() -> None:
    panel_dates = pd.date_range("2026-01-01", periods=6, freq="D")
    panel = pd.DataFrame(
        {
            "symbol": ["BTC-USD"] * len(panel_dates),
            "timestamp": panel_dates,
        }
    )
    rows = pd.DataFrame(
        {
            "signal_date": panel_dates[:4],
            "effective_date": panel_dates[1:5],
            "crypto_regime_risk_off": [1, 0, 0, 0],
            "crypto_regime_trend_state": [1, 1, 0, -1],
        }
    )
    policy = RegimeParticipationPolicyConfig(
        name="bull100_sideways50_bear25",
        bull_floor=1.0,
        sideways_floor=0.50,
        bear_floor=0.25,
        risk_off_cap=0.25,
    )

    weights = _deterministic_regime_fallback_weights_for_rows(
        panel=panel,
        rows=rows,
        frequency="bar",
        strategy_name="ml_indicator_tuned__long_only__cash",
        policy=policy,
    )

    assert weights["weight"].tolist() == [0.25, 1.0, 0.50, 0.25, 0.0]
    assert weights["symbol"].eq("BTC-USD").all()
    assert weights["strategy"].eq("ml_indicator_tuned__long_only__cash").all()


def test_deterministic_regime_fallback_weights_require_regime_columns() -> None:
    panel = pd.DataFrame(
        {
            "symbol": ["BTC-USD"],
            "timestamp": [pd.Timestamp("2026-01-01")],
        }
    )
    rows = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2026-01-01")],
            "effective_date": [pd.Timestamp("2026-01-02")],
        }
    )

    with pytest.raises(ValueError, match="no_candidate_fallback_regime_policy"):
        _deterministic_regime_fallback_weights_for_rows(
            panel=panel,
            rows=rows,
            frequency="bar",
            strategy_name="ml_indicator_tuned__long_only__cash",
            policy=RegimeParticipationPolicyConfig(name="bull100_sideways25"),
        )


def test_strict_research_gate_fails_when_cost_case_misses_benchmark() -> None:
    config = ExperimentConfig()
    config.evaluation.strict_research_gate.enabled = True

    strategy_summary = pd.DataFrame(
        [
            {
                "strategy": "buy_hold",
                "cumulative_return": 0.20,
                "sharpe_like": 1.0,
                "max_drawdown": -0.30,
                "avg_gross_exposure": 1.0,
            },
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "cumulative_return": 0.28,
                "sharpe_like": 1.1,
                "max_drawdown": -0.25,
                "avg_gross_exposure": 0.50,
            },
        ]
    )
    cost_sensitivity = pd.DataFrame(
        [
            {"strategy": "buy_hold", "bps_per_trade": 35.0, "cumulative_return": 0.20},
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "bps_per_trade": 35.0,
                "cumulative_return": 0.19,
            },
            {"strategy": "buy_hold", "bps_per_trade": 50.0, "cumulative_return": 0.20},
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "bps_per_trade": 50.0,
                "cumulative_return": 0.24,
            },
        ]
    )

    gate = _strict_research_gate(
        config=config,
        strategy_summary=strategy_summary,
        cost_sensitivity=cost_sensitivity,
        regime_slices=pd.DataFrame({"slice_name": ["bull", "bear", "sideways"], "active_return": [1, 1, 1]}),
        ml_strategy_tuning_selections=pd.DataFrame({"selection_status": ["selected", "selected"]}),
    )

    assert bool(gate.loc[gate["condition"] == "cost_gate_bps", "passed"].iloc[0]) is False
    assert bool(gate.loc[gate["condition"] == "overall", "passed"].iloc[0]) is False


def test_strict_research_gate_fails_when_turnover_budget_is_exceeded() -> None:
    config = ExperimentConfig()
    config.evaluation.periods_per_year = 10.0
    config.evaluation.strict_research_gate.enabled = True
    config.evaluation.ml_strategy_tuning.max_annualized_turnover = 1.0

    strategy_summary = pd.DataFrame(
        [
            {
                "strategy": "buy_hold",
                "cumulative_return": 0.20,
                "sharpe_like": 1.0,
                "max_drawdown": -0.30,
                "avg_gross_exposure": 1.0,
                "avg_turnover": 0.0,
            },
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "cumulative_return": 0.28,
                "sharpe_like": 1.1,
                "max_drawdown": -0.25,
                "avg_gross_exposure": 0.50,
                "avg_turnover": 0.20,
            },
        ]
    )
    cost_sensitivity = pd.DataFrame(
        [
            {"strategy": "buy_hold", "bps_per_trade": 35.0, "cumulative_return": 0.20},
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "bps_per_trade": 35.0,
                "cumulative_return": 0.26,
            },
            {"strategy": "buy_hold", "bps_per_trade": 50.0, "cumulative_return": 0.20},
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "bps_per_trade": 50.0,
                "cumulative_return": 0.24,
            },
        ]
    )

    gate = _strict_research_gate(
        config=config,
        strategy_summary=strategy_summary,
        cost_sensitivity=cost_sensitivity,
        regime_slices=pd.DataFrame(
            {"slice_name": ["bull", "bear", "sideways"], "active_return": [1, 1, 1]}
        ),
        ml_strategy_tuning_selections=pd.DataFrame(
            {"selection_status": ["selected", "selected"]}
        ),
    )

    assert bool(
        gate.loc[gate["condition"] == "annualized_turnover_budget", "passed"].iloc[0]
    ) is False
    assert bool(gate.loc[gate["condition"] == "overall", "passed"].iloc[0]) is False


def test_strict_research_gate_requires_static_partial_benchmark_outperformance() -> None:
    config = ExperimentConfig()
    config.evaluation.strict_research_gate.enabled = True
    config.evaluation.strict_research_gate.required_benchmark_strategies = [
        "buy_hold",
        "btc_static_25",
        "btc_static_50",
        "btc_static_75",
    ]

    strategy_summary = pd.DataFrame(
        [
            {
                "strategy": "buy_hold",
                "cumulative_return": 0.10,
                "sharpe_like": 1.0,
                "max_drawdown": -0.30,
                "avg_gross_exposure": 1.0,
            },
            {
                "strategy": "btc_static_25",
                "cumulative_return": 0.12,
                "sharpe_like": 1.0,
                "max_drawdown": -0.10,
                "avg_gross_exposure": 0.25,
            },
            {
                "strategy": "btc_static_50",
                "cumulative_return": 0.13,
                "sharpe_like": 1.0,
                "max_drawdown": -0.20,
                "avg_gross_exposure": 0.50,
            },
            {
                "strategy": "btc_static_75",
                "cumulative_return": 0.14,
                "sharpe_like": 1.0,
                "max_drawdown": -0.25,
                "avg_gross_exposure": 0.75,
            },
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "cumulative_return": 0.11,
                "sharpe_like": 1.2,
                "max_drawdown": -0.20,
                "avg_gross_exposure": 0.50,
            },
        ]
    )
    cost_sensitivity = pd.DataFrame(
        [
            {"strategy": strategy, "bps_per_trade": bps, "cumulative_return": cumulative_return}
            for bps in [35.0, 50.0]
            for strategy, cumulative_return in [
                ("buy_hold", 0.10),
                ("btc_static_25", 0.12),
                ("btc_static_50", 0.13),
                ("btc_static_75", 0.14),
                ("ml_indicator_tuned__long_only__cash", 0.11),
            ]
        ]
    )

    gate = _strict_research_gate(
        config=config,
        strategy_summary=strategy_summary,
        cost_sensitivity=cost_sensitivity,
        regime_slices=pd.DataFrame(
            {"slice_name": ["bull", "bear", "sideways"], "active_return": [1, 1, 1]}
        ),
        ml_strategy_tuning_selections=pd.DataFrame(
            {"selection_status": ["selected", "selected", "selected", "selected"]}
        ),
    )

    assert bool(
        gate.loc[
            gate["condition"] == "net_cumulative_return_beats_btc_static_25",
            "passed",
        ].iloc[0]
    ) is False
    assert bool(
        gate.loc[gate["condition"] == "cost_gate_bps_vs_btc_static_75", "passed"].iloc[0]
    ) is False
    assert bool(gate.loc[gate["condition"] == "overall", "passed"].iloc[0]) is False


def test_strict_research_gate_requires_rebalanced_partial_benchmark_outperformance() -> None:
    config = ExperimentConfig()
    config.evaluation.strict_research_gate.enabled = True
    config.evaluation.strict_research_gate.required_benchmark_strategies = [
        "buy_hold",
        "btc_rebalanced_25",
        "btc_rebalanced_50",
        "btc_rebalanced_75",
    ]

    strategy_summary = pd.DataFrame(
        [
            {
                "strategy": "buy_hold",
                "cumulative_return": 0.10,
                "sharpe_like": 1.0,
                "max_drawdown": -0.30,
                "avg_gross_exposure": 1.0,
            },
            {
                "strategy": "btc_rebalanced_25",
                "cumulative_return": 0.12,
                "sharpe_like": 1.0,
                "max_drawdown": -0.10,
                "avg_gross_exposure": 0.25,
            },
            {
                "strategy": "btc_rebalanced_50",
                "cumulative_return": 0.16,
                "sharpe_like": 1.0,
                "max_drawdown": -0.20,
                "avg_gross_exposure": 0.50,
            },
            {
                "strategy": "btc_rebalanced_75",
                "cumulative_return": 0.18,
                "sharpe_like": 1.0,
                "max_drawdown": -0.25,
                "avg_gross_exposure": 0.75,
            },
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "cumulative_return": 0.15,
                "sharpe_like": 1.2,
                "max_drawdown": -0.20,
                "avg_gross_exposure": 0.50,
            },
        ]
    )
    cost_sensitivity = pd.DataFrame(
        [
            {"strategy": strategy, "bps_per_trade": bps, "cumulative_return": cumulative_return}
            for bps in [35.0, 50.0]
            for strategy, cumulative_return in [
                ("buy_hold", 0.10),
                ("btc_rebalanced_25", 0.12),
                ("btc_rebalanced_50", 0.16),
                ("btc_rebalanced_75", 0.18),
                ("ml_indicator_tuned__long_only__cash", 0.15),
            ]
        ]
    )

    gate = _strict_research_gate(
        config=config,
        strategy_summary=strategy_summary,
        cost_sensitivity=cost_sensitivity,
        regime_slices=pd.DataFrame(
            {"slice_name": ["bull", "bear", "sideways"], "active_return": [1, 1, 1]}
        ),
        ml_strategy_tuning_selections=pd.DataFrame(
            {"selection_status": ["selected", "selected", "selected", "selected"]}
        ),
    )

    assert bool(
        gate.loc[
            gate["condition"] == "net_cumulative_return_beats_btc_rebalanced_50",
            "passed",
        ].iloc[0]
    ) is False
    assert bool(
        gate.loc[
            gate["condition"] == "acceptable_cost_bps_vs_btc_rebalanced_75",
            "passed",
        ].iloc[0]
    ) is False
    assert bool(gate.loc[gate["condition"] == "overall", "passed"].iloc[0]) is False


def test_strict_research_gate_fails_when_rebalanced_benchmark_artifact_is_missing() -> None:
    config = ExperimentConfig()
    config.evaluation.strict_research_gate.enabled = True
    config.evaluation.strict_research_gate.required_benchmark_strategies = [
        "buy_hold",
        "btc_rebalanced_25",
    ]

    strategy_summary = pd.DataFrame(
        [
            {
                "strategy": "buy_hold",
                "cumulative_return": 0.10,
                "sharpe_like": 1.0,
                "max_drawdown": -0.30,
                "avg_gross_exposure": 1.0,
            },
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "cumulative_return": 0.20,
                "sharpe_like": 1.2,
                "max_drawdown": -0.20,
                "avg_gross_exposure": 0.50,
            },
        ]
    )
    cost_sensitivity = pd.DataFrame(
        [
            {"strategy": "buy_hold", "bps_per_trade": 35.0, "cumulative_return": 0.10},
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "bps_per_trade": 35.0,
                "cumulative_return": 0.20,
            },
            {"strategy": "buy_hold", "bps_per_trade": 50.0, "cumulative_return": 0.10},
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "bps_per_trade": 50.0,
                "cumulative_return": 0.19,
            },
        ]
    )

    gate = _strict_research_gate(
        config=config,
        strategy_summary=strategy_summary,
        cost_sensitivity=cost_sensitivity,
        regime_slices=pd.DataFrame(
            {"slice_name": ["bull", "bear", "sideways"], "active_return": [1, 1, 1]}
        ),
        ml_strategy_tuning_selections=pd.DataFrame(
            {"selection_status": ["selected", "selected", "selected", "selected"]}
        ),
    )

    assert bool(
        gate.loc[
            gate["condition"] == "required_benchmark_strategies_present",
            "passed",
        ].iloc[0]
    ) is False
    assert bool(
        gate.loc[
            gate["condition"] == "net_cumulative_return_beats_btc_rebalanced_25",
            "passed",
        ].iloc[0]
    ) is False
    assert bool(gate.loc[gate["condition"] == "overall", "passed"].iloc[0]) is False


def test_strict_research_gate_fails_when_selected_fold_fraction_is_low() -> None:
    config = ExperimentConfig()
    config.evaluation.strict_research_gate.enabled = True
    config.evaluation.strict_research_gate.min_selected_fold_fraction = 0.75

    strategy_summary = pd.DataFrame(
        [
            {
                "strategy": "buy_hold",
                "cumulative_return": 0.20,
                "sharpe_like": 1.0,
                "max_drawdown": -0.30,
                "avg_gross_exposure": 1.0,
            },
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "cumulative_return": 0.28,
                "sharpe_like": 1.1,
                "max_drawdown": -0.25,
                "avg_gross_exposure": 0.50,
            },
        ]
    )
    cost_sensitivity = pd.DataFrame(
        [
            {"strategy": "buy_hold", "bps_per_trade": 35.0, "cumulative_return": 0.20},
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "bps_per_trade": 35.0,
                "cumulative_return": 0.26,
            },
            {"strategy": "buy_hold", "bps_per_trade": 50.0, "cumulative_return": 0.20},
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "bps_per_trade": 50.0,
                "cumulative_return": 0.24,
            },
        ]
    )

    gate = _strict_research_gate(
        config=config,
        strategy_summary=strategy_summary,
        cost_sensitivity=cost_sensitivity,
        regime_slices=pd.DataFrame(
            {"slice_name": ["bull", "bear", "sideways"], "active_return": [1, 1, 1]}
        ),
        ml_strategy_tuning_selections=pd.DataFrame(
            {
                "selection_status": [
                    "selected",
                    "no_valid_candidate",
                    "no_valid_candidate",
                    "no_valid_candidate",
                ]
            }
        ),
    )

    assert bool(
        gate.loc[gate["condition"] == "selected_walk_forward_fold_fraction", "passed"].iloc[0]
    ) is False
    assert bool(gate.loc[gate["condition"] == "overall", "passed"].iloc[0]) is False


def test_strict_gate_counts_deterministic_regime_fallback_rows_as_selected() -> None:
    config = ExperimentConfig()
    config.evaluation.strict_research_gate.enabled = True
    config.evaluation.strict_research_gate.min_selected_fold_fraction = 0.75
    strategy_summary, cost_sensitivity, regime_slices, _ = (
        _passing_target_support_gate_inputs()
    )
    selections = pd.DataFrame(
        {
            "fold_id": [1, 2, 3, 4],
            "selection_status": ["selected", "selected", "selected", "selected"],
            "selection_source": ["deterministic_regime_fallback"] * 4,
            "passed_gate": [False, False, False, False],
            "selected_model_name": [pd.NA, pd.NA, pd.NA, pd.NA],
            "selected_regime_policy": ["bull100_sideways25"] * 4,
        }
    )

    gate = _strict_research_gate(
        config=config,
        strategy_summary=strategy_summary,
        cost_sensitivity=cost_sensitivity,
        regime_slices=regime_slices,
        ml_strategy_tuning_selections=selections,
    )

    assert bool(
        gate.loc[gate["condition"] == "selected_walk_forward_fold_fraction", "passed"].iloc[0]
    ) is True


def test_strict_research_gate_fails_low_oos_exposure_even_with_positive_active_return() -> None:
    config = ExperimentConfig()
    config.evaluation.strict_research_gate.enabled = True

    strategy_summary = pd.DataFrame(
        [
            {
                "strategy": "buy_hold",
                "cumulative_return": -0.20,
                "sharpe_like": -1.0,
                "max_drawdown": -0.50,
                "avg_gross_exposure": 1.0,
            },
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "cumulative_return": 0.02,
                "sharpe_like": 0.2,
                "max_drawdown": -0.05,
                "avg_gross_exposure": 0.05,
            },
        ]
    )
    cost_sensitivity = pd.DataFrame(
        [
            {"strategy": "buy_hold", "bps_per_trade": 35.0, "cumulative_return": -0.20},
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "bps_per_trade": 35.0,
                "cumulative_return": 0.02,
            },
            {"strategy": "buy_hold", "bps_per_trade": 50.0, "cumulative_return": -0.20},
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "bps_per_trade": 50.0,
                "cumulative_return": 0.01,
            },
        ]
    )

    gate = _strict_research_gate(
        config=config,
        strategy_summary=strategy_summary,
        cost_sensitivity=cost_sensitivity,
        regime_slices=pd.DataFrame(
            {"slice_name": ["bull", "bear", "sideways"], "active_return": [1, 1, 1]}
        ),
        ml_strategy_tuning_selections=pd.DataFrame(
            {"selection_status": ["selected", "selected", "selected", "selected"]}
        ),
    )

    assert bool(
        gate.loc[gate["condition"] == "average_exposure_in_range", "passed"].iloc[0]
    ) is False
    assert bool(gate.loc[gate["condition"] == "overall", "passed"].iloc[0]) is False


def test_ml_strategy_candidate_selection_uses_net_then_risk_then_lower_turnover() -> None:
    candidates = [
        {
            "model_name": "higher_sharpe",
            "excess_cumulative_return": 0.10,
            "drawdown_delta": 0.05,
            "sharpe_like_delta": 0.30,
            "annualized_turnover": 10.0,
        },
        {
            "model_name": "better_drawdown",
            "excess_cumulative_return": 0.10,
            "drawdown_delta": 0.06,
            "sharpe_like_delta": 0.10,
            "annualized_turnover": 20.0,
        },
        {
            "model_name": "lower_turnover",
            "excess_cumulative_return": 0.10,
            "drawdown_delta": 0.06,
            "sharpe_like_delta": 0.10,
            "annualized_turnover": 5.0,
        },
        {
            "model_name": "lower_net_return",
            "excess_cumulative_return": 0.09,
            "drawdown_delta": 1.0,
            "sharpe_like_delta": 1.0,
            "annualized_turnover": 1.0,
        },
    ]

    selected = _select_ml_strategy_candidate(candidates)

    assert selected["model_name"] == "lower_turnover"


def test_ml_strategy_candidate_selection_prefers_required_benchmark_margin() -> None:
    candidates = [
        {
            "model_name": "better_buy_hold_excess",
            "excess_cumulative_return": 0.20,
            "min_benchmark_excess_cumulative_return": 0.01,
            "drawdown_delta": 1.0,
            "sharpe_like_delta": 1.0,
            "annualized_turnover": 1.0,
        },
        {
            "model_name": "better_benchmark_margin",
            "excess_cumulative_return": 0.12,
            "min_benchmark_excess_cumulative_return": 0.05,
            "drawdown_delta": 0.0,
            "sharpe_like_delta": 0.0,
            "annualized_turnover": 10.0,
        },
    ]

    selected = _select_ml_strategy_candidate(candidates)

    assert selected["model_name"] == "better_benchmark_margin"


def _runtime_selection_candidate(
    *,
    model_name: str,
    passed_gate: bool,
    failure_reasons: str,
    min_benchmark_excess: float,
    excess_return: float = 0.10,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "passed_gate": passed_gate,
        "failure_reasons": failure_reasons,
        "excess_cumulative_return": excess_return,
        "min_benchmark_excess_cumulative_return": min_benchmark_excess,
        "drawdown_delta": 0.05,
        "sharpe_like_delta": 0.05,
        "annualized_turnover": 4.0,
    }


def test_runtime_selection_policy_prefers_strict_candidate() -> None:
    selected, source = _select_ml_strategy_candidate_for_policy(
        [
            _runtime_selection_candidate(
                model_name="fallback",
                passed_gate=False,
                failure_reasons="non_positive_required_benchmark_excess",
                min_benchmark_excess=-0.01,
                excess_return=0.50,
            ),
            _runtime_selection_candidate(
                model_name="strict",
                passed_gate=True,
                failure_reasons="",
                min_benchmark_excess=0.01,
                excess_return=0.02,
            ),
        ],
        selection_policy="best_active_fallback",
    )

    assert selected is not None
    assert selected["model_name"] == "strict"
    assert source == "strict"


def test_runtime_selection_policy_accepts_benchmark_only_fallback() -> None:
    selected, source = _select_ml_strategy_candidate_for_policy(
        [
            _runtime_selection_candidate(
                model_name="weaker_fallback",
                passed_gate=False,
                failure_reasons="non_positive_required_benchmark_excess",
                min_benchmark_excess=-0.25,
            ),
            _runtime_selection_candidate(
                model_name="better_fallback",
                passed_gate=False,
                failure_reasons="non_positive_required_benchmark_excess",
                min_benchmark_excess=-0.05,
            ),
        ],
        selection_policy="best_active_fallback",
    )

    assert selected is not None
    assert selected["model_name"] == "better_fallback"
    assert source == "best_active_fallback"


def test_runtime_selection_policy_rejects_non_benchmark_fallback_failures() -> None:
    selected, source = _select_ml_strategy_candidate_for_policy(
        [
            _runtime_selection_candidate(
                model_name="inactive",
                passed_gate=False,
                failure_reasons="inactive_candidate;non_positive_required_benchmark_excess",
                min_benchmark_excess=-0.01,
            ),
            _runtime_selection_candidate(
                model_name="turnover",
                passed_gate=False,
                failure_reasons="turnover_budget_exceeded",
                min_benchmark_excess=0.20,
            ),
            _runtime_selection_candidate(
                model_name="predicted_support",
                passed_gate=False,
                failure_reasons="insufficient_predicted_tier_support;non_positive_required_benchmark_excess",
                min_benchmark_excess=-0.01,
            ),
            _runtime_selection_candidate(
                model_name="risk",
                passed_gate=False,
                failure_reasons="non_positive_required_benchmark_excess;risk_not_improved",
                min_benchmark_excess=-0.01,
            ),
            _runtime_selection_candidate(
                model_name="missing_benchmark",
                passed_gate=False,
                failure_reasons="missing_selection_benchmark",
                min_benchmark_excess=-0.01,
            ),
        ],
        selection_policy="best_active_fallback",
    )

    assert selected is None
    assert source == "none"


def test_runtime_selection_policy_strict_does_not_fallback() -> None:
    selected, source = _select_ml_strategy_candidate_for_policy(
        [
            _runtime_selection_candidate(
                model_name="fallback",
                passed_gate=False,
                failure_reasons="non_positive_required_benchmark_excess",
                min_benchmark_excess=-0.01,
            ),
        ],
        selection_policy="strict",
    )

    assert selected is None
    assert source == "none"


def _passing_target_support_gate_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategy_summary = pd.DataFrame(
        [
            {
                "strategy": "buy_hold",
                "cumulative_return": 0.20,
                "sharpe_like": 1.0,
                "max_drawdown": -0.30,
                "avg_gross_exposure": 1.0,
            },
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "cumulative_return": 0.28,
                "sharpe_like": 1.1,
                "max_drawdown": -0.25,
                "avg_gross_exposure": 0.50,
            },
        ]
    )
    cost_sensitivity = pd.DataFrame(
        [
            {"strategy": "buy_hold", "bps_per_trade": 35.0, "cumulative_return": 0.20},
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "bps_per_trade": 35.0,
                "cumulative_return": 0.26,
            },
            {"strategy": "buy_hold", "bps_per_trade": 50.0, "cumulative_return": 0.20},
            {
                "strategy": "ml_indicator_tuned__long_only__cash",
                "bps_per_trade": 50.0,
                "cumulative_return": 0.24,
            },
        ]
    )
    regime_slices = pd.DataFrame(
        {
            "slice_name": ["bull", "bear", "sideways"],
            "active_return": [0.02, 0.01, 0.03],
        }
    )
    selections = pd.DataFrame(
        {
            "fold_id": [1, 2, 3],
            "selection_status": ["selected", "selected", "selected"],
        }
    )
    return strategy_summary, cost_sensitivity, regime_slices, selections


def _target_support_rows(*, partial_count: int, fold_count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "fold_id": "all",
            "scope": "global",
            "regime": "all",
            "target": 1,
            "target_weight": 0.25,
            "row_count": partial_count,
            "row_fraction": partial_count / 100,
        },
        {
            "fold_id": "all",
            "scope": "global",
            "regime": "all",
            "target": 2,
            "target_weight": 0.50,
            "row_count": partial_count,
            "row_fraction": partial_count / 100,
        },
        {
            "fold_id": "all",
            "scope": "global",
            "regime": "all",
            "target": 3,
            "target_weight": 1.0,
            "row_count": 100 - (partial_count * 2),
            "row_fraction": 1.0 - (partial_count * 2 / 100),
        },
    ]
    for fold_id in range(1, 6):
        for target_weight, target in [(0.25, 1), (0.50, 2)]:
            rows.append(
                {
                    "fold_id": fold_id,
                    "scope": "train_validation",
                    "regime": "all",
                    "target": target,
                    "target_weight": target_weight,
                    "row_count": 1 if fold_id <= fold_count else 0,
                    "row_fraction": 0.01 if fold_id <= fold_count else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _predicted_support_rows(*, include_half: bool = True) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_id in range(1, 4):
        rows.extend(
            [
                {"fold_id": fold_id, "score": 0.25, "predicted_tier_weight": 0.25},
                {
                    "fold_id": fold_id,
                    "score": 0.50 if include_half else 0.0,
                    "predicted_tier_weight": 0.50 if include_half else 0.0,
                },
                {"fold_id": fold_id, "score": 1.0, "predicted_tier_weight": 1.0},
            ]
        )
    return pd.DataFrame(rows)


def test_predicted_tier_support_requires_partial_tier_predictions() -> None:
    predictions = pd.DataFrame({"score": [0.0, 0.25, 0.25, 1.0]})

    passed, fractions = _predicted_tier_support(
        predictions=predictions,
        required_weights=[0.25, 0.50],
        min_fraction=0.10,
    )

    assert passed is False
    assert fractions[0.25] == 0.5
    assert fractions[0.50] == 0.0


def test_strict_research_gate_fails_when_partial_target_support_is_low() -> None:
    config = ExperimentConfig()
    config.target.type = "allocation_utility"
    config.evaluation.strict_research_gate.enabled = True
    config.evaluation.strict_research_gate.required_partial_target_weights = [0.25, 0.50]
    strategy_summary, cost_sensitivity, regime_slices, selections = (
        _passing_target_support_gate_inputs()
    )

    gate = _strict_research_gate(
        config=config,
        strategy_summary=strategy_summary,
        cost_sensitivity=cost_sensitivity,
        regime_slices=regime_slices,
        ml_strategy_tuning_selections=selections,
        allocation_target_diagnostics=_target_support_rows(
            partial_count=4,
            fold_count=2,
        ),
        allocation_probability_diagnostics=_predicted_support_rows(),
    )

    assert bool(
        gate.loc[gate["condition"] == "partial_target_25_global_fraction", "passed"].iloc[0]
    ) is False
    assert bool(
        gate.loc[gate["condition"] == "partial_target_50_fold_fraction", "passed"].iloc[0]
    ) is False
    assert bool(gate.loc[gate["condition"] == "overall", "passed"].iloc[0]) is False


def test_strict_research_gate_passes_partial_target_support_rows() -> None:
    config = ExperimentConfig()
    config.target.type = "allocation_utility"
    config.evaluation.strict_research_gate.enabled = True
    config.evaluation.strict_research_gate.required_partial_target_weights = [0.25, 0.50]
    strategy_summary, cost_sensitivity, regime_slices, selections = (
        _passing_target_support_gate_inputs()
    )

    gate = _strict_research_gate(
        config=config,
        strategy_summary=strategy_summary,
        cost_sensitivity=cost_sensitivity,
        regime_slices=regime_slices,
        ml_strategy_tuning_selections=selections,
        allocation_target_diagnostics=_target_support_rows(
            partial_count=6,
            fold_count=3,
        ),
        allocation_probability_diagnostics=_predicted_support_rows(),
    )

    assert bool(
        gate.loc[gate["condition"] == "partial_target_25_global_fraction", "passed"].iloc[0]
    ) is True
    assert bool(
        gate.loc[gate["condition"] == "partial_target_50_fold_fraction", "passed"].iloc[0]
    ) is True
    assert bool(gate.loc[gate["condition"] == "overall", "passed"].iloc[0]) is True


def test_strict_research_gate_fails_when_predicted_target_support_is_low() -> None:
    config = ExperimentConfig()
    config.target.type = "allocation_utility"
    config.evaluation.strict_research_gate.enabled = True
    config.evaluation.strict_research_gate.required_partial_target_weights = [0.25, 0.50]
    config.evaluation.strict_research_gate.required_predicted_target_weights = [0.25, 0.50]
    config.evaluation.strict_research_gate.min_predicted_target_fraction = 0.10
    config.evaluation.strict_research_gate.min_predicted_target_fold_fraction = 0.50
    strategy_summary, cost_sensitivity, regime_slices, selections = (
        _passing_target_support_gate_inputs()
    )

    gate = _strict_research_gate(
        config=config,
        strategy_summary=strategy_summary,
        cost_sensitivity=cost_sensitivity,
        regime_slices=regime_slices,
        ml_strategy_tuning_selections=selections,
        allocation_target_diagnostics=_target_support_rows(
            partial_count=6,
            fold_count=3,
        ),
        allocation_probability_diagnostics=_predicted_support_rows(include_half=False),
    )

    assert bool(
        gate.loc[gate["condition"] == "predicted_target_50_global_fraction", "passed"].iloc[0]
    ) is False
    assert bool(
        gate.loc[gate["condition"] == "predicted_target_50_fold_fraction", "passed"].iloc[0]
    ) is False
    assert bool(gate.loc[gate["condition"] == "overall", "passed"].iloc[0]) is False


def test_latest_training_rows_uses_only_recent_signal_bars() -> None:
    rows = pd.DataFrame(
        {
            "symbol": ["BTC/USD", "ETH/USD"] * 4,
            "signal_date": list(pd.date_range("2026-01-01", periods=4, freq="4h")) * 2,
            "target": [0, 1] * 4,
        }
    )

    selected = _latest_training_rows(rows, rolling_train_bars=2)

    assert set(pd.to_datetime(selected["signal_date"]).unique()) == set(
        pd.date_range("2026-01-01 08:00", periods=2, freq="4h")
    )
