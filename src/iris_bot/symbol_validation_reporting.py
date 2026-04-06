from __future__ import annotations

from pathlib import Path
from typing import Any

from iris_bot.artifacts import wrap_artifact
from iris_bot.config import Settings
from iris_bot.logging_utils import write_json_report
from iris_bot.symbols import default_symbol_strategy_profile


def _initialize_aggregate_portfolio() -> dict[str, Any]:
    return {
        "symbols_evaluated": 0,
        "enabled_symbols": 0,
        "caution_symbols": 0,
        "disabled_symbols": 0,
        "enabled_selection_net_pnl_usd": 0.0,
        "enabled_final_test_net_pnl_usd": 0.0,
    }


def _initialize_validation_reports() -> dict[str, dict[str, Any]]:
    return {
        "strategy_validation_report": {"symbols": {}},
        "model_comparison_report": {"symbols": {}},
        "dynamic_exit_report": {"symbols": {}},
        "threshold_report": {"symbols": {}},
        "symbol_enablement_report": {"symbols": {}},
        "strategy_block_root_cause_report": {"symbols": {}},
        "symbol_block_diagnostics_report": {"symbols": {}},
        "threshold_sensitivity_report": {"symbols": {}},
        "gate_failure_matrix": {"symbols": {}},
        "session_timeframe_diagnostics_report": {"symbols": {}},
        "exit_policy_diagnostics_report": {"symbols": {}},
        "symbol_recommendation_report": {"symbols": {}},
    }


def _build_common_profile(settings: Settings) -> dict[str, Any]:
    return {
        "allowed_timeframes": [settings.trading.primary_timeframe],
        "allowed_sessions": ["asia", "london", "new_york"],
        "allow_long": settings.trading.allow_long,
        "allow_short": settings.trading.allow_short,
        "risk_multiplier": 1.0,
        "max_open_positions": settings.risk.max_open_positions,
        "stop_policy": settings.exit_policy.stop_policy,
        "target_policy": settings.exit_policy.target_policy,
    }


def _record_short_symbol_decision(
    reports: dict[str, dict[str, Any]],
    symbol: str,
    decision: dict[str, Any],
) -> None:
    reports["symbol_enablement_report"]["symbols"][symbol] = decision


def _record_symbol_validation_outputs(
    reports: dict[str, dict[str, Any]],
    *,
    symbol: str,
    chosen_model: str,
    chosen_payload: dict[str, Any],
    comparison: dict[str, Any],
    symbol_split_reports: dict[str, Any],
    walkforward: dict[str, Any],
    decision: dict[str, Any],
    block_matrix: dict[str, Any],
    session_timeframe_diagnostics: dict[str, Any],
    exit_policy_diagnostics: dict[str, Any],
    recommendation: dict[str, Any],
) -> None:
    reports["model_comparison_report"]["symbols"][symbol] = {
        "chosen_model": chosen_model,
        "comparison": comparison,
    }
    reports["dynamic_exit_report"]["symbols"][symbol] = {
        "preferred_exit_policy": chosen_payload["preferred_exit_policy"],
        "static": chosen_payload["static"],
        "atr_dynamic": chosen_payload["atr_dynamic"],
    }
    reports["threshold_report"]["symbols"][symbol] = {
        "chosen_model": chosen_model,
        "thresholds": chosen_payload["thresholds"],
    }
    reports["threshold_sensitivity_report"]["symbols"][symbol] = {
        "chosen_model": chosen_model,
        "chosen_exit_policy": chosen_payload["preferred_exit_policy"],
        "selection_score_distribution": chosen_payload["threshold_sensitivity"]["selection_score_distribution"],
        "final_test_score_distribution": chosen_payload["threshold_sensitivity"]["final_test_score_distribution"],
        "threshold_grid": {
            "static": chosen_payload["threshold_sensitivity"]["static"],
            "atr_dynamic": chosen_payload["threshold_sensitivity"]["atr_dynamic"],
        },
    }
    reports["strategy_validation_report"]["symbols"][symbol] = {
        "chosen_model": chosen_model,
        "chosen_exit_policy": chosen_payload["preferred_exit_policy"],
        "selection_protocol": symbol_split_reports,
        "walkforward": walkforward,
        "comparison": comparison,
        "decision": decision,
    }
    reports["symbol_enablement_report"]["symbols"][symbol] = decision
    reports["gate_failure_matrix"]["symbols"][symbol] = block_matrix
    reports["session_timeframe_diagnostics_report"]["symbols"][symbol] = session_timeframe_diagnostics
    reports["exit_policy_diagnostics_report"]["symbols"][symbol] = exit_policy_diagnostics
    reports["symbol_recommendation_report"]["symbols"][symbol] = {
        "symbol": symbol,
        "current_validation_state": decision["state"],
        "dominant_reason": block_matrix["dominant_reason"],
        **recommendation,
    }
    reports["strategy_block_root_cause_report"]["symbols"][symbol] = {
        "symbol": symbol,
        "current_validation_state": decision["state"],
        "chosen_model": chosen_model,
        "chosen_exit_policy": chosen_payload["preferred_exit_policy"],
        "chosen_threshold": chosen_payload["thresholds"][chosen_payload["preferred_exit_policy"]]["threshold"],
        "reasons": decision["reasons"],
        "dominant_reason": block_matrix["dominant_reason"],
        "ordered_failures": block_matrix["ordered_failures"],
    }
    reports["symbol_block_diagnostics_report"]["symbols"][symbol] = {
        "symbol": symbol,
        "selection_protocol": symbol_split_reports,
        "decision": decision,
        "gate_failures": block_matrix,
        "session_timeframe": session_timeframe_diagnostics,
        "exit_policy": exit_policy_diagnostics,
    }


def _build_symbol_profile_override(
    settings: Settings,
    *,
    symbol: str,
    decision: dict[str, Any],
    chosen_model: str,
    chosen_payload: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    default_profile = default_symbol_strategy_profile(settings, symbol)
    chosen_threshold = chosen_payload["thresholds"][chosen_payload["preferred_exit_policy"]]["threshold"]
    return {
        "enabled_state": decision["state"],
        "enabled": decision["enabled"],
        "threshold": chosen_threshold,
        "stop_policy": chosen_payload["preferred_exit_policy"],
        "target_policy": chosen_payload["preferred_exit_policy"],
        "stop_atr_multiplier": default_profile.stop_atr_multiplier,
        "target_atr_multiplier": default_profile.target_atr_multiplier,
        "stop_min_pct": default_profile.stop_min_pct,
        "stop_max_pct": default_profile.stop_max_pct,
        "target_min_pct": default_profile.target_min_pct,
        "target_max_pct": default_profile.target_max_pct,
        "notes": f"chosen_model={chosen_model}",
        "profile_id": f"{symbol}-{run_name}",
        "model_variant": chosen_model,
        "source_run_id": run_name,
        "promotion_state": "candidate",
        "promotion_reason": "strategy_validation_generated",
        "rollback_target": None,
    }


def _update_aggregate_portfolio(
    aggregate_portfolio: dict[str, Any],
    decision: dict[str, Any],
    chosen_payload: dict[str, Any],
) -> None:
    if decision["state"] == "enabled":
        aggregate_portfolio["enabled_symbols"] += 1
        aggregate_portfolio["enabled_selection_net_pnl_usd"] += chosen_payload[chosen_payload["preferred_exit_policy"]]["selection_evaluation"]["economic_metrics"]["net_pnl_usd"]
        aggregate_portfolio["enabled_final_test_net_pnl_usd"] += chosen_payload[chosen_payload["preferred_exit_policy"]]["final_test_evaluation"]["economic_metrics"]["net_pnl_usd"]
    elif decision["state"] == "caution":
        aggregate_portfolio["caution_symbols"] += 1
    else:
        aggregate_portfolio["disabled_symbols"] += 1


def _write_validation_reports(
    run_dir: Path,
    *,
    global_split_reports: dict[str, Any],
    symbol_enablement_report: dict[str, Any],
    aggregate_portfolio: dict[str, Any],
    reports: dict[str, dict[str, Any]],
) -> None:
    leakage_fix_report = {
        "protocol": "fit_train -> fit_validation -> selection -> final_test",
        "test_used_for_selection": False,
        "global_split": global_split_reports,
        "symbols": {
            symbol: {
                "chosen_model": payload["chosen_model"],
                "chosen_exit_policy": payload["chosen_exit_policy"],
                "selection_based": payload["selection_based"],
            }
            for symbol, payload in symbol_enablement_report["symbols"].items()
        },
    }
    write_json_report(run_dir, "leakage_fix_report.json", wrap_artifact("strategy_validation", leakage_fix_report))
    write_json_report(run_dir, "strategy_validation_report.json", wrap_artifact("strategy_validation", reports["strategy_validation_report"]))
    write_json_report(run_dir, "model_comparison_report.json", wrap_artifact("model_comparison", reports["model_comparison_report"]))
    write_json_report(run_dir, "dynamic_exit_report.json", wrap_artifact("dynamic_exit_report", reports["dynamic_exit_report"]))
    write_json_report(run_dir, "threshold_report.json", wrap_artifact("threshold_report", reports["threshold_report"]))
    write_json_report(run_dir, "symbol_enablement_report.json", wrap_artifact("symbol_enablement", reports["symbol_enablement_report"]))
    write_json_report(run_dir, "aggregated_portfolio_report.json", wrap_artifact("strategy_validation", aggregate_portfolio))
    write_json_report(run_dir, "strategy_block_root_cause_report.json", wrap_artifact("strategy_validation_audit", reports["strategy_block_root_cause_report"]))
    write_json_report(run_dir, "symbol_block_diagnostics_report.json", wrap_artifact("strategy_validation_audit", reports["symbol_block_diagnostics_report"]))
    write_json_report(run_dir, "threshold_sensitivity_report.json", wrap_artifact("strategy_validation_audit", reports["threshold_sensitivity_report"]))
    write_json_report(run_dir, "gate_failure_matrix.json", wrap_artifact("strategy_validation_audit", reports["gate_failure_matrix"]))
    write_json_report(run_dir, "session_timeframe_diagnostics_report.json", wrap_artifact("strategy_validation_audit", reports["session_timeframe_diagnostics_report"]))
    write_json_report(run_dir, "exit_policy_diagnostics_report.json", wrap_artifact("strategy_validation_audit", reports["exit_policy_diagnostics_report"]))
    write_json_report(run_dir, "symbol_recommendation_report.json", wrap_artifact("strategy_validation_audit", reports["symbol_recommendation_report"]))
