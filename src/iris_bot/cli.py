from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

from iris_bot.commands.audit import run_corrective_audit_command
from iris_bot.commands.backtest import (
    run_backtest_command,
    run_paper_command,
    run_walkforward_backtest_command,
)
from iris_bot.commands.data import (
    build_dataset_command,
    fetch_market_data,
    inspect_dataset_command,
    validate_market_data,
)
from iris_bot.commands.governance import (
    active_portfolio_status_command,
    active_strategy_status_command,
    approved_demo_gate_audit_command,
    audit_governance_consistency_command,
    audit_governance_locking_command,
    demo_execution_readiness_command,
    diagnose_profile_activation_command,
    evidence_store_status_cmd,
    list_strategy_profiles_command,
    materialize_active_profiles_command,
    repair_strategy_profile_registry_command,
    promote_strategy_profile_command,
    review_approved_demo_readiness_command,
    rollback_strategy_profile_command,
    symbol_reactivation_readiness_command,
    validate_strategy_profile_command,
)
from iris_bot.commands.lifecycle import (
    lifecycle_audit_report_command,
    reconcile_lifecycle_command,
)
from iris_bot.commands.operations import (
    mt5_check_command,
    run_demo_live_checklist_command,
    run_demo_live_probe_command,
    operational_status_command,
    reconcile_state_command,
    restore_state_check_command,
    run_demo_dry_command,
    run_demo_dry_resilient_command,
    run_paper_resilient_command,
)
from iris_bot.commands.research import (
    audit_strategy_block_causes_command,
    build_symbol_profiles_command,
    compare_symbol_models_command,
    evaluate_dynamic_exits_command,
    run_experiment_command,
    run_strategy_validation_command,
    run_symbol_research_command,
    symbol_go_no_go_command,
)
from iris_bot.commands.soak import (
    audit_endurance_reporting_command,
    go_no_go_report_command,
    run_chaos_scenario_command,
    run_demo_dry_soak_command,
    run_enabled_symbols_soak_command,
    run_paper_soak_command,
    run_symbol_endurance_command,
    symbol_stability_report_command,
)
from iris_bot.config import Settings


CommandHandler = Callable[[Settings], int]
BacktestHandler = Callable[[Settings, str | None], int]

BACKTEST_COMMANDS = frozenset({"run-backtest", "backtest"})


def _fetch_command(settings: Settings) -> int:
    return fetch_market_data(settings)


def build_command_handlers() -> dict[str, CommandHandler | None]:
    return {
        "fetch": _fetch_command,
        "fetch-historical": _fetch_command,
        "validate-data": validate_market_data,
        "build-dataset": build_dataset_command,
        "inspect-dataset": inspect_dataset_command,
        "run-experiment": run_experiment_command,
        "run-backtest": None,
        "backtest": None,
        "run-paper": run_paper_command,
        "mt5-check": mt5_check_command,
        "run-demo-dry": run_demo_dry_command,
        "run-demo-live-checklist": run_demo_live_checklist_command,
        "run-demo-live-probe": run_demo_live_probe_command,
        "reconcile-state": reconcile_state_command,
        "restore-state-check": restore_state_check_command,
        "run-paper-resilient": run_paper_resilient_command,
        "run-demo-dry-resilient": run_demo_dry_resilient_command,
        "operational-status": operational_status_command,
        "run-paper-soak": run_paper_soak_command,
        "run-demo-dry-soak": run_demo_dry_soak_command,
        "run-symbol-endurance": run_symbol_endurance_command,
        "run-enabled-symbols-soak": run_enabled_symbols_soak_command,
        "symbol-stability-report": symbol_stability_report_command,
        "audit-endurance-reporting": audit_endurance_reporting_command,
        "run-chaos-scenario": run_chaos_scenario_command,
        "go-no-go-report": go_no_go_report_command,
        "build-symbol-profiles": build_symbol_profiles_command,
        "run-symbol-research": run_symbol_research_command,
        "run-strategy-validation": run_strategy_validation_command,
        "audit-strategy-block-causes": audit_strategy_block_causes_command,
        "compare-symbol-models": compare_symbol_models_command,
        "evaluate-dynamic-exits": evaluate_dynamic_exits_command,
        "symbol-go-no-go": symbol_go_no_go_command,
        "run-corrective-audit": run_corrective_audit_command,
        "list-strategy-profiles": list_strategy_profiles_command,
        "validate-strategy-profile": validate_strategy_profile_command,
        "review-approved-demo-readiness": review_approved_demo_readiness_command,
        "promote-strategy-profile": promote_strategy_profile_command,
        "rollback-strategy-profile": rollback_strategy_profile_command,
        "active-strategy-status": active_strategy_status_command,
        "diagnose-profile-activation": diagnose_profile_activation_command,
        "audit-governance-consistency": audit_governance_consistency_command,
        "symbol-reactivation-readiness": symbol_reactivation_readiness_command,
        "reconcile-lifecycle": reconcile_lifecycle_command,
        "lifecycle-audit-report": lifecycle_audit_report_command,
        "audit-governance-locking": audit_governance_locking_command,
        "materialize-active-profiles": materialize_active_profiles_command,
        "repair-strategy-profile-registry": repair_strategy_profile_registry_command,
        "evidence-store-status": evidence_store_status_cmd,
        "approved-demo-gate-audit": approved_demo_gate_audit_command,
        "active-portfolio-status": active_portfolio_status_command,
        "demo-execution-readiness": demo_execution_readiness_command,
    }


def command_choices(command_handlers: Mapping[str, CommandHandler | None]) -> tuple[str, ...]:
    return tuple(command_handlers.keys())


def run_cli_command(
    *,
    command: str | None,
    settings: Settings,
    command_handlers: Mapping[str, CommandHandler | None],
    intrabar_policy_override: str | None = None,
    walk_forward: bool = False,
    backtest_handler: BacktestHandler = run_backtest_command,
    walkforward_backtest_handler: BacktestHandler = run_walkforward_backtest_command,
) -> int:
    if command in BACKTEST_COMMANDS or command is None:
        if walk_forward:
            return walkforward_backtest_handler(settings, intrabar_policy_override)
        return backtest_handler(settings, intrabar_policy_override)

    handler = command_handlers[command]
    assert handler is not None
    return handler(settings)
