from __future__ import annotations

from iris_bot.config import Settings
from iris_bot.edge_diagnostics import (
    run_audit_class_separability,
    run_audit_edge_baseline,
    run_audit_edge_hypotheses,
    run_audit_horizon_exits,
    run_audit_label_noise,
    run_audit_regime_value,
)
from iris_bot.exit_lifecycle_realignment import (
    run_audit_exit_lifecycle,
    run_compare_h12_exit_variants,
    run_evaluate_gbpusd_demo_candidate,
    run_h12_exit_realignment,
)
from iris_bot.experiments import run_experiment
from iris_bot.label_horizon_realignment import (
    run_audit_timeout_impact,
    run_audit_trade_duration,
    run_compare_exit_alignment,
    run_evaluate_label_horizon_candidate,
    run_label_horizon_realignment,
)
from iris_bot.quant_experiments import (
    audit_effective_config_command,
    audit_symbol_context_command,
    compare_experiment_results_command,
    run_experiment_matrix,
)
from iris_bot.regime_rework import (
    run_audit_regime_features,
    run_compare_regime_experiments,
    run_evaluate_regime_demo_candidate,
    run_fetch_extended_history,
    run_regime_aware_rework,
)
from iris_bot.structural_rework import run_structural_rework_evaluation
from iris_bot.symbol_focused_rework import (
    run_audit_symbol_signal,
    run_compare_symbol_variants,
    run_evaluate_demo_execution_candidate,
    run_symbol_structural_rework,
)
from iris_bot.symbol_research import run_symbol_research
from iris_bot.symbol_validation import (
    audit_strategy_block_causes,
    build_symbol_profiles,
    compare_symbol_models,
    evaluate_dynamic_exits,
    run_strategy_validation,
    symbol_go_no_go,
)


def run_experiment_command(settings: Settings) -> int:
    return run_experiment(settings)


def run_experiment_matrix_command(settings: Settings) -> int:
    return run_experiment_matrix(settings)


def audit_effective_runtime_config_command(settings: Settings) -> int:
    return audit_effective_config_command(settings)


def audit_symbol_context_runtime_command(settings: Settings) -> int:
    return audit_symbol_context_command(settings)


def compare_experiment_results_runtime_command(settings: Settings) -> int:
    return compare_experiment_results_command(settings)


def run_structural_rework_evaluation_command(settings: Settings) -> int:
    return run_structural_rework_evaluation(settings)


def build_symbol_profiles_command(settings: Settings) -> int:
    return build_symbol_profiles(settings)


def run_symbol_research_command(settings: Settings) -> int:
    return run_symbol_research(settings)


def run_strategy_validation_command(settings: Settings) -> int:
    return run_strategy_validation(settings)


def compare_symbol_models_command(settings: Settings) -> int:
    return compare_symbol_models(settings)


def evaluate_dynamic_exits_command(settings: Settings) -> int:
    return evaluate_dynamic_exits(settings)


def symbol_go_no_go_command(settings: Settings) -> int:
    return symbol_go_no_go(settings)


def audit_strategy_block_causes_command(settings: Settings) -> int:
    return audit_strategy_block_causes(settings)


def run_symbol_structural_rework_command(settings: Settings) -> int:
    return run_symbol_structural_rework(settings)


def audit_symbol_signal_command(settings: Settings) -> int:
    return run_audit_symbol_signal(settings)


def compare_symbol_variants_command(settings: Settings) -> int:
    return run_compare_symbol_variants(settings)


def evaluate_demo_execution_candidate_command(settings: Settings) -> int:
    return run_evaluate_demo_execution_candidate(settings)


def fetch_extended_history_command(settings: Settings) -> int:
    return run_fetch_extended_history(settings)


def audit_regime_features_command(settings: Settings) -> int:
    return run_audit_regime_features(settings)


def run_regime_aware_rework_command(settings: Settings) -> int:
    return run_regime_aware_rework(settings)


def compare_regime_experiments_command(settings: Settings) -> int:
    return run_compare_regime_experiments(settings)


def evaluate_regime_demo_candidate_command(settings: Settings) -> int:
    return run_evaluate_regime_demo_candidate(settings)


def audit_edge_baseline_command(settings: Settings) -> int:
    return run_audit_edge_baseline(settings)


def audit_label_noise_command(settings: Settings) -> int:
    return run_audit_label_noise(settings)


def audit_horizon_exits_command(settings: Settings) -> int:
    return run_audit_horizon_exits(settings)


def audit_regime_value_command(settings: Settings) -> int:
    return run_audit_regime_value(settings)


def audit_class_separability_command(settings: Settings) -> int:
    return run_audit_class_separability(settings)


def audit_edge_hypotheses_command(settings: Settings) -> int:
    return run_audit_edge_hypotheses(settings)


def audit_trade_duration_command(settings: Settings) -> int:
    return run_audit_trade_duration(settings)


def audit_timeout_impact_command(settings: Settings) -> int:
    return run_audit_timeout_impact(settings)


def run_label_horizon_realignment_command(settings: Settings) -> int:
    return run_label_horizon_realignment(settings)


def compare_exit_alignment_command(settings: Settings) -> int:
    return run_compare_exit_alignment(settings)


def evaluate_label_horizon_candidate_command(settings: Settings) -> int:
    return run_evaluate_label_horizon_candidate(settings)


def audit_exit_lifecycle_command(settings: Settings) -> int:
    return run_audit_exit_lifecycle(settings)


def run_h12_exit_realignment_command(settings: Settings) -> int:
    return run_h12_exit_realignment(settings)


def compare_h12_exit_variants_command(settings: Settings) -> int:
    return run_compare_h12_exit_variants(settings)


def evaluate_gbpusd_demo_candidate_command(settings: Settings) -> int:
    return run_evaluate_gbpusd_demo_candidate(settings)
