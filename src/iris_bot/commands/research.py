from __future__ import annotations

from iris_bot.config import Settings
from iris_bot.experiments import run_experiment
from iris_bot.quant_experiments import (
    audit_effective_config_command,
    audit_symbol_context_command,
    compare_experiment_results_command,
    run_experiment_matrix,
)
from iris_bot.structural_rework import run_structural_rework_evaluation
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
