from __future__ import annotations

from iris_bot.config import Settings
from iris_bot.experiments import run_experiment
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
