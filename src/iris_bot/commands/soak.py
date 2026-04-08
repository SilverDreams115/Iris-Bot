from __future__ import annotations

from iris_bot.config import Settings
from iris_bot.soak import regenerate_go_no_go_report, run_chaos_scenario, run_demo_guarded_soak, run_soak
from iris_bot.symbol_endurance import audit_endurance_reporting, run_symbol_endurance, symbol_stability_report


def run_paper_soak_command(settings: Settings) -> int:
    exit_code, _ = run_soak(settings, mode="paper", require_broker=False)
    return exit_code


def run_demo_guarded_soak_command(settings: Settings) -> int:
    exit_code, _ = run_demo_guarded_soak(settings, mode="paper", require_broker=False)
    return exit_code


def run_demo_dry_soak_command(settings: Settings) -> int:
    exit_code, _ = run_soak(settings, mode="demo_dry", require_broker=True)
    return exit_code


def run_chaos_scenario_command(settings: Settings) -> int:
    mode = "demo_dry" if settings.mt5.enabled else "paper"
    exit_code, _ = run_chaos_scenario(settings, mode=mode, require_broker=settings.mt5.enabled)
    return exit_code


def go_no_go_report_command(settings: Settings) -> int:
    exit_code, _ = regenerate_go_no_go_report(settings)
    return exit_code


def run_symbol_endurance_command(settings: Settings) -> int:
    exit_code, _ = run_symbol_endurance(settings, only_enabled=False)
    return exit_code


def run_enabled_symbols_soak_command(settings: Settings) -> int:
    exit_code, _ = run_symbol_endurance(settings, only_enabled=True)
    return exit_code


def symbol_stability_report_command(settings: Settings) -> int:
    exit_code, _ = symbol_stability_report(settings)
    return exit_code


def audit_endurance_reporting_command(settings: Settings) -> int:
    exit_code, _ = audit_endurance_reporting(settings)
    return exit_code
