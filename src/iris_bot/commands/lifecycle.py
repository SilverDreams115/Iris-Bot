from __future__ import annotations

from iris_bot.config import Settings
from iris_bot.lifecycle import lifecycle_audit_report, run_lifecycle_reconciliation
from iris_bot.logging_utils import build_run_directory, configure_logging
from iris_bot.windows_mt5_bridge import requires_windows_mt5_bridge, run_windows_mt5_bridge


def reconcile_lifecycle_command(settings: Settings) -> int:
    if requires_windows_mt5_bridge("reconcile-lifecycle"):
        run_dir = build_run_directory(settings.data.runs_dir, "lifecycle_reconciliation")
        logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
        return run_windows_mt5_bridge(settings, "reconcile-lifecycle", logger)
    exit_code, _ = run_lifecycle_reconciliation(settings)
    return exit_code


def lifecycle_audit_report_command(settings: Settings) -> int:
    exit_code, _ = lifecycle_audit_report(settings)
    return exit_code
