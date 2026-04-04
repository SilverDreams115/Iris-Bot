from __future__ import annotations

from iris_bot.config import Settings
from iris_bot.lifecycle import lifecycle_audit_report, run_lifecycle_reconciliation


def reconcile_lifecycle_command(settings: Settings) -> int:
    exit_code, _ = run_lifecycle_reconciliation(settings)
    return exit_code


def lifecycle_audit_report_command(settings: Settings) -> int:
    exit_code, _ = lifecycle_audit_report(settings)
    return exit_code
