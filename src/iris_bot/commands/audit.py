from __future__ import annotations

from iris_bot.config import Settings
from iris_bot.corrective import run_corrective_audit
from iris_bot.demo_trade_audit import run_demo_trade_audit


def run_corrective_audit_command(settings: Settings) -> int:
    return run_corrective_audit(settings)


def run_demo_trade_audit_command(settings: Settings) -> int:
    exit_code, _, _ = run_demo_trade_audit(settings)
    return exit_code
