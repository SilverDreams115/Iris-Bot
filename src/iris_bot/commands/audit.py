from __future__ import annotations

from iris_bot.config import Settings
from iris_bot.corrective import run_corrective_audit


def run_corrective_audit_command(settings: Settings) -> int:
    return run_corrective_audit(settings)
