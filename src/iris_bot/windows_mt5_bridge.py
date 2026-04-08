from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
from logging import Logger

from iris_bot.config import Settings


SUPPORTED_WINDOWS_MT5_COMMANDS = frozenset(
    {
        "mt5-check",
        "run-demo-live-probe",
        "demo-execution-preflight",
        "activate-demo-execution",
        "run-demo-execution",
        "reconcile-lifecycle",
    }
)


def is_wsl_runtime() -> bool:
    if platform.system() != "Linux":
        return False
    release = platform.release().lower()
    return bool(os.getenv("WSL_INTEROP") or os.getenv("WSL_DISTRO_NAME") or "microsoft" in release)


def metatrader5_importable() -> bool:
    return importlib.util.find_spec("MetaTrader5") is not None


def requires_windows_mt5_bridge(command: str) -> bool:
    return command in SUPPORTED_WINDOWS_MT5_COMMANDS and is_wsl_runtime() and not metatrader5_importable()


def run_windows_mt5_bridge(settings: Settings, command: str, logger: Logger) -> int:
    if command not in SUPPORTED_WINDOWS_MT5_COMMANDS:
        raise ValueError(f"Unsupported Windows MT5 bridge command: {command}")
    script_path = settings.project_root / "scripts" / "run_mt5_research_windows.sh"
    if not script_path.exists():
        logger.error("windows_mt5_bridge missing script=%s command=%s", script_path, command)
        return 2
    logger.info(
        "windows_mt5_bridge delegating command=%s runtime=wsl_missing_metatrader5 script=%s",
        command,
        script_path,
    )
    try:
        completed = subprocess.run(
            ["bash", str(script_path), command],
            cwd=settings.project_root,
            check=False,
        )
    except FileNotFoundError:
        logger.error(
            "windows_mt5_bridge unavailable command=%s reason=bash_or_powershell_not_found",
            command,
        )
        return 2
    logger.info(
        "windows_mt5_bridge completed command=%s exit_code=%s",
        command,
        completed.returncode,
    )
    return int(completed.returncode)
