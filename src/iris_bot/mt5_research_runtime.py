from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from iris_bot.artifacts import wrap_artifact
from iris_bot.cli import build_command_handlers, run_cli_command
from iris_bot.config import env_source, load_settings
from iris_bot.logging_utils import configure_logging, write_json_report
from iris_bot.mt5 import MT5Client
from iris_bot.runtime_provenance import load_runtime_provenance_from_env


SUPPORTED_MT5_RESEARCH_COMMANDS = (
    "fetch-extended-history",
    "audit-regime-features",
    "run-regime-aware-rework",
    "compare-regime-experiments",
    "evaluate-demo-candidate",
)


@dataclass(frozen=True)
class RuntimeProbe:
    platform_system: str
    platform_release: str
    python_executable: str
    python_version: str
    project_root: str
    runs_dir: str
    raw_dir: str
    runtime_provenance: dict[str, Any]
    metatrader5_importable: bool
    metatrader5_module_path: str | None
    xgboost_importable: bool
    mt5_env_sources: dict[str, str]


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    issues: list[str]
    terminal_accessible: bool
    account_accessible: bool
    runs_dir_writable: bool
    src_importable: bool
    mt5_connected: bool
    mt5_last_error: object
    terminal_path: str | None
    account_server: str | None
    symbol_universe: list[str]
    timeframes: list[str]


def _importable(module_name: str) -> tuple[bool, str | None]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return False, None
    return True, spec.origin


def _runtime_probe() -> RuntimeProbe:
    settings = load_settings()
    mt5_ok, mt5_origin = _importable("MetaTrader5")
    xgb_ok, _ = _importable("xgboost")
    return RuntimeProbe(
        platform_system=platform.system(),
        platform_release=platform.release(),
        python_executable=sys.executable,
        python_version=sys.version,
        project_root=str(settings.project_root),
        runs_dir=str(settings.data.runs_dir),
        raw_dir=str(settings.data.raw_dir),
        runtime_provenance=load_runtime_provenance_from_env(),
        metatrader5_importable=mt5_ok,
        metatrader5_module_path=mt5_origin,
        xgboost_importable=xgb_ok,
        mt5_env_sources={
            "IRIS_MT5_ENABLED": env_source("IRIS_MT5_ENABLED"),
            "IRIS_MT5_LOGIN": env_source("IRIS_MT5_LOGIN"),
            "IRIS_MT5_PASSWORD": env_source("IRIS_MT5_PASSWORD"),
            "IRIS_MT5_SERVER": env_source("IRIS_MT5_SERVER"),
            "IRIS_MT5_PATH": env_source("IRIS_MT5_PATH"),
        },
    )


def _runs_dir_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".mt5_research_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _preflight(command: str) -> PreflightResult:
    settings = load_settings()
    issues: list[str] = []
    runtime = _runtime_probe()

    if runtime.platform_system != "Windows":
        issues.append("platform_not_windows")
    if not runtime.metatrader5_importable:
        issues.append("metatrader5_not_importable")
    if command in {"run-regime-aware-rework", "compare-regime-experiments", "evaluate-demo-candidate"} and not runtime.xgboost_importable:
        issues.append("xgboost_not_importable")

    runs_dir_writable = _runs_dir_writable(settings.data.runs_dir)
    if not runs_dir_writable:
        issues.append("runs_dir_not_writable")

    src_importable = "iris_bot" in sys.modules or importlib.util.find_spec("iris_bot") is not None
    if not src_importable:
        issues.append("src_not_importable")

    terminal_accessible = False
    account_accessible = False
    mt5_connected = False
    mt5_last_error: object = None
    terminal_path: str | None = settings.mt5.path
    account_server: str | None = settings.mt5.server

    client = MT5Client(settings.mt5)
    if client.connect():
        mt5_connected = True
        health = client.session_health()
        terminal_accessible = bool(health and health.terminal_available)
        account_accessible = bool(health and health.account_accessible)
        mt5_last_error = client.last_error()
        info = client.account_info() or {}
        account_server = str(info.get("server") or settings.mt5.server or "")
        try:
            if client._mt5 is not None and hasattr(client._mt5, "terminal_info"):  # noqa: SLF001
                terminal = client._mt5.terminal_info()  # noqa: SLF001
                if terminal is not None:
                    if hasattr(terminal, "_asdict"):
                        terminal_data = terminal._asdict()
                    elif isinstance(terminal, dict):
                        terminal_data = terminal
                    else:
                        terminal_data = {}
                    terminal_path = str(terminal_data.get("path") or terminal_data.get("data_path") or settings.mt5.path or "")
        finally:
            client.shutdown()
    else:
        issues.append("mt5_connect_failed")
        mt5_last_error = client.last_error()

    if mt5_connected and not terminal_accessible:
        issues.append("mt5_terminal_unavailable")
    if mt5_connected and not account_accessible:
        issues.append("mt5_account_unavailable")

    return PreflightResult(
        ok=not issues,
        issues=issues,
        terminal_accessible=terminal_accessible,
        account_accessible=account_accessible,
        runs_dir_writable=runs_dir_writable,
        src_importable=src_importable,
        mt5_connected=mt5_connected,
        mt5_last_error=mt5_last_error,
        terminal_path=terminal_path,
        account_server=account_server,
        symbol_universe=list(settings.trading.symbols),
        timeframes=list(settings.trading.timeframes),
    )


def _prepare_run_dir(run_id: str) -> Path:
    settings = load_settings()
    run_dir = settings.data.runs_dir / f"{run_id}_mt5_research_execution"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _latest_matching_run(runs_dir: Path, suffix: str, before: set[str]) -> str | None:
    candidates = sorted(path.name for path in runs_dir.glob(f"*_{suffix}"))
    for candidate in reversed(candidates):
        if candidate not in before:
            return candidate
    return candidates[-1] if candidates else None


def _execution_payload(run_id: str, command: str, delegated_run_dir: str | None, exit_code: int) -> dict[str, Any]:
    return wrap_artifact(
        "mt5_research_execution_report",
        {
            "run_id": run_id,
            "command": command,
            "delegated_run_dir": delegated_run_dir,
            "exit_code": exit_code,
            "runtime_provenance": load_runtime_provenance_from_env(),
        },
    )


def validate_runtime_only(run_id: str) -> int:
    settings = load_settings()
    run_dir = _prepare_run_dir(run_id)
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    runtime_report = wrap_artifact("mt5_research_runtime_report", asdict(_runtime_probe()))
    preflight = _preflight("fetch-extended-history")
    preflight_report = wrap_artifact("mt5_research_preflight_report", asdict(preflight))
    write_json_report(run_dir, "mt5_research_runtime_report.json", runtime_report)
    write_json_report(run_dir, "mt5_research_preflight_report.json", preflight_report)
    execution = _execution_payload(run_id, "validate-runtime", None, 0 if preflight.ok else 2)
    write_json_report(run_dir, "mt5_research_execution_report.json", execution)
    (settings.project_root / ".mt5_research_last_execution.json").write_text(
        json.dumps({"command": "validate-runtime", "run_dir_name": run_dir.name, "delegated_run_dir_name": None, "exit_code": 0 if preflight.ok else 2}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("mt5_research_validate_runtime ok=%s", preflight.ok)
    return 0 if preflight.ok else 2


def run_runtime(command: str, run_id: str) -> int:
    if command not in SUPPORTED_MT5_RESEARCH_COMMANDS:
        raise ValueError(f"Unsupported MT5 research command: {command}")

    settings = load_settings()
    run_dir = _prepare_run_dir(run_id)
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    runtime_report = wrap_artifact("mt5_research_runtime_report", asdict(_runtime_probe()))
    preflight = _preflight(command)
    preflight_report = wrap_artifact("mt5_research_preflight_report", asdict(preflight))
    write_json_report(run_dir, "mt5_research_runtime_report.json", runtime_report)
    write_json_report(run_dir, "mt5_research_preflight_report.json", preflight_report)

    if not preflight.ok:
        execution = _execution_payload(run_id, command, None, 2)
        write_json_report(run_dir, "mt5_research_execution_report.json", execution)
        (settings.project_root / ".mt5_research_last_execution.json").write_text(
            json.dumps({"command": command, "run_dir_name": run_dir.name, "delegated_run_dir_name": None, "exit_code": 2}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.error("mt5_research_preflight_failed issues=%s", ",".join(preflight.issues))
        return 2

    runs_before = {path.name for path in settings.data.runs_dir.iterdir() if path.is_dir()}
    suffix = command.replace("-", "_")
    handlers = build_command_handlers()
    exit_code = run_cli_command(command=command, settings=settings, command_handlers=handlers)
    delegated_run_dir = _latest_matching_run(settings.data.runs_dir, suffix, runs_before)

    provenance_report = wrap_artifact(
        "environment_provenance_report",
        {
            "run_id": run_id,
            "command": command,
            "python_executable": sys.executable,
            "platform_system": platform.system(),
            "terminal_path": preflight.terminal_path,
            "account_server": preflight.account_server,
            "symbol_universe": preflight.symbol_universe,
            "timeframes": preflight.timeframes,
            "runtime_provenance": load_runtime_provenance_from_env(),
        },
    )
    if delegated_run_dir is not None:
        delegated_path = settings.data.runs_dir / delegated_run_dir / "environment_provenance_report.json"
        delegated_path.write_text(json.dumps(provenance_report, indent=2, sort_keys=True), encoding="utf-8")

    execution = _execution_payload(
        run_id,
        command,
        delegated_run_dir,
        exit_code,
    )
    debt_report = wrap_artifact(
        "technical_debt_avoidance",
        {
            "single_recommended_flow": "wsl_tar_bridge_to_windows_native_python",
            "avoided_shortcuts": [
                "no_unc_execution_of_project_code",
                "no_duplicate_quant_logic_in_wrapper",
                "no_secret_persistence_in_repo",
                "no_parallel_shadow_project_for_research",
            ],
        },
    )
    write_json_report(run_dir, "mt5_research_execution_report.json", execution)
    write_json_report(run_dir, "technical_debt_avoidance_report.json", debt_report)
    (settings.project_root / ".mt5_research_last_execution.json").write_text(
        json.dumps(
            {
                "command": command,
                "run_dir_name": run_dir.name,
                "delegated_run_dir_name": delegated_run_dir,
                "exit_code": exit_code,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    logger.info("mt5_research_execution command=%s exit_code=%s delegated_run=%s", command, exit_code, delegated_run_dir)
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Windows-native MT5 research runtime")
    parser.add_argument("command", choices=("validate-runtime", "run"))
    parser.add_argument("--delegated-command", choices=SUPPORTED_MT5_RESEARCH_COMMANDS, default="fetch-extended-history")
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate-runtime":
        raise SystemExit(validate_runtime_only(args.run_id))
    raise SystemExit(run_runtime(args.delegated_command, args.run_id))


if __name__ == "__main__":
    main()
