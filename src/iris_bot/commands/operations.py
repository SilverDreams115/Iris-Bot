from __future__ import annotations

from iris_bot.config import Settings
from iris_bot.demo_execution import (
    activate_demo_execution_command,
    demo_execution_preflight_command,
    demo_execution_status_command,
    run_demo_execution_command,
    validate_model_artifact_command,
)
from iris_bot.demo_session_series import (
    close_demo_session_series_command,
    demo_forward_series_status_command,
    start_demo_session_series_command,
)
from iris_bot.demo_live_checklist import demo_live_checklist_command
from iris_bot.demo_live_probe import run_demo_live_probe
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.mt5 import MT5Client, OrderRequest
from iris_bot.paper import ExecutionDecision, OrderIntent, run_paper_session
from iris_bot.resilient import (
    run_operational_status,
    run_reconcile_state,
    run_reconciliation_drills,
    run_recovery_drills,
    run_resilient_session,
    run_restore_safety_drill,
    run_restore_state_check,
)
from iris_bot.windows_mt5_bridge import requires_windows_mt5_bridge, run_windows_mt5_bridge


def mt5_check_command(settings: Settings) -> int:
    if requires_windows_mt5_bridge("mt5-check"):
        run_dir = build_run_directory(settings.data.runs_dir, "mt5_check")
        logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
        return run_windows_mt5_bridge(settings, "mt5-check", logger)
    run_dir = build_run_directory(settings.data.runs_dir, "mt5_check")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    client = MT5Client(settings.mt5)
    if not client.connect():
        logger.error("No se pudo inicializar la conexion MT5")
        write_json_report(run_dir, "validation_report.json", {"ok": False, "connected": False, "terminal_initialized": False, "issues": ["connect_failed"]})
        return 1
    try:
        report = client.check(settings.trading.symbols)
        write_json_report(run_dir, "validation_report.json", report.to_dict())
        logger.info("mt5_check ok=%s issues=%s", report.ok, len(report.issues))
        return 0 if report.ok else 2
    finally:
        client.shutdown()


def run_demo_dry_command(settings: Settings) -> int:
    client = MT5Client(settings.mt5)
    if not client.connect():
        run_dir = build_run_directory(settings.data.runs_dir, "demo_dry")
        logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
        logger.error("No se pudo inicializar la conexion MT5 para demo dry-run")
        write_json_report(run_dir, "validation_report.json", {"ok": False, "connected": False, "terminal_initialized": False, "issues": ["connect_failed"]})
        return 1

    def validator(intent: OrderIntent) -> ExecutionDecision:
        result = client.dry_run_market_order(
            OrderRequest(
                symbol=intent.symbol,
                side=intent.side,
                volume=intent.volume,
                stop_loss=intent.stop_loss,
                take_profit=intent.take_profit,
                price=intent.entry_price,
            )
        )
        return ExecutionDecision(accepted=result.accepted, reason=result.reason, details=result.to_dict())

    try:
        exit_code, _ = run_paper_session(settings, mode="demo_dry", execution_validator=validator)
        return exit_code
    finally:
        client.shutdown()


def reconcile_state_command(settings: Settings) -> int:
    exit_code, _ = run_reconcile_state(settings)
    return exit_code


def restore_state_check_command(settings: Settings) -> int:
    exit_code, _ = run_restore_state_check(settings)
    return exit_code


def run_paper_resilient_command(settings: Settings) -> int:
    exit_code, _ = run_resilient_session(settings, mode="paper", require_broker=False)
    return exit_code


def run_demo_dry_resilient_command(settings: Settings) -> int:
    exit_code, _ = run_resilient_session(settings, mode="demo_dry", require_broker=True)
    return exit_code


def run_demo_live_probe_command(settings: Settings) -> int:
    if requires_windows_mt5_bridge("run-demo-live-probe"):
        run_dir = build_run_directory(settings.data.runs_dir, "demo_live_probe")
        logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
        return run_windows_mt5_bridge(settings, "run-demo-live-probe", logger)
    exit_code, _, _ = run_demo_live_probe(settings)
    return exit_code


def run_demo_live_checklist_command(settings: Settings) -> int:
    exit_code, _, _ = demo_live_checklist_command(settings)
    return exit_code


def validate_model_artifact_runtime_command(settings: Settings) -> int:
    return validate_model_artifact_command(settings)


def activate_demo_execution_runtime_command(settings: Settings) -> int:
    if requires_windows_mt5_bridge("activate-demo-execution"):
        run_dir = build_run_directory(settings.data.runs_dir, "activate_demo_execution")
        logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
        return run_windows_mt5_bridge(settings, "activate-demo-execution", logger)
    return activate_demo_execution_command(settings)


def demo_execution_preflight_runtime_command(settings: Settings) -> int:
    if requires_windows_mt5_bridge("demo-execution-preflight"):
        run_dir = build_run_directory(settings.data.runs_dir, "demo_execution_preflight")
        logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
        return run_windows_mt5_bridge(settings, "demo-execution-preflight", logger)
    return demo_execution_preflight_command(settings)


def run_demo_execution_runtime_command(settings: Settings) -> int:
    if requires_windows_mt5_bridge("run-demo-execution"):
        run_dir = build_run_directory(settings.data.runs_dir, "run_demo_execution")
        logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
        return run_windows_mt5_bridge(settings, "run-demo-execution", logger)
    return run_demo_execution_command(settings)


def demo_execution_status_runtime_command(settings: Settings) -> int:
    return demo_execution_status_command(settings)


def start_demo_forward_series_runtime_command(settings: Settings) -> int:
    return start_demo_session_series_command(settings)


def demo_forward_series_status_runtime_command(settings: Settings) -> int:
    return demo_forward_series_status_command(settings)


def close_demo_forward_series_runtime_command(settings: Settings) -> int:
    return close_demo_session_series_command(settings)


def operational_status_command(settings: Settings) -> int:
    exit_code, _ = run_operational_status(settings)
    return exit_code


def restore_safety_drill_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "restore_safety_drill")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    exit_code, report = run_restore_safety_drill(settings)
    write_json_report(run_dir, "restore_safety_drill_report.json", report)
    logger.info("restore_safety_drill ok=%s failed=%s", report["ok"], report.get("failed_checks", []))
    return exit_code


def reconciliation_drills_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "reconciliation_drills")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    exit_code, report = run_reconciliation_drills(settings)
    write_json_report(run_dir, "reconciliation_drills_report.json", report)
    logger.info("reconciliation_drills ok=%s failed=%s", report["ok"], report.get("failed_scenarios", []))
    return exit_code


def recovery_drills_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "recovery_drills")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    exit_code, report = run_recovery_drills(settings)
    write_json_report(run_dir, "recovery_drills_report.json", report)
    logger.info("recovery_drills ok=%s failed=%s", report["ok"], report.get("failed_scenarios", []))
    return exit_code
