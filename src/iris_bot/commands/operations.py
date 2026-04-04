from __future__ import annotations

from iris_bot.config import Settings
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.mt5 import MT5Client, OrderRequest
from iris_bot.paper import ExecutionDecision, OrderIntent, run_paper_session
from iris_bot.resilient import run_operational_status, run_reconcile_state, run_resilient_session, run_restore_state_check


def mt5_check_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "mt5_check")
    logger = configure_logging(run_dir, settings.logging.level)
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
        logger = configure_logging(run_dir, settings.logging.level)
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


def operational_status_command(settings: Settings) -> int:
    exit_code, _ = run_operational_status(settings)
    return exit_code
