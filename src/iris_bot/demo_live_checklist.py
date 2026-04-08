from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from iris_bot.artifacts import wrap_artifact
from iris_bot.config import Settings
from iris_bot.demo_readiness import generate_demo_execution_readiness_report
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.mt5 import MT5Client


@dataclass(frozen=True)
class ChecklistItem:
    ok: bool
    reason: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DemoLiveChecklistReport:
    decision: str
    all_required_ok: bool
    failed_checks: list[str]
    checks: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _credentials_check(settings: Settings) -> ChecklistItem:
    mt5 = settings.mt5
    missing = []
    if not mt5.enabled:
        missing.append("enabled")
    if not mt5.login:
        missing.append("login")
    if not mt5.password:
        missing.append("password")
    if not mt5.server:
        missing.append("server")
    ok = len(missing) == 0
    return ChecklistItem(
        ok=ok,
        reason="ok" if ok else f"missing_mt5_fields:{','.join(missing)}",
        details={
            "enabled": mt5.enabled,
            "login_configured": mt5.login is not None,
            "password_configured": mt5.password is not None,
            "server_configured": mt5.server is not None,
            "path_configured": mt5.path is not None,
        },
    )


def _mt5_connectivity_check(settings: Settings, client_factory: Callable[[], MT5Client] | None = None) -> ChecklistItem:
    client = client_factory() if client_factory is not None else MT5Client(settings.mt5)
    if not client.connect():
        return ChecklistItem(ok=False, reason="connect_failed", details={"last_error": client.last_error()})
    try:
        report = client.check(settings.trading.symbols)
        return ChecklistItem(
            ok=report.ok,
            reason="ok" if report.ok else "symbol_validation_failed",
            details=report.to_dict(),
        )
    finally:
        client.shutdown()


def _demo_readiness_check(settings: Settings) -> ChecklistItem:
    report = generate_demo_execution_readiness_report(settings)
    ok = report["decision"] in {"ready_for_demo_guarded", "ready_for_demo_with_reservations"}
    return ChecklistItem(
        ok=ok,
        reason="ok" if ok else f"readiness_decision:{report['decision']}",
        details=report,
    )


def _latest_probe_check(settings: Settings) -> ChecklistItem:
    candidates = sorted(settings.data.runs_dir.glob("*_demo_live_probe/demo_live_probe_report.json"))
    if not candidates:
        return ChecklistItem(ok=False, reason="no_probe_report", details={})
    latest = candidates[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))
    ok = bool(payload.get("ok", False))
    return ChecklistItem(
        ok=ok,
        reason="ok" if ok else str(payload.get("reason", "probe_failed")),
        details={"path": str(latest), "report": payload},
    )


def generate_demo_live_checklist_report(
    settings: Settings,
    client_factory: Callable[[], MT5Client] | None = None,
) -> DemoLiveChecklistReport:
    checks = {
        "mt5_credentials": _credentials_check(settings),
        "demo_execution_readiness": _demo_readiness_check(settings),
        "mt5_connectivity": _mt5_connectivity_check(settings, client_factory=client_factory),
        "latest_demo_live_probe": _latest_probe_check(settings),
    }
    required = ["mt5_credentials", "demo_execution_readiness", "mt5_connectivity", "latest_demo_live_probe"]
    failed = [name for name in required if not checks[name].ok]
    decision = "ready" if not failed else "not_ready"
    return DemoLiveChecklistReport(
        decision=decision,
        all_required_ok=len(failed) == 0,
        failed_checks=failed,
        checks={name: item.to_dict() for name, item in checks.items()},
    )


def demo_live_checklist_command(
    settings: Settings,
    client_factory: Callable[[], MT5Client] | None = None,
) -> tuple[int, Path, DemoLiveChecklistReport]:
    run_dir = build_run_directory(settings.data.runs_dir, "demo_live_checklist")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    report = generate_demo_live_checklist_report(settings, client_factory=client_factory)
    write_json_report(run_dir, "demo_live_checklist_report.json", wrap_artifact("demo_live_checklist", report.to_dict()))
    logger.info("demo_live_checklist decision=%s failed=%s run_dir=%s", report.decision, report.failed_checks, run_dir)
    return (0 if report.all_required_ok else 1), run_dir, report
