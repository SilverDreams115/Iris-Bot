from __future__ import annotations

import csv
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, cast

from iris_bot.config import SessionConfig, Settings
from iris_bot.logging_utils import build_run_directory, configure_logging
from iris_bot.mt5 import BrokerSnapshot, DryRunOrderResult, MT5Client, OrderRequest
from iris_bot.operational import write_json, write_rows_csv
from iris_bot.resilient import (
    build_runtime_state_path,
    run_resilient_session,
)


@dataclass(frozen=True)
class CycleHealth:
    cycle: int
    status: str
    restore_ok: bool
    reconciliation_ok: bool
    validation_ok: bool
    duplicate_event_count: int
    orphan_open_positions: int
    alerts_by_severity: dict[str, int]
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoNoGoDecision:
    decision: str
    reasons: list[str]
    severity_summary: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChaosMT5Client(MT5Client):
    def __init__(
        self,
        base_factory: Callable[[], MT5Client] | None,
        scenario_names: set[str],
        cycle_index: int,
        every_n_cycles: int,
    ) -> None:
        self._base = base_factory() if base_factory is not None else None
        super().__init__(self._base.config if self._base is not None else __import__("iris_bot.config", fromlist=["MT5Config"]).MT5Config(enabled=True))
        self._scenario_names = scenario_names
        self._cycle_index = cycle_index
        self._every_n_cycles = every_n_cycles
        self._connect_calls = 0

    def _active(self, name: str) -> bool:
        if name not in self._scenario_names:
            return False
        if self._every_n_cycles > 0:
            return self._cycle_index % self._every_n_cycles == 0
        if name.endswith("_once"):
            return self._cycle_index == 1
        return True

    def connect(self) -> bool:
        self._connect_calls += 1
        if self._active("reconnect_fail_once"):
            self._connected = False
            return False
        if self._active("disconnect_once") and self._connect_calls == 1:
            self._connected = False
            return False
        if self._base is None:
            self._connected = True
            return True
        ok = self._base.connect()
        self._connected = ok
        return ok

    def shutdown(self) -> None:
        if self._base is not None:
            self._base.shutdown()
        self._connected = False

    def last_error(self) -> object:
        if not self._connected:
            return (500, "Chaos disconnect")
        if self._base is not None:
            return self._base.last_error()
        return (1, "Success")

    def broker_state_snapshot(self, symbols: tuple[str, ...]) -> BrokerSnapshot:
        if self._base is not None:
            snapshot = self._base.broker_state_snapshot(symbols)
        else:
            snapshot = BrokerSnapshot(True, {"balance": 1000.0, "equity": 1000.0}, [], [], [])
        if self._active("broker_mismatch_once"):
            fake_position = {
                "ticket": 999,
                "symbol": symbols[0] if symbols else "EURUSD",
                "type": 0,
                "volume": 0.5,
                "price_open": 1.3000,
                "sl": 1.2900,
                "tp": 1.3200,
                "time": 1,
            }
            snapshot.positions.append(fake_position)
        return snapshot

    def dry_run_market_order(self, order: OrderRequest) -> DryRunOrderResult:
        if self._active("repeated_rejections"):
            return DryRunOrderResult(False, "not enough money", None, None, [])
        if self._active("communication_error_once"):
            return DryRunOrderResult(False, "communication error", None, None, [])
        if self._base is not None:
            return self._base.dry_run_market_order(order)
        return DryRunOrderResult(True, "dry_run_only", order.volume, {"symbol": order.symbol}, [])


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return cast(dict[str, Any], raw)


def _read_alerts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _count_duplicate_events(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen: set[tuple[str, str, str]] = set()
    duplicates = 0
    for row in rows:
        key = (row["timestamp"], row["symbol"], row["event_type"])
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def _cycle_health(cycle: int, cycle_dir: Path) -> CycleHealth:
    issues: list[str] = []
    try:
        restore = _read_json(cycle_dir / "restore_state_report.json")
        reconciliation = _read_json(cycle_dir / "reconciliation_report.json")
        validation = _read_json(cycle_dir / "validation_report.json")
        open_snapshot = _read_json(cycle_dir / "open_positions_snapshot.json")
        alerts = _read_alerts(cycle_dir / "alerts_log.jsonl")
    except FileNotFoundError:
        restore = {"ok": False}
        reconciliation = {"ok": False}
        validation = {"ok": False}
        open_snapshot = {}
        alerts = []
        issues.append("missing_cycle_artifacts")
    severity_summary = {"info": 0, "warning": 0, "error": 0, "critical": 0}
    for item in alerts:
        severity = item.get("severity", "info")
        severity_summary[severity] = severity_summary.get(severity, 0) + 1
    duplicate_event_count = _count_duplicate_events(cycle_dir / "execution_journal.csv")
    orphan_open_positions = len(open_snapshot.get("open_positions", {})) if open_snapshot.get("current_session_status", {}).get("status") == "completed" else 0
    if not restore.get("ok", False):
        issues.append("restore_failed")
    if reconciliation and not reconciliation.get("ok", True):
        issues.append("reconciliation_failed")
    if not validation.get("ok", validation.get("is_valid", False)):
        issues.append("validation_failed")
    if duplicate_event_count > 0:
        issues.append("duplicate_events")
    if severity_summary.get("critical", 0) > 0:
        issues.append("critical_alerts")
    if orphan_open_positions > 0:
        issues.append("orphan_open_positions")
    status = "go"
    if any(item in issues for item in {"restore_failed", "reconciliation_failed", "validation_failed", "critical_alerts", "orphan_open_positions"}):
        status = "no_go"
    elif issues or severity_summary.get("warning", 0) or severity_summary.get("error", 0):
        status = "caution"
    return CycleHealth(
        cycle=cycle,
        status=status,
        restore_ok=restore.get("ok", False),
        reconciliation_ok=reconciliation.get("ok", True),
        validation_ok=validation.get("ok", validation.get("is_valid", False)),
        duplicate_event_count=duplicate_event_count,
        orphan_open_positions=orphan_open_positions,
        alerts_by_severity=severity_summary,
        issues=issues,
    )


def _aggregate_csv(target: Path, sources: list[Path]) -> None:
    rows: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    for source in sources:
        if not source.exists():
            continue
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not fieldnames:
                fieldnames = list(reader.fieldnames or [])
            for row in reader:
                rows.append(row)
    if fieldnames:
        write_rows_csv(target, rows, fieldnames)


def classify_go_no_go(cycle_health: list[CycleHealth]) -> GoNoGoDecision:
    severity_summary = {"info": 0, "warning": 0, "error": 0, "critical": 0}
    reasons: list[str] = []
    decision = "go"
    for cycle in cycle_health:
        for severity, count in cycle.alerts_by_severity.items():
            severity_summary[severity] = severity_summary.get(severity, 0) + count
        if cycle.status == "no_go":
            decision = "no_go"
            reasons.append(f"cycle_{cycle.cycle}_no_go:{','.join(cycle.issues)}")
        elif cycle.status == "caution" and decision != "no_go":
            decision = "caution"
            reasons.append(f"cycle_{cycle.cycle}_caution:{','.join(cycle.issues) or 'warnings'}")
    return GoNoGoDecision(decision, reasons, severity_summary)


def _scenario_report(settings: Settings, run_dir: Path, applied: list[dict[str, Any]]) -> None:
    write_json(run_dir / "chaos_scenarios_applied.json", {"enabled": settings.chaos.enabled, "scenarios": applied})


def _incident_log(run_dir: Path, incidents: list[dict[str, Any]]) -> None:
    path = run_dir / "incident_log.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in incidents:
            handle.write(json.dumps(item, sort_keys=True))
            handle.write("\n")


def _apply_pre_cycle_chaos(settings: Settings, cycle: int, incidents: list[dict[str, Any]]) -> Settings:
    if not settings.chaos.enabled:
        return settings
    scenario_names = set(settings.chaos.scenarios)
    runtime_state_path = build_runtime_state_path(settings)
    if "corrupt_restore_once" in scenario_names and cycle == 2:
        runtime_state_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_state_path.write_text("{corrupt", encoding="utf-8")
        incidents.append({"cycle": cycle, "category": "corrupt_restore_once", "severity": "critical"})
    if "duplicate_event_once" in scenario_names and cycle == 2 and runtime_state_path.exists():
        payload = _read_json(runtime_state_path)
        state = payload.get("state", {})
        processing = state.setdefault("processing_state", {})
        processing["last_processed_timestamp_by_symbol"] = {symbol: "9999-01-01T00:00:00" for symbol in settings.trading.symbols}
        payload["state"] = state
        runtime_state_path.write_text(json.dumps(payload), encoding="utf-8")
        incidents.append({"cycle": cycle, "category": "duplicate_event_once", "severity": "warning"})
    if "market_session_blocked" in scenario_names and cycle == 1:
        settings = replace_settings_session(settings, SessionConfig(enabled=True, allowed_weekdays=(6,), allowed_start_hour_utc=0, allowed_end_hour_utc=0))
        incidents.append({"cycle": cycle, "category": "market_session_blocked", "severity": "warning"})
    return settings


def replace_settings_session(settings: Settings, session: SessionConfig) -> Settings:
    from dataclasses import replace

    return replace(settings, session=session)


def run_soak(
    settings: Settings,
    mode: str,
    require_broker: bool,
    base_client_factory: Callable[[], MT5Client] | None = None,
    allowed_profile_states: set[str] | None = None,
) -> tuple[int, Path]:
    command_name = "paper_soak" if mode == "paper" else "demo_dry_soak"
    run_dir = build_run_directory(settings.data.runs_dir, command_name)
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    cycle_summaries_dir = run_dir / "cycle_summaries"
    cycle_summaries_dir.mkdir(parents=True, exist_ok=True)
    incidents: list[dict[str, Any]] = []
    applied_scenarios: list[dict[str, Any]] = []
    cycle_health: list[CycleHealth] = []
    cycle_dirs: list[Path] = []
    runtime_state_path = build_runtime_state_path(settings)
    if not settings.soak.restore_between_cycles and runtime_state_path.exists():
        runtime_state_path.unlink()
    for cycle in range(1, settings.soak.cycles + 1):
        cycle_settings = _apply_pre_cycle_chaos(settings, cycle, incidents)
        if not cycle_settings.soak.restore_between_cycles and runtime_state_path.exists():
            runtime_state_path.unlink()
        scenario_names = set(cycle_settings.chaos.scenarios) if cycle_settings.chaos.enabled else set()
        if scenario_names:
            applied_scenarios.append({"cycle": cycle, "scenarios": sorted(scenario_names)})
        client_factory = None
        if require_broker or scenario_names:
            client_factory = lambda cycle=cycle: ChaosMT5Client(base_client_factory, scenario_names, cycle, cycle_settings.chaos.every_n_cycles)
        exit_code, cycle_dir = run_resilient_session(
            cycle_settings,
            mode,
            require_broker,
            client_factory=client_factory,
            allowed_profile_states=allowed_profile_states,
        )
        cycle_dirs.append(cycle_dir)
        summary = _cycle_health(cycle, cycle_dir)
        cycle_health.append(summary)
        write_json(cycle_summaries_dir / f"cycle_{cycle:02d}.json", {"run_dir": str(cycle_dir), **summary.to_dict(), "exit_code": exit_code})
        if settings.soak.pause_seconds > 0:
            time.sleep(settings.soak.pause_seconds)
        if summary.status == "no_go":
            incidents.append({"cycle": cycle, "category": "no_go_cycle", "severity": "critical", "issues": summary.issues})
    decision = classify_go_no_go(cycle_health)
    health_report = {"cycles": [item.to_dict() for item in cycle_health]}
    soak_report = {
        "mode": mode,
        "cycles_requested": settings.soak.cycles,
        "cycles_completed": len(cycle_health),
        "restore_between_cycles": settings.soak.restore_between_cycles,
        "cycle_run_dirs": [str(path) for path in cycle_dirs],
    }
    write_json(run_dir / "soak_report.json", soak_report)
    write_json(run_dir / "health_report.json", health_report)
    write_json(run_dir / "go_no_go_report.json", decision.to_dict())
    _scenario_report(settings, run_dir, applied_scenarios)
    _incident_log(run_dir, incidents)
    execution_sources = [path / "execution_journal.csv" for path in cycle_dirs]
    signal_sources = [path / "signal_log.csv" for path in cycle_dirs]
    closed_sources = [path / "closed_trades.csv" for path in cycle_dirs]
    _aggregate_csv(run_dir / "execution_journal.csv", execution_sources)
    _aggregate_csv(run_dir / "signal_log.csv", signal_sources)
    _aggregate_csv(run_dir / "closed_trades.csv", closed_sources)
    if cycle_dirs:
        for filename in (
            "open_positions_snapshot.json",
            "run_report.json",
            "validation_report.json",
            "operational_status.json",
            "reconciliation_report.json",
            "restore_state_report.json",
        ):
            source = cycle_dirs[-1] / filename
            if source.exists():
                shutil.copyfile(source, run_dir / filename)
        alerts = []
        for cycle_dir in cycle_dirs:
            alerts.extend(_read_alerts(cycle_dir / "alerts_log.jsonl"))
        with (run_dir / "alerts_log.jsonl").open("w", encoding="utf-8") as handle:
            for alert in alerts:
                handle.write(json.dumps(alert, sort_keys=True))
                handle.write("\n")
    logger.info("%s soak complete decision=%s cycles=%s", mode, decision.decision, len(cycle_health))
    return (0 if decision.decision == "go" else 2 if decision.decision == "caution" else 3), run_dir


def run_chaos_scenario(settings: Settings, mode: str, require_broker: bool) -> tuple[int, Path]:
    return run_soak(settings, mode, require_broker)


# ---------------------------------------------------------------------------
# BLOQUE 4 — Demo-Guarded Soak
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DemoGuardedSoakSummary:
    cycles_requested: int
    cycles_completed: int
    restore_events: int
    restore_failures: int
    reconcile_events: int
    reconcile_failures: int
    blocked_trade_events: int
    critical_alerts: int
    warning_alerts: int
    circuit_breaker_triggers: int
    no_go_cycles: int
    overall_decision: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_demo_guarded_soak(
    settings: Settings,
    mode: str = "paper",
    require_broker: bool = False,
    base_client_factory: Callable[[], MT5Client] | None = None,
    allowed_profile_states: set[str] | None = None,
) -> tuple[int, Path]:
    """Demo-guarded soak: enforces stricter audit than standard soak.

    In addition to standard soak artifacts it accumulates:
    - restore_events / restore_failures per cycle
    - reconcile_events / reconcile_failures per cycle
    - blocked_trade_events per cycle
    - circuit_breaker_triggers (currently from critical alerts)
    - no_go_cycles count
    - DemoGuardedSoakSummary as a first-class artifact

    Returns: (exit_code, run_dir)
    exit_code: 0=go, 2=caution, 3=no_go
    """
    command_name = "demo_guarded_paper_soak" if mode == "paper" else "demo_guarded_soak"
    run_dir = build_run_directory(settings.data.runs_dir, command_name)
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    cycle_summaries_dir = run_dir / "cycle_summaries"
    cycle_summaries_dir.mkdir(parents=True, exist_ok=True)

    incidents: list[dict[str, Any]] = []
    cycle_health: list[CycleHealth] = []
    cycle_dirs: list[Path] = []

    # Counters for demo-guarded summary
    restore_events = 0
    restore_failures = 0
    reconcile_events = 0
    reconcile_failures = 0
    blocked_trade_events = 0
    critical_alert_total = 0
    warning_alert_total = 0
    circuit_breaker_triggers = 0
    no_go_cycles = 0

    runtime_state_path = build_runtime_state_path(settings)
    if not settings.soak.restore_between_cycles and runtime_state_path.exists():
        runtime_state_path.unlink()

    for cycle in range(1, settings.soak.cycles + 1):
        if settings.soak.pause_seconds > 0:
            time.sleep(settings.soak.pause_seconds)

        client_factory: Callable[[], MT5Client] | None = base_client_factory
        exit_code, cycle_dir = run_resilient_session(
            settings,
            mode,
            require_broker,
            client_factory=client_factory,
            allowed_profile_states=allowed_profile_states,
        )
        cycle_dirs.append(cycle_dir)
        health = _cycle_health(cycle, cycle_dir)
        cycle_health.append(health)

        # Accumulate demo-guarded metrics
        restore_events += 1
        if not health.restore_ok:
            restore_failures += 1

        reconcile_events += 1
        if not health.reconciliation_ok:
            reconcile_failures += 1

        blocked_trade_events += sum(
            1 for issue in health.issues if "blocked" in issue or "rejection" in issue
        )
        critical_alert_total += health.alerts_by_severity.get("critical", 0)
        warning_alert_total += health.alerts_by_severity.get("warning", 0)

        # Circuit breaker: any critical alert is a trigger
        if health.alerts_by_severity.get("critical", 0) > 0:
            circuit_breaker_triggers += 1
            incidents.append({
                "cycle": cycle,
                "category": "circuit_breaker_trigger",
                "severity": "critical",
                "alerts": health.alerts_by_severity,
            })

        if health.status == "no_go":
            no_go_cycles += 1
            incidents.append({
                "cycle": cycle,
                "category": "no_go_cycle",
                "severity": "critical",
                "issues": health.issues,
            })

        write_json(
            cycle_summaries_dir / f"cycle_{cycle:02d}.json",
            {
                "run_dir": str(cycle_dir),
                **health.to_dict(),
                "exit_code": exit_code,
                "restore_ok": health.restore_ok,
                "reconciliation_ok": health.reconciliation_ok,
            },
        )

    decision_obj = classify_go_no_go(cycle_health)
    summary = DemoGuardedSoakSummary(
        cycles_requested=settings.soak.cycles,
        cycles_completed=len(cycle_health),
        restore_events=restore_events,
        restore_failures=restore_failures,
        reconcile_events=reconcile_events,
        reconcile_failures=reconcile_failures,
        blocked_trade_events=blocked_trade_events,
        critical_alerts=critical_alert_total,
        warning_alerts=warning_alert_total,
        circuit_breaker_triggers=circuit_breaker_triggers,
        no_go_cycles=no_go_cycles,
        overall_decision=decision_obj.decision,
    )

    write_json(run_dir / "demo_guarded_soak_summary.json", summary.to_dict())
    write_json(run_dir / "soak_report.json", {
        "mode": mode,
        "soak_type": "demo_guarded",
        "cycles_requested": settings.soak.cycles,
        "cycles_completed": len(cycle_health),
        "cycle_run_dirs": [str(p) for p in cycle_dirs],
    })
    write_json(run_dir / "health_report.json", {"cycles": [h.to_dict() for h in cycle_health]})
    write_json(run_dir / "go_no_go_report.json", decision_obj.to_dict())
    _incident_log(run_dir, incidents)

    # Aggregate CSVs from cycle dirs
    _aggregate_csv(run_dir / "execution_journal.csv", [p / "execution_journal.csv" for p in cycle_dirs])
    _aggregate_csv(run_dir / "signal_log.csv", [p / "signal_log.csv" for p in cycle_dirs])
    _aggregate_csv(run_dir / "closed_trades.csv", [p / "closed_trades.csv" for p in cycle_dirs])

    if cycle_dirs:
        for filename in ("open_positions_snapshot.json", "run_report.json", "validation_report.json", "operational_status.json", "reconciliation_report.json", "restore_state_report.json"):
            src = cycle_dirs[-1] / filename
            if src.exists():
                shutil.copyfile(src, run_dir / filename)
        all_alerts: list[dict[str, Any]] = []
        for cd in cycle_dirs:
            all_alerts.extend(_read_alerts(cd / "alerts_log.jsonl"))
        with (run_dir / "alerts_log.jsonl").open("w", encoding="utf-8") as fh:
            for alert in all_alerts:
                fh.write(json.dumps(alert, sort_keys=True))
                fh.write("\n")

    logger.info(
        "demo_guarded_soak complete mode=%s decision=%s cycles=%s restore_fail=%s reconcile_fail=%s",
        mode, decision_obj.decision, len(cycle_health), restore_failures, reconcile_failures,
    )
    return (0 if decision_obj.decision == "go" else 2 if decision_obj.decision == "caution" else 3), run_dir


def regenerate_go_no_go_report(settings: Settings) -> tuple[int, Path]:
    candidates = sorted(list(settings.data.runs_dir.glob("*_paper_soak")) + list(settings.data.runs_dir.glob("*_demo_dry_soak")))
    if not candidates:
        raise FileNotFoundError("No soak runs available")
    run_dir = candidates[-1]
    health = _read_json(run_dir / "health_report.json")
    cycles = [CycleHealth(**item) for item in health.get("cycles", [])]
    decision = classify_go_no_go(cycles)
    write_json(run_dir / "go_no_go_report.json", decision.to_dict())
    return 0, run_dir
