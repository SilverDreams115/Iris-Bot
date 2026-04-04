from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from iris_bot.artifacts import read_artifact_payload, wrap_artifact
from iris_bot.config import Settings
from iris_bot.governance import load_strategy_profile_registry
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.soak import run_soak
from iris_bot.symbols import load_symbol_strategy_profiles


REVIEW_ALLOWED_PROFILE_STATES = {"enabled", "caution", "validated", "blocked", "approved_demo", "disabled"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _cycle_summary_files(soak_dir: Path) -> list[Path]:
    cycle_dir = soak_dir / "cycle_summaries"
    if not cycle_dir.exists():
        return []
    return sorted(cycle_dir.glob("cycle_*.json"))


def _endurance_consistency(soak_dir: Path) -> dict[str, Any]:
    health = _read_json(soak_dir / "health_report.json")
    soak_report = _read_json(soak_dir / "soak_report.json")
    cycle_files = _cycle_summary_files(soak_dir)
    health_cycles = health.get("cycles", [])
    reported_completed = int(soak_report.get("cycles_completed", 0))
    cycle_file_count = len(cycle_files)
    health_cycle_count = len(health_cycles)
    cycle_numbers: list[int] = []
    for path in cycle_files:
        payload = _read_json(path)
        cycle_numbers.append(int(payload.get("cycle", 0)))
    duplicates = len(cycle_numbers) - len(set(cycle_numbers))
    expected_sequence = list(range(1, cycle_file_count + 1))
    mismatches: list[str] = []
    if reported_completed != health_cycle_count:
        mismatches.append("soak_vs_health_cycle_count")
    if reported_completed != cycle_file_count:
        mismatches.append("soak_vs_cycle_summaries_count")
    if health_cycle_count != cycle_file_count:
        mismatches.append("health_vs_cycle_summaries_count")
    if cycle_numbers and sorted(cycle_numbers) != expected_sequence:
        mismatches.append("cycle_sequence_gap_or_reorder")
    if duplicates > 0:
        mismatches.append("duplicate_cycle_summary")
    return {
        "ok": not mismatches,
        "source_of_truth": "health_report.cycles",
        "reported_completed_cycles": reported_completed,
        "health_cycle_count": health_cycle_count,
        "cycle_summary_count": cycle_file_count,
        "cycle_numbers": cycle_numbers,
        "duplicate_cycle_count": duplicates,
        "mismatches": mismatches,
    }


def _replace_symbols(settings: Settings, symbols: tuple[str, ...]) -> Settings:
    return replace(settings, trading=replace(settings.trading, symbols=symbols))


def _replace_runtime_dir(settings: Settings, symbol: str) -> Settings:
    runtime_dir = settings.data.runtime_dir / "endurance_reviews" / symbol
    return replace(settings, data=replace(settings.data, runtime_dir=runtime_dir))


def _select_endurance_symbols(settings: Settings, only_enabled: bool) -> tuple[str, ...]:
    if settings.endurance.target_symbol:
        return (settings.endurance.target_symbol,)
    if settings.endurance.symbols:
        return settings.endurance.symbols
    profiles = load_symbol_strategy_profiles(settings)
    if not only_enabled:
        return tuple(profile.symbol for profile in profiles.values())
    return tuple(symbol for symbol, profile in profiles.items() if profile.enabled_state == "enabled")


def _trade_metrics(closed_rows: list[dict[str, Any]]) -> dict[str, float]:
    pnls = [float(item.get("net_pnl_usd", 0.0) or 0.0) for item in closed_rows]
    gross_profit = sum(item for item in pnls if item > 0.0)
    gross_loss = abs(sum(item for item in pnls if item < 0.0))
    trades = len(pnls)
    expectancy = sum(pnls) / trades if trades else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (999.0 if gross_profit > 0.0 else 0.0)
    return {
        "trades": trades,
        "expectancy_usd": expectancy,
        "profit_factor": profit_factor,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "net_pnl_usd": sum(pnls),
    }


def _threshold_metrics(signal_rows: list[dict[str, Any]], closed_rows: list[dict[str, Any]]) -> dict[str, float]:
    thresholds = [float(item.get("threshold", 0.0) or 0.0) for item in signal_rows]
    generated = sum(1 for item in signal_rows if item.get("status") == "generated")
    trades = len(closed_rows)
    return {
        "threshold_min": min(thresholds) if thresholds else 0.0,
        "threshold_max": max(thresholds) if thresholds else 0.0,
        "threshold_spread": (max(thresholds) - min(thresholds)) if thresholds else 0.0,
        "trade_conversion_ratio": trades / generated if generated else 0.0,
    }


def _session_breakdown(closed_rows: list[dict[str, Any]]) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for row in closed_rows:
        details = row.get("target_policy_details", "")
        session_name = "unknown"
        if isinstance(details, str) and details:
            try:
                payload = json.loads(details)
                session_name = str(payload.get("session", "unknown"))
            except json.JSONDecodeError:
                session_name = "unknown"
        breakdown[session_name] = breakdown.get(session_name, 0) + 1
    return breakdown


def _degradation(series: list[float]) -> float:
    if len(series) < 2:
        return 0.0
    base = series[0]
    if abs(base) < 1e-12:
        return 0.0 if abs(series[-1]) < 1e-12 else 1.0
    return (base - series[-1]) / abs(base)


def _symbol_stability(symbol: str, cycle_dirs: list[Path], settings: Settings) -> dict[str, Any]:
    cycle_metrics: list[dict[str, Any]] = []
    cycle_health: list[dict[str, Any]] = []
    total_blocked = 0
    total_no_trade = 0
    severity_summary = {"info": 0, "warning": 0, "error": 0, "critical": 0}
    consistency_reports: list[dict[str, Any]] = []
    total_cycles_completed = 0
    for cycle_dir in cycle_dirs:
        health = _read_json(cycle_dir / "health_report.json")
        soak_report = _read_json(cycle_dir / "soak_report.json")
        decision = _read_json(cycle_dir / "go_no_go_report.json")
        consistency = _endurance_consistency(cycle_dir)
        consistency_reports.append({"run_dir": str(cycle_dir), **consistency})
        closed_rows = _read_csv(cycle_dir / "closed_trades.csv")
        signal_rows = [row for row in _read_csv(cycle_dir / "signal_log.csv") if row.get("symbol") == symbol]
        execution_rows = [row for row in _read_csv(cycle_dir / "execution_journal.csv") if row.get("symbol") == symbol]
        symbol_closed = [row for row in closed_rows if row.get("symbol") == symbol]
        trade_metrics = _trade_metrics(symbol_closed)
        threshold_metrics = _threshold_metrics(signal_rows, symbol_closed)
        blocked_count = sum(1 for row in signal_rows if row.get("status") == "blocked")
        total_blocked += blocked_count
        total_no_trade += sum(1 for row in signal_rows if str(row.get("signal", "0")) == "0")
        for item in health.get("cycles", []):
            cycle_health.append(item)
            for severity, count in item.get("alerts_by_severity", {}).items():
                severity_summary[severity] = severity_summary.get(severity, 0) + count
        total_cycles_completed += consistency["health_cycle_count"]
        cycle_metrics.append(
            {
                "run_dir": str(cycle_dir),
                "decision": decision.get("decision", "unknown"),
                "cycles_completed": consistency["health_cycle_count"],
                "blocked_trades": blocked_count,
                "execution_events": len(execution_rows),
                "session_breakdown": _session_breakdown(symbol_closed),
                **trade_metrics,
                **threshold_metrics,
            }
        )
    expectancy_series = [item["expectancy_usd"] for item in cycle_metrics]
    profit_factor_series = [item["profit_factor"] for item in cycle_metrics]
    expectancy_degradation = _degradation(expectancy_series)
    profit_factor_degradation = _degradation(profit_factor_series)
    critical_alerts = severity_summary.get("critical", 0)
    if not cycle_metrics:
        decision = "blocked"
        reasons = ["no_cycles_executed"]
    else:
        reasons = []
        decision = "go"
        if any(not item["ok"] for item in consistency_reports):
            decision = "blocked"
            reasons.append("endurance_reporting_inconsistent")
        if critical_alerts > 0:
            decision = "no_go"
            reasons.append("critical_alerts")
        if any(item.get("decision") == "no_go" for item in cycle_metrics):
            decision = "no_go"
            reasons.append("cycle_no_go")
        if expectancy_degradation > settings.endurance.max_expectancy_degradation_pct:
            decision = "caution" if decision == "go" else decision
            reasons.append("expectancy_degradation")
        if profit_factor_degradation > settings.endurance.max_profit_factor_degradation_pct:
            decision = "caution" if decision == "go" else decision
            reasons.append("profit_factor_degradation")
        if total_cycles_completed < settings.endurance.min_cycles_for_stability:
            decision = "caution" if decision == "go" else decision
            reasons.append("insufficient_cycles")
        if sum(item["trades"] for item in cycle_metrics) == 0:
            decision = "blocked" if decision == "go" else decision
            reasons.append("no_trades_executed")
    return {
        "symbol": symbol,
        "decision": decision,
        "reasons": reasons,
        "cycles_completed": total_cycles_completed,
        "source_of_truth": "health_report.cycles",
        "blocked_trades": total_blocked,
        "no_trade_count": total_no_trade,
        "expectancy_degradation_pct": expectancy_degradation,
        "profit_factor_degradation_pct": profit_factor_degradation,
        "alerts_by_severity": severity_summary,
        "cycle_metrics": cycle_metrics,
        "cycle_health": cycle_health,
        "consistency_reports": consistency_reports,
    }


def _apply_endurance_review_guards(
    stability: dict[str, Any],
    settings: Settings,
    profile: Any,
) -> dict[str, Any]:
    guarded = dict(stability)
    reasons = list(guarded.get("reasons", []))
    cycle_metrics = guarded.get("cycle_metrics", [])
    primary_metrics = cycle_metrics[0] if cycle_metrics else {}
    if profile is None:
        guarded["decision"] = "blocked"
        reasons.append("missing_symbol_profile")
    elif getattr(profile, "enabled_state", "") in {"disabled", "blocked"}:
        guarded["decision"] = "blocked"
        reasons.append("profile_not_tradable")
    if primary_metrics:
        expectancy = float(primary_metrics.get("expectancy_usd", 0.0) or 0.0)
        profit_factor = float(primary_metrics.get("profit_factor", 0.0) or 0.0)
        if expectancy <= settings.strategy.min_expectancy_usd:
            guarded["decision"] = "blocked"
            reasons.append("non_positive_expectancy")
        if profit_factor < settings.strategy.min_profit_factor:
            guarded["decision"] = "blocked"
            reasons.append("profit_factor_below_floor")
    guarded["reasons"] = sorted(set(reasons))
    return guarded


def run_symbol_endurance(settings: Settings, only_enabled: bool) -> tuple[int, Path]:
    command_name = "enabled_symbols_soak" if only_enabled else "symbol_endurance"
    run_dir = build_run_directory(settings.data.runs_dir, command_name)
    logger = configure_logging(run_dir, settings.logging.level)
    symbols = _select_endurance_symbols(settings, only_enabled)
    if not symbols:
        logger.error("No hay simbolos elegibles para endurance")
        return 1, run_dir
    registry = load_strategy_profile_registry(settings)
    symbol_cycle_dirs: dict[str, list[Path]] = {}
    for symbol in symbols:
        profile = load_symbol_strategy_profiles(settings).get(symbol)
        if only_enabled and (profile is None or profile.enabled_state != "enabled"):
            continue
        scoped_settings = _replace_runtime_dir(_replace_symbols(settings, (symbol,)), symbol)
        if only_enabled:
            exit_code, soak_dir = run_soak(
                scoped_settings,
                mode=settings.endurance.mode,
                require_broker=settings.endurance.mode == "demo_dry",
            )
        else:
            try:
                exit_code, soak_dir = run_soak(
                    scoped_settings,
                    mode=settings.endurance.mode,
                    require_broker=settings.endurance.mode == "demo_dry",
                    allowed_profile_states=REVIEW_ALLOWED_PROFILE_STATES,
                )
            except TypeError:
                exit_code, soak_dir = run_soak(
                    scoped_settings,
                    mode=settings.endurance.mode,
                    require_broker=settings.endurance.mode == "demo_dry",
                )
        if exit_code not in {0, 2, 3}:
            logger.error("Fallo soak para %s", symbol)
            return exit_code, run_dir
        symbol_cycle_dirs.setdefault(symbol, []).append(soak_dir)
    symbols_report: dict[str, Any] = {}
    for symbol, cycle_dirs in sorted(symbol_cycle_dirs.items()):
        stability = _symbol_stability(symbol, cycle_dirs, settings)
        symbols_report[symbol] = _apply_endurance_review_guards(
            stability,
            settings,
            load_symbol_strategy_profiles(settings).get(symbol),
        )
    aggregate = {
        "mode": settings.endurance.mode,
        "symbols": symbols_report,
        "active_registry_profiles": registry.get("active_profiles", {}),
        "enabled_symbol_count": sum(1 for item in symbols_report.values() if item["decision"] == "go"),
    }
    artifact_type = "enabled_symbols_soak" if only_enabled else "symbol_endurance"
    write_json_report(run_dir, "symbol_endurance_report.json", wrap_artifact("symbol_endurance", aggregate))
    write_json_report(run_dir, "symbol_stability_report.json", wrap_artifact("symbol_stability", aggregate))
    write_json_report(
        run_dir,
        "endurance_consistency_report.json",
        wrap_artifact(
            "symbol_endurance",
            {
                "source_of_truth": "health_report.cycles",
                "symbols": {
                    symbol: {
                        "cycles_completed": payload["cycles_completed"],
                        "consistency_reports": payload["consistency_reports"],
                    }
                    for symbol, payload in symbols_report.items()
                },
            },
        ),
    )
    write_json_report(
        run_dir,
        "endurance_reporting_fix_report.json",
        wrap_artifact(
            "symbol_endurance",
            {
                "root_cause": "symbol_stability_report counted soak runs instead of health_report cycles",
                "source_of_truth": "health_report.cycles",
                "symbols": {
                    symbol: {
                        "cycles_completed": payload["cycles_completed"],
                        "decision": payload["decision"],
                        "reasons": payload["reasons"],
                    }
                    for symbol, payload in symbols_report.items()
                },
            },
        ),
    )
    if only_enabled:
        write_json_report(run_dir, "enabled_symbols_soak_report.json", wrap_artifact("enabled_symbols_soak", aggregate))
    logger.info("symbol_endurance symbols=%s run_dir=%s", len(symbols_report), run_dir)
    exit_code = 0 if all(item["decision"] == "go" for item in symbols_report.values()) else 2
    return exit_code, run_dir


def symbol_stability_report(settings: Settings) -> tuple[int, Path]:
    candidates = sorted(list(settings.data.runs_dir.glob("*_symbol_endurance")) + list(settings.data.runs_dir.glob("*_enabled_symbols_soak")))
    if not candidates:
        raise FileNotFoundError("No hay corridas de endurance disponibles")
    run_dir = candidates[-1]
    payload = read_artifact_payload(run_dir / "symbol_stability_report.json", expected_type="symbol_stability")
    out_dir = build_run_directory(settings.data.runs_dir, "symbol_stability")
    logger = configure_logging(out_dir, settings.logging.level)
    write_json_report(out_dir, "symbol_stability_report.json", wrap_artifact("symbol_stability", payload))
    logger.info("symbol_stability source=%s run_dir=%s", run_dir, out_dir)
    return 0, out_dir


def audit_endurance_reporting(settings: Settings) -> tuple[int, Path]:
    candidates = sorted(list(settings.data.runs_dir.glob("*_symbol_endurance")) + list(settings.data.runs_dir.glob("*_enabled_symbols_soak")))
    if not candidates:
        raise FileNotFoundError("No hay corridas de endurance disponibles")
    source = candidates[-1]
    payload = read_artifact_payload(source / "endurance_consistency_report.json", expected_type="symbol_endurance")
    out_dir = build_run_directory(settings.data.runs_dir, "audit_endurance_reporting")
    logger = configure_logging(out_dir, settings.logging.level)
    write_json_report(out_dir, "endurance_consistency_report.json", wrap_artifact("symbol_endurance", payload))
    ok = all(
        all(report.get("ok", False) for report in symbol_payload.get("consistency_reports", []))
        for symbol_payload in payload.get("symbols", {}).values()
    )
    write_json_report(out_dir, "endurance_reporting_fix_report.json", wrap_artifact("symbol_endurance", {"source_run": str(source), "ok": ok}))
    logger.info("audit_endurance_reporting source=%s run_dir=%s ok=%s", source, out_dir, ok)
    return (0 if ok else 2), out_dir
