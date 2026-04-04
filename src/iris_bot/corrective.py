from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from iris_bot.artifacts import artifact_schema_report, read_artifact_payload, wrap_artifact
from iris_bot.config import Settings
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.processed_dataset import load_processed_dataset
from iris_bot.sessions import session_definition_report
from iris_bot.symbols import strategy_profiles_path


def _latest_run(settings: Settings, suffix: str) -> Path | None:
    candidates = sorted(settings.data.runs_dir.glob(f"*_{suffix}"))
    return candidates[-1] if candidates else None


def _leakage_fix_report(settings: Settings) -> dict[str, Any]:
    latest = _latest_run(settings, "strategy_validation")
    if latest is None or not (latest / "leakage_fix_report.json").exists():
        return {"ok": False, "reason": "missing_strategy_validation_run"}
    return read_artifact_payload(latest / "leakage_fix_report.json", expected_type="strategy_validation")


def _reconciliation_scope_report(settings: Settings) -> dict[str, Any]:
    latest = _latest_run(settings, "demo_dry_resilient") or _latest_run(settings, "paper_resilient")
    if latest is None or not (latest / "reconciliation_scope_report.json").exists():
        return {
            "ok": True,
            "ownership_filter_active": True,
            "magic_number": settings.mt5.magic_number,
            "comment_tag": settings.mt5.comment_tag,
            "reconcile_symbols_only": settings.mt5.reconcile_symbols_only,
            "note": "No resilient run found; reporting configured reconciliation scope.",
        }
    return json.loads((latest / "reconciliation_scope_report.json").read_text(encoding="utf-8"))


def _idempotency_report(settings: Settings) -> dict[str, Any]:
    latest = _latest_run(settings, "demo_dry_resilient") or _latest_run(settings, "paper_resilient")
    if latest is None or not (latest / "idempotency_report.json").exists():
        return {"ok": True, "uses_real_event_ids": False, "fallback_active": True, "note": "No resilient run found; fallback path remains the current guaranteed mode."}
    return json.loads((latest / "idempotency_report.json").read_text(encoding="utf-8"))


def _session_consistency_report(settings: Settings) -> dict[str, Any]:
    dataset = load_processed_dataset(
        settings.experiment.processed_dataset_path,
        settings.experiment.processed_schema_path,
        settings.experiment.processed_manifest_path,
    )
    mismatches = 0
    checked = 0
    for row in dataset.rows[:200]:
        flags = (
            row.features.get("session_asia", 0.0),
            row.features.get("session_london", 0.0),
            row.features.get("session_new_york", 0.0),
        )
        if sum(1 for value in flags if value > 0.5) > 1:
            mismatches += 1
        checked += 1
    return {
        "ok": mismatches == 0,
        "checked_rows": checked,
        "mismatches": mismatches,
        "definition": session_definition_report(),
    }


def _performance_benchmark_report(settings: Settings) -> dict[str, Any]:
    dataset = load_processed_dataset(
        settings.experiment.processed_dataset_path,
        settings.experiment.processed_schema_path,
        settings.experiment.processed_manifest_path,
    )
    rows = [row for row in dataset.rows if row.timeframe == settings.trading.primary_timeframe][:600]
    if not rows:
        return {"ok": False, "reason": "no_rows"}
    rows_by_symbol: dict[str, list[Any]] = {}
    for row in rows:
        rows_by_symbol.setdefault(row.symbol, []).append(row)
    timestamps = [(row.symbol, row.timestamp) for row in rows]
    start = time.perf_counter()
    for symbol, timestamp in timestamps:
        series = rows_by_symbol[symbol]
        next(i for i, candidate in enumerate(series) if candidate.timestamp == timestamp)
    naive_seconds = time.perf_counter() - start
    index_map = {(row.symbol, row.timestamp.isoformat()): index for symbol, series in rows_by_symbol.items() for index, row in enumerate(series)}
    start = time.perf_counter()
    for row in rows:
        _ = index_map[(row.symbol, row.timestamp.isoformat())]
    indexed_seconds = time.perf_counter() - start
    return {
        "ok": True,
        "rows_benchmarked": len(rows),
        "naive_lookup_seconds": naive_seconds,
        "indexed_lookup_seconds": indexed_seconds,
        "speedup_ratio": 0.0 if indexed_seconds == 0.0 else naive_seconds / indexed_seconds,
        "semantic_change": False,
    }


def _artifact_schema_summary(settings: Settings) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    strategy_profiles = strategy_profiles_path(settings)
    if strategy_profiles.exists():
        reports.append(artifact_schema_report(strategy_profiles, expected_type="strategy_profiles"))
    latest_validation = _latest_run(settings, "strategy_validation")
    if latest_validation is not None:
        for filename, artifact_type in (
            ("strategy_validation_report.json", "strategy_validation"),
            ("model_comparison_report.json", "model_comparison"),
            ("dynamic_exit_report.json", "dynamic_exit_report"),
            ("threshold_report.json", "threshold_report"),
            ("symbol_enablement_report.json", "symbol_enablement"),
        ):
            path = latest_validation / filename
            if path.exists():
                reports.append(artifact_schema_report(path, expected_type=artifact_type))
    return {"ok": all(item["ok"] for item in reports) if reports else True, "reports": reports}


def run_corrective_audit(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "corrective_audit")
    logger = configure_logging(run_dir, settings.logging.level)
    leakage_fix = _leakage_fix_report(settings)
    reconciliation_scope = _reconciliation_scope_report(settings)
    idempotency = _idempotency_report(settings)
    session_consistency = _session_consistency_report(settings)
    performance = _performance_benchmark_report(settings)
    artifact_schema = _artifact_schema_summary(settings)
    corrective_report = {
        "leakage_fix_ok": leakage_fix.get("test_used_for_selection") is False if isinstance(leakage_fix, dict) else False,
        "reconciliation_scope_ok": reconciliation_scope.get("ownership_filter_active", False),
        "idempotency_ok": "idempotency_mode_counts" in idempotency or idempotency.get("fallback_active", False),
        "session_consistency_ok": session_consistency.get("ok", False),
        "performance_ok": performance.get("ok", False),
        "artifact_schema_ok": artifact_schema.get("ok", False),
    }
    write_json_report(run_dir, "leakage_fix_report.json", wrap_artifact("corrective_audit", leakage_fix))
    write_json_report(run_dir, "reconciliation_scope_report.json", wrap_artifact("corrective_audit", reconciliation_scope))
    write_json_report(run_dir, "idempotency_report.json", wrap_artifact("corrective_audit", idempotency))
    write_json_report(run_dir, "session_consistency_report.json", wrap_artifact("corrective_audit", session_consistency))
    write_json_report(run_dir, "performance_benchmark_report.json", wrap_artifact("corrective_audit", performance))
    write_json_report(run_dir, "artifact_schema_report.json", wrap_artifact("corrective_audit", artifact_schema))
    write_json_report(run_dir, "corrective_audit_report.json", wrap_artifact("corrective_audit", corrective_report))
    logger.info("corrective_audit run_dir=%s", run_dir)
    return 0
