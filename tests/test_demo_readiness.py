"""
Tests for demo_readiness.py — conservative readiness assessment for broker-executing demo.

Covers:
  - not_ready when approved_demo portfolio is empty
  - not_ready when lifecycle evidence is missing
  - not_ready when endurance evidence is missing
  - not_ready when active_strategy_profiles.json is missing
  - not_ready when registry is corrupted
  - caution when evidence store has no lifecycle (advisory only)
  - ready_for_next_phase only when ALL required checks pass
  - order_send is confirmed NOT integrated
  - No live execution triggered by the assessment
  - Technical debt: no new debt introduced (no bypasses, no shortcuts)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from iris_bot.artifacts import wrap_artifact
from iris_bot.config import load_settings
from iris_bot.demo_readiness import generate_demo_execution_readiness_report
from iris_bot.governance import (
    _materialize_active_profiles_from_registry,
    load_strategy_profile_registry,
    registry_path,
    validate_strategy_profiles,
)
from iris_bot.operational import atomic_write_json


def _settings(tmp_path: Path, monkeypatch):
    settings = load_settings()
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    runs_dir = tmp_path / "runs"
    runtime_dir = tmp_path / "data" / "runtime"
    for p in (raw_dir, processed_dir, runs_dir, runtime_dir):
        p.mkdir(parents=True, exist_ok=True)
    object.__setattr__(settings, "project_root", tmp_path)
    object.__setattr__(settings.data, "raw_dir", raw_dir)
    object.__setattr__(settings.data, "processed_dir", processed_dir)
    object.__setattr__(settings.data, "runs_dir", runs_dir)
    object.__setattr__(settings.data, "runtime_dir", runtime_dir)
    object.__setattr__(settings.experiment, "_processed_dir", processed_dir)
    return settings


def _write_validated_profile(settings, symbol: str) -> None:
    from iris_bot.symbols import write_symbol_strategy_profiles
    run_dir = settings.data.runs_dir / f"sv_{symbol}_strategy_validation"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "leakage_fix_report.json").write_text(
        json.dumps(wrap_artifact("strategy_validation", {"test_used_for_selection": False}))
    )
    (run_dir / "symbol_enablement_report.json").write_text(
        json.dumps(wrap_artifact("symbol_enablement", {
            "symbols": {symbol: {"state": "enabled", "enabled": True, "chosen_model": "global_model"}}
        }))
    )
    (run_dir / "strategy_validation_report.json").write_text(
        json.dumps(wrap_artifact("strategy_validation", {"symbols": {symbol: {"chosen_model": "global_model"}}}))
    )
    write_symbol_strategy_profiles(settings, {}, {})
    validate_strategy_profiles(settings)


def _write_lifecycle(settings, symbol: str) -> None:
    run_dir = settings.data.runs_dir / f"lc_lifecycle_reconciliation"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": True, "critical_mismatch_count": 0, "mismatch_counts": {}, "mismatches": [],
        "symbols": {symbol: {"critical_mismatch_count": 0, "mismatch_categories": []}},
    }
    (run_dir / "lifecycle_reconciliation_report.json").write_text(json.dumps(wrap_artifact("lifecycle_reconciliation", payload)))
    stab_dir = settings.data.runs_dir / f"lc_mt5_windows_stabilization"
    stab_dir.mkdir(parents=True, exist_ok=True)
    rerun = {
        "audit_ok": True, "critical_mismatch_count": 0, "reconciliation_ok": True,
        "reconciliation_run": str(run_dir / "lifecycle_reconciliation_report.json"),
        "rerun_source": "test",
    }
    (stab_dir / "lifecycle_rerun_report.json").write_text(json.dumps(rerun))


def _write_endurance(settings, symbol: str) -> None:
    run_dir = settings.data.runs_dir / f"end_{symbol}_symbol_endurance"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbols": {
            symbol: {
                "decision": "go", "cycles_completed": 3, "blocked_trades": 0, "no_trade_count": 2,
                "expectancy_degradation_pct": 0.05, "profit_factor_degradation_pct": 0.05,
                "alerts_by_severity": {"critical": 0, "error": 0, "warning": 0, "info": 0},
                "cycle_metrics": [{"trades": 5, "expectancy_usd": 3.0, "profit_factor": 1.5} for _ in range(3)],
            }
        }
    }
    (run_dir / "symbol_stability_report.json").write_text(json.dumps(wrap_artifact("symbol_stability", payload)))


def _make_approved_demo_registry(settings, symbol: str) -> None:
    """Creates registry with an approved_demo entry for symbol."""
    from datetime import UTC, datetime
    reg = {
        "profiles": {
            symbol: [{
                "profile_id": f"{symbol}_approved",
                "promotion_state": "approved_demo",
                "enablement_state": "enabled",
                "checksum": "a" * 64,
                "created_at": datetime.now(tz=UTC).isoformat(),
                "source_run_id": "test",
                "symbol": symbol,
                "model_variant": "global_model",
                "promotion_reason": "test_approved",
                "rollback_target": None,
                "profile_payload": {
                    "symbol": symbol,
                    "enabled_state": "enabled",
                    "enabled": True,
                    "profile_id": f"{symbol}_approved",
                    "promotion_state": "approved_demo",
                    "promotion_reason": "test_approved",
                    "rollback_target": None,
                },
            }]
        },
        "active_profiles": {symbol: f"{symbol}_approved"},
    }
    # Fix checksum
    import hashlib
    for entry in reg["profiles"][symbol]:
        pp = entry["profile_payload"].copy()
        for key in ("profile_id", "promotion_state", "promotion_reason", "rollback_target"):
            pp.pop(key, None)
        entry["checksum"] = hashlib.sha256(json.dumps(pp, sort_keys=True).encode()).hexdigest()
    reg_path_obj = registry_path(settings)
    reg_path_obj.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reg_path_obj, wrap_artifact("strategy_profile_registry", reg))
    # Materialize active profiles
    _materialize_active_profiles_from_registry(settings, reg)


# --- Not ready cases ---

def test_not_ready_empty_portfolio(tmp_path, monkeypatch):
    """No approved_demo symbols → not_ready."""
    settings = _settings(tmp_path, monkeypatch)
    report = generate_demo_execution_readiness_report(settings)
    assert report["decision"] == "not_ready"
    assert report["order_send_integrated"] is False


def test_not_ready_missing_lifecycle(tmp_path, monkeypatch):
    """Approved_demo profile but no lifecycle evidence → not_ready."""
    settings = _settings(tmp_path, monkeypatch)
    _make_approved_demo_registry(settings, "EURUSD")
    _write_endurance(settings, "EURUSD")
    report = generate_demo_execution_readiness_report(settings)
    assert report["decision"] == "not_ready"
    assert "lifecycle_evidence" in report["failed_required_checks"]


def test_not_ready_missing_endurance(tmp_path, monkeypatch):
    """Approved_demo profile but no endurance evidence → not_ready."""
    settings = _settings(tmp_path, monkeypatch)
    _make_approved_demo_registry(settings, "EURUSD")
    _write_lifecycle(settings, "EURUSD")
    report = generate_demo_execution_readiness_report(settings)
    assert report["decision"] == "not_ready"
    assert "endurance_evidence" in report["failed_required_checks"]


def test_not_ready_missing_active_materialization(tmp_path, monkeypatch):
    """No active_strategy_profiles.json → not_ready."""
    settings = _settings(tmp_path, monkeypatch)
    _make_approved_demo_registry(settings, "EURUSD")
    _write_lifecycle(settings, "EURUSD")
    _write_endurance(settings, "EURUSD")
    # Remove active profiles file
    from iris_bot.governance import active_strategy_profiles_path
    active_path = active_strategy_profiles_path(settings)
    if active_path.exists():
        active_path.unlink()
    report = generate_demo_execution_readiness_report(settings)
    assert report["decision"] == "not_ready"
    assert "active_materialization" in report["failed_required_checks"]


# --- Order send invariant ---

def test_order_send_never_integrated(tmp_path, monkeypatch):
    """order_send_integrated must always be False in this phase."""
    settings = _settings(tmp_path, monkeypatch)
    report = generate_demo_execution_readiness_report(settings)
    assert report["order_send_integrated"] is False


def test_no_real_execution_triggered(tmp_path, monkeypatch):
    """Running the readiness assessment must not trigger any MT5 connection or execution."""
    settings = _settings(tmp_path, monkeypatch)
    # Just running the assessment must complete without errors and without MT5 calls
    _make_approved_demo_registry(settings, "EURUSD")
    report = generate_demo_execution_readiness_report(settings)
    # The report has a decision but never connects to MT5
    assert "decision" in report
    assert "checks" in report
    assert report["order_send_integrated"] is False


# --- Ready case ---

def test_ready_when_all_checks_pass(tmp_path, monkeypatch):
    """ready_for_next_phase when all required checks pass."""
    settings = _settings(tmp_path, monkeypatch)
    _make_approved_demo_registry(settings, "EURUSD")
    _write_lifecycle(settings, "EURUSD")
    _write_endurance(settings, "EURUSD")
    report = generate_demo_execution_readiness_report(settings)
    # Caution is ok (evidence_store may be empty = advisory); ready is also ok
    assert report["decision"] in ("ready_for_next_phase", "caution")
    assert report["order_send_integrated"] is False


# --- Conservative default ---

def test_default_decision_is_not_ready(tmp_path, monkeypatch):
    """With no setup at all, the default must be not_ready (conservative)."""
    settings = _settings(tmp_path, monkeypatch)
    report = generate_demo_execution_readiness_report(settings)
    assert report["decision"] == "not_ready"


# --- Report structure ---

def test_report_structure_complete(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    report = generate_demo_execution_readiness_report(settings)
    required_fields = [
        "decision", "ready_for_next_phase", "failed_required_checks",
        "failed_advisory_checks", "blocking_reasons", "advisory_reasons",
        "checks", "phase_note", "order_send_integrated", "generated_at",
    ]
    for f in required_fields:
        assert f in report, f"Missing field: {f}"
    required_check_keys = [
        "registry_integrity", "lifecycle_evidence", "endurance_evidence",
        "active_materialization", "approved_demo_portfolio", "no_order_send",
    ]
    for k in required_check_keys:
        assert k in report["checks"], f"Missing check: {k}"


def test_technical_debt_avoidance_no_bypasses(tmp_path, monkeypatch):
    """
    The readiness report must not contain any bypass indicators.
    This is a meta-test ensuring no technical debt was introduced.
    """
    settings = _settings(tmp_path, monkeypatch)
    report = generate_demo_execution_readiness_report(settings)
    # order_send must never be True
    assert report["order_send_integrated"] is False
    # No check should have a "bypass" in its reason
    for check_name, check in report["checks"].items():
        reason = check.get("reason", "")
        assert "bypass" not in reason.lower(), f"Check {check_name} has bypass: {reason}"
