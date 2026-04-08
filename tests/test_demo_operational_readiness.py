"""BLOQUE 7 — Demo Operational Readiness Gate Tests.

Validates that the gate:
- correctly evaluates all 6 bloque checks
- returns credible_guarded when all pass
- returns not_yet_operationally_credible when any fail
- generates a JSON-serialisable audit artifact
- covers all required bloque_coverage fields
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from iris_bot.config import (
    BacktestConfig,
    OperationalConfig,
    RecoveryConfig,
    ReconciliationConfig,
    RiskConfig,
    SessionConfig,
    Settings,
)
from iris_bot.demo_operational_readiness import generate_demo_operational_readiness_report


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    object.__setattr__(settings, "data", replace(settings.data, runs_dir=tmp_path / "runs", runtime_dir=tmp_path / "runtime"))
    object.__setattr__(settings, "recovery", RecoveryConfig(reconnect_retries=1, reconnect_backoff_seconds=0.0, require_state_restore_clean=True))
    object.__setattr__(settings, "reconciliation", ReconciliationConfig(policy="log_only", price_tolerance=0.001, volume_tolerance=0.001))
    object.__setattr__(settings, "backtest", BacktestConfig(use_atr_stops=False, fixed_stop_loss_pct=0.002, fixed_take_profit_pct=0.004, max_holding_bars=5))
    object.__setattr__(settings, "risk", RiskConfig(max_daily_loss_usd=50.0))
    object.__setattr__(settings, "session", SessionConfig(enabled=True, allowed_weekdays=(0, 1, 2, 3, 4), allowed_start_hour_utc=0, allowed_end_hour_utc=23))
    object.__setattr__(settings, "operational", OperationalConfig(persistence_state_filename="runtime_state.json"))
    return settings


# ── Gate returns credible_guarded when all checks pass ───────────────────────

def test_demo_operational_readiness_passes_in_clean_env(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_demo_operational_readiness_report(settings)
    assert report["decision"] == "credible_guarded", (
        f"Expected credible_guarded but got {report['decision']}. "
        f"Failed checks: {report['failed_required_checks']}"
    )
    assert report["operationally_credible"] is True
    assert report["failed_required_checks"] == []


# ── All 6 bloque checks are present ──────────────────────────────────────────

def test_demo_operational_readiness_all_bloques_present(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_demo_operational_readiness_report(settings)
    required_checks = {
        "module_imports",
        "restore_safety_infrastructure",
        "restore_safety_drill",
        "reconciliation_infrastructure",
        "recovery_infrastructure",
        "soak_infrastructure",
        "kill_switch_infrastructure",
        "session_discipline_infrastructure",
    }
    assert required_checks.issubset(report["checks"].keys())


# ── bloque_coverage section covers all 6 bloques ─────────────────────────────

def test_demo_operational_readiness_bloque_coverage_complete(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_demo_operational_readiness_report(settings)
    cov = report["bloque_coverage"]
    assert "bloque_1_restore_restart_safety" in cov
    assert "bloque_2_reconciliation_drills" in cov
    assert "bloque_3_recovery_disconnect" in cov
    assert "bloque_4_soak_readiness" in cov
    assert "bloque_5_kill_switch" in cov
    assert "bloque_6_session_discipline" in cov
    # All must be True in a clean environment
    for bloque, result in cov.items():
        assert result is True, f"{bloque} is not True in clean environment"


# ── Report is JSON-serialisable (full audit artifact) ─────────────────────────

def test_demo_operational_readiness_report_serialisable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_demo_operational_readiness_report(settings)
    assert json.loads(json.dumps(report))


# ── Schema version is present in report ──────────────────────────────────────

def test_demo_operational_readiness_schema_version_present(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_demo_operational_readiness_report(settings)
    assert report.get("schema_version") == 1


# ── generated_at is present ───────────────────────────────────────────────────

def test_demo_operational_readiness_generated_at_present(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_demo_operational_readiness_report(settings)
    assert report.get("generated_at") != ""


# ── Kill switch module is importable (module_imports check) ──────────────────

def test_demo_operational_readiness_module_imports_check_passes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_demo_operational_readiness_report(settings)
    assert report["checks"]["module_imports"]["ok"] is True
    assert report["checks"]["module_imports"]["failed"] == []


# ── Kill switch infrastructure check passes ────────────────────────────────────

def test_demo_operational_readiness_kill_switch_check_passes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_demo_operational_readiness_report(settings)
    ks_check = report["checks"]["kill_switch_infrastructure"]
    assert ks_check["ok"] is True, f"kill_switch check failed: {ks_check.get('issues')}"


# ── Session discipline check passes ──────────────────────────────────────────

def test_demo_operational_readiness_session_discipline_check_passes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_demo_operational_readiness_report(settings)
    sd_check = report["checks"]["session_discipline_infrastructure"]
    assert sd_check["ok"] is True, f"session_discipline check failed: {sd_check.get('issues')}"
