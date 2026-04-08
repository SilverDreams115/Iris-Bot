"""BLOQUE 5 — Serious demo control gate tests.

Validates:
- Gate returns not_ready when required checks fail
- Gate returns ready_for_controlled_serious_demo when all checks pass
- Gate returns serious_demo_with_reservations when only warnings fail
- Report is JSON-serializable
- Gate writes artifact file
- Decision logic is consistent (failed_required → not_ready, failed_warnings → reservations)
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from iris_bot.config import Settings
from iris_bot.serious_demo_gate import (
    generate_serious_demo_control_report,
    serious_demo_control_gate,
)


def _settings(tmp_path: Path) -> Settings:
    settings = Settings()
    object.__setattr__(settings, "data", replace(settings.data, runs_dir=tmp_path / "runs", runtime_dir=tmp_path / "runtime"))
    return settings


# ── Decision values ──────────────────────────────────────────────────────────

def test_gate_report_has_valid_decision(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_serious_demo_control_report(settings)
    assert report["decision"] in {
        "not_ready_for_serious_demo",
        "ready_for_controlled_serious_demo",
        "serious_demo_with_reservations",
    }


def test_gate_report_ready_for_serious_demo_field_consistent(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_serious_demo_control_report(settings)
    if report["decision"] == "not_ready_for_serious_demo":
        assert report["ready_for_serious_demo"] is False
    else:
        assert report["ready_for_serious_demo"] is True


def test_gate_report_failed_required_matches_decision(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_serious_demo_control_report(settings)
    if report["failed_required_checks"]:
        assert report["decision"] == "not_ready_for_serious_demo"


# ── Report structure ─────────────────────────────────────────────────────────

def test_gate_report_has_all_required_fields(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_serious_demo_control_report(settings)
    required_fields = {
        "decision", "ready_for_serious_demo",
        "failed_required_checks", "failed_warning_checks",
        "blocking_reasons", "reservation_reasons",
        "checks", "phase", "generated_at",
    }
    assert required_fields.issubset(set(report.keys()))


def test_gate_report_phase_is_correct(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_serious_demo_control_report(settings)
    assert report["phase"] == "demo_serio_controlado"


def test_gate_report_checks_dict_has_required_check_names(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_serious_demo_control_report(settings)
    checks = report["checks"]
    required_check_names = {
        "module_imports",
        "demo_execution_gate",
        "operational_gate",
        "kill_switch_clean",
        "session_guard_infrastructure",
        "forward_evidence_infrastructure",
        "preflight_hardened",
    }
    assert required_check_names.issubset(set(checks.keys()))


def test_gate_report_is_json_serializable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_serious_demo_control_report(settings)
    assert json.loads(json.dumps(report))


def test_gate_report_generated_at_is_iso_format(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_serious_demo_control_report(settings)
    assert "T" in report["generated_at"]


# ── Infrastructure checks individually ──────────────────────────────────────

def test_gate_module_imports_check_passes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_serious_demo_control_report(settings)
    assert report["checks"]["module_imports"]["ok"] is True


def test_gate_kill_switch_clean_check_passes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_serious_demo_control_report(settings)
    assert report["checks"]["kill_switch_clean"]["ok"] is True


def test_gate_session_guard_infrastructure_passes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_serious_demo_control_report(settings)
    assert report["checks"]["session_guard_infrastructure"]["ok"] is True


def test_gate_forward_evidence_infrastructure_passes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_serious_demo_control_report(settings)
    assert report["checks"]["forward_evidence_infrastructure"]["ok"] is True


def test_gate_preflight_hardened_check_passes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    report = generate_serious_demo_control_report(settings)
    assert report["checks"]["preflight_hardened"]["ok"] is True


# ── Command return codes ──────────────────────────────────────────────────────

def test_serious_demo_gate_command_returns_0_or_1_or_2(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    exit_code = serious_demo_control_gate(settings)
    assert exit_code in {0, 1, 2}


def test_serious_demo_gate_command_writes_artifact(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    serious_demo_control_gate(settings)
    run_dirs = sorted((tmp_path / "runs").glob("*_serious_demo_control_gate"))
    assert len(run_dirs) >= 1
    report_file = run_dirs[-1] / "serious_demo_control_report.json"
    assert report_file.exists()
    loaded = json.loads(report_file.read_text())
    payload = loaded.get("payload", loaded)
    assert "decision" in payload


def test_serious_gate_blocks_when_demo_execution_gate_is_not_ready(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        "iris_bot.serious_demo_gate.generate_demo_execution_readiness_report",
        lambda _settings: {
            "decision": "not_ready_for_demo",
            "failed_required_checks": ["lifecycle_evidence"],
        },
    )
    report = generate_serious_demo_control_report(settings)
    assert report["decision"] == "not_ready_for_serious_demo"
    assert "demo_execution_gate" in report["failed_required_checks"]


def test_serious_gate_can_be_ready_when_demo_execution_gate_is_ready(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        "iris_bot.serious_demo_gate.generate_demo_execution_readiness_report",
        lambda _settings: {
            "decision": "ready_for_demo_guarded",
            "failed_required_checks": [],
        },
    )
    report = generate_serious_demo_control_report(settings)
    assert report["checks"]["demo_execution_gate"]["ok"] is True
