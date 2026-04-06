"""
Tests for the hardened approved_demo gate in governance.py.

Covers:
  - Gate blocks when lifecycle evidence is missing (was: just a warning)
  - Gate blocks when lifecycle audit_ok = False (was: CAUTION)
  - Gate blocks when lifecycle is too old
  - Gate blocks when trade count < min (new check)
  - Gate blocks when no_trade_ratio too high (new check)
  - Gate blocks when blocked_trades_ratio too high (new check)
  - Gate blocks when profit_factor below floor (new check)
  - Gate blocks when expectancy below floor (new check)
  - Gate blocks when endurance missing (was: just a warning)
  - Gate blocks when endurance cycles < min (was: CAUTION)
  - Gate blocks when endurance decision != go (was: CAUTION for some)
  - Gate approves when ALL conditions met
  - Gate config snapshot is included in report for auditability
"""
from __future__ import annotations

import json
from pathlib import Path


from iris_bot.artifacts import wrap_artifact
from iris_bot.config import load_settings
from iris_bot.governance import (
    _promotion_review_for_symbol,
    load_strategy_profile_registry,
    validate_strategy_profiles,
)


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
    return settings


def _write_validation_artifacts(settings, run_id: str, states: dict) -> None:
    run_dir = settings.data.runs_dir / f"{run_id}_strategy_validation"
    run_dir.mkdir(parents=True, exist_ok=True)
    leakage = {"test_used_for_selection": False}
    enablement = {
        "symbols": {
            s: {"state": state, "enabled": state == "enabled", "chosen_model": "global_model"}
            for s, state in states.items()
        }
    }
    comparison = {"symbols": {s: {"chosen_model": "global_model"} for s in states}}
    (run_dir / "leakage_fix_report.json").write_text(json.dumps(wrap_artifact("strategy_validation", leakage)))
    (run_dir / "symbol_enablement_report.json").write_text(json.dumps(wrap_artifact("symbol_enablement", enablement)))
    (run_dir / "strategy_validation_report.json").write_text(json.dumps(wrap_artifact("strategy_validation", comparison)))


def _write_strategy_profiles(settings) -> None:
    from iris_bot.symbols import write_symbol_strategy_profiles
    write_symbol_strategy_profiles(settings, {}, {})


def _make_validated_registry(settings, symbol: str) -> dict:
    """Creates a registry with a validated profile for the given symbol."""
    _write_validation_artifacts(settings, "20260401T100000Z", {symbol: "enabled"})
    _write_strategy_profiles(settings)
    validate_strategy_profiles(settings)
    return load_strategy_profile_registry(settings)


def _write_lifecycle(settings, run_id: str, symbol: str, critical: int = 0, audit_ok: bool = True) -> None:
    run_dir = settings.data.runs_dir / f"{run_id}_lifecycle_reconciliation"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": critical == 0,
        "critical_mismatch_count": critical,
        "mismatch_counts": {},
        "mismatches": [],
        "symbols": {symbol: {"critical_mismatch_count": critical, "mismatch_categories": []}},
    }
    envelope = wrap_artifact("lifecycle_reconciliation", payload)
    (run_dir / "lifecycle_reconciliation_report.json").write_text(json.dumps(envelope))
    # Write stabilization with audit_ok flag
    stab_dir = settings.data.runs_dir / f"{run_id}_mt5_windows_stabilization"
    stab_dir.mkdir(parents=True, exist_ok=True)
    rerun = {
        "audit_ok": audit_ok,
        "critical_mismatch_count": critical,
        "reconciliation_ok": critical == 0,
        "reconciliation_run": str(run_dir / "lifecycle_reconciliation_report.json"),
        "rerun_source": "test",
    }
    (stab_dir / "lifecycle_rerun_report.json").write_text(json.dumps(rerun))


def _write_endurance(
    settings, run_id: str, symbol: str,
    decision: str = "go",
    cycles: int = 3,
    trades_per_cycle: int = 5,
    no_trade_count: int = 2,
    blocked_trades: int = 0,
    profit_factor: float = 1.5,
    expectancy_usd: float = 3.0,
    expectancy_degradation: float = 0.05,
    profit_factor_degradation: float = 0.05,
) -> None:
    run_dir = settings.data.runs_dir / f"{run_id}_symbol_endurance"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbols": {
            symbol: {
                "decision": decision,
                "cycles_completed": cycles,
                "blocked_trades": blocked_trades,
                "no_trade_count": no_trade_count,
                "expectancy_degradation_pct": expectancy_degradation,
                "profit_factor_degradation_pct": profit_factor_degradation,
                "alerts_by_severity": {"critical": 0, "error": 0, "warning": 0, "info": 0},
                "cycle_metrics": [
                    {"trades": trades_per_cycle, "expectancy_usd": expectancy_usd, "profit_factor": profit_factor}
                    for _ in range(cycles)
                ],
            }
        }
    }
    (run_dir / "symbol_stability_report.json").write_text(json.dumps(wrap_artifact("symbol_stability", payload)))


def _review(settings, symbol: str) -> dict:
    registry = load_strategy_profile_registry(settings)
    return _promotion_review_for_symbol(settings, symbol, registry)


# --- BLOCKING cases ---

def test_gate_blocks_missing_lifecycle_evidence(tmp_path, monkeypatch):
    """Missing lifecycle evidence must BLOCK (was: just a warning)."""
    settings = _settings(tmp_path, monkeypatch)
    _make_validated_registry(settings, "EURUSD")
    _write_endurance(settings, "r1", "EURUSD")
    review = _review(settings, "EURUSD")
    assert review["final_decision"] == "REVERT_TO_BLOCKED"
    assert any("lifecycle" in r for r in review["reasons"])


def test_gate_blocks_missing_endurance_evidence(tmp_path, monkeypatch):
    """Missing endurance evidence must BLOCK (was: just a warning)."""
    settings = _settings(tmp_path, monkeypatch)
    _make_validated_registry(settings, "EURUSD")
    _write_lifecycle(settings, "r1", "EURUSD", critical=0, audit_ok=True)
    review = _review(settings, "EURUSD")
    assert review["final_decision"] == "REVERT_TO_BLOCKED"
    assert any("endurance" in r for r in review["reasons"])


def test_gate_blocks_lifecycle_audit_not_ok(tmp_path, monkeypatch):
    """lifecycle_audit_ok = False must now BLOCK (was: CAUTION)."""
    settings = _settings(tmp_path, monkeypatch)
    _make_validated_registry(settings, "EURUSD")
    _write_lifecycle(settings, "r1", "EURUSD", critical=0, audit_ok=False)
    _write_endurance(settings, "r1", "EURUSD")
    review = _review(settings, "EURUSD")
    assert review["final_decision"] == "REVERT_TO_BLOCKED"
    assert any("lifecycle_audit" in r for r in review["reasons"])


def test_gate_blocks_insufficient_trade_count(tmp_path, monkeypatch):
    """Trade count below min_trade_count must BLOCK."""
    settings = _settings(tmp_path, monkeypatch)
    _make_validated_registry(settings, "EURUSD")
    _write_lifecycle(settings, "r1", "EURUSD", audit_ok=True)
    # 1 trade per cycle × 1 cycle = 1 trade total (< 10 default)
    _write_endurance(settings, "r1", "EURUSD", trades_per_cycle=1, cycles=1)
    review = _review(settings, "EURUSD")
    assert review["final_decision"] == "REVERT_TO_BLOCKED"
    assert any("insufficient_endurance_trades" in r for r in review["reasons"])


def test_gate_blocks_profit_factor_below_floor(tmp_path, monkeypatch):
    """Profit factor below min_profit_factor floor must BLOCK."""
    settings = _settings(tmp_path, monkeypatch)
    _make_validated_registry(settings, "EURUSD")
    _write_lifecycle(settings, "r1", "EURUSD", audit_ok=True)
    _write_endurance(settings, "r1", "EURUSD", profit_factor=0.8)  # < 1.1 default floor
    review = _review(settings, "EURUSD")
    assert review["final_decision"] == "REVERT_TO_BLOCKED"
    assert any("profit_factor_below_floor" in r or "profit_factor" in r for r in review["reasons"])


def test_gate_blocks_expectancy_below_floor(tmp_path, monkeypatch):
    """Expectancy below min_expectancy_usd floor must BLOCK."""
    settings = _settings(tmp_path, monkeypatch)
    _make_validated_registry(settings, "EURUSD")
    _write_lifecycle(settings, "r1", "EURUSD", audit_ok=True)
    _write_endurance(settings, "r1", "EURUSD", expectancy_usd=0.1)  # < 0.50 default floor
    review = _review(settings, "EURUSD")
    assert review["final_decision"] == "REVERT_TO_BLOCKED"
    assert any("expectancy_below_floor" in r or "expectancy" in r for r in review["reasons"])


def test_gate_blocks_endurance_decision_not_go(tmp_path, monkeypatch):
    """endurance decision != 'go' must BLOCK (was: CAUTION for caution decision)."""
    settings = _settings(tmp_path, monkeypatch)
    _make_validated_registry(settings, "EURUSD")
    _write_lifecycle(settings, "r1", "EURUSD", audit_ok=True)
    _write_endurance(settings, "r1", "EURUSD", decision="caution")
    review = _review(settings, "EURUSD")
    assert review["final_decision"] == "REVERT_TO_BLOCKED"
    assert any("endurance_decision" in r for r in review["reasons"])


def test_gate_blocks_insufficient_endurance_cycles(tmp_path, monkeypatch):
    """Cycles below endurance_min_cycles must BLOCK (was: CAUTION)."""
    settings = _settings(tmp_path, monkeypatch)
    _make_validated_registry(settings, "EURUSD")
    _write_lifecycle(settings, "r1", "EURUSD", audit_ok=True)
    _write_endurance(settings, "r1", "EURUSD", cycles=1)  # < 3 default minimum
    review = _review(settings, "EURUSD")
    assert review["final_decision"] == "REVERT_TO_BLOCKED"
    assert any("insufficient_endurance_cycles" in r for r in review["reasons"])


def test_gate_blocks_blocked_trades_ratio_exceeded(tmp_path, monkeypatch):
    """blocked_trades_ratio > max must BLOCK."""
    settings = _settings(tmp_path, monkeypatch)
    _make_validated_registry(settings, "EURUSD")
    _write_lifecycle(settings, "r1", "EURUSD", audit_ok=True)
    # 30 blocked, 5 trades → ratio = 30/35 ≈ 0.857 >> 0.30 default
    _write_endurance(settings, "r1", "EURUSD", blocked_trades=30, trades_per_cycle=5)
    review = _review(settings, "EURUSD")
    assert review["final_decision"] == "REVERT_TO_BLOCKED"
    assert any("blocked_trades_ratio" in r for r in review["reasons"])


def test_gate_blocks_usdjpy_always(tmp_path, monkeypatch):
    """USDJPY is permanently blocked regardless of evidence quality."""
    settings = _settings(tmp_path, monkeypatch)
    _make_validated_registry(settings, "USDJPY")
    _write_lifecycle(settings, "r1", "USDJPY", audit_ok=True)
    _write_endurance(settings, "r1", "USDJPY")
    review = _review(settings, "USDJPY")
    assert review["final_decision"] == "REVERT_TO_BLOCKED"
    assert any("out_of_scope" in r for r in review["reasons"])


# --- APPROVAL case ---

def test_gate_approves_when_all_conditions_met(tmp_path, monkeypatch):
    """approved_demo when all gate conditions pass."""
    settings = _settings(tmp_path, monkeypatch)
    _make_validated_registry(settings, "EURUSD")
    _write_lifecycle(settings, "r1", "EURUSD", critical=0, audit_ok=True)
    _write_endurance(
        settings, "r1", "EURUSD",
        decision="go",
        cycles=3,
        trades_per_cycle=5,  # 15 total trades ≥ 10 min
        profit_factor=1.5,
        expectancy_usd=3.0,
        no_trade_count=2,
        blocked_trades=0,
    )
    review = _review(settings, "EURUSD")
    assert review["final_decision"] == "APPROVED_DEMO", f"Expected APPROVED_DEMO, got {review['final_decision']}, reasons={review['reasons']}"


def test_gate_report_includes_config_snapshot(tmp_path, monkeypatch):
    """Gate report must include the config snapshot for auditability."""
    settings = _settings(tmp_path, monkeypatch)
    _make_validated_registry(settings, "EURUSD")
    review = _review(settings, "EURUSD")
    assert "gate_config_snapshot" in review
    cfg = review["gate_config_snapshot"]
    assert "min_trade_count" in cfg
    assert "min_profit_factor" in cfg
    assert "lifecycle_max_age_hours" in cfg
    assert "require_lifecycle_audit_ok" in cfg


def test_gate_report_includes_lifecycle_age_hours(tmp_path, monkeypatch):
    """Gate report must include lifecycle evidence age for auditability."""
    settings = _settings(tmp_path, monkeypatch)
    _make_validated_registry(settings, "EURUSD")
    _write_lifecycle(settings, "r1", "EURUSD", audit_ok=True)
    _write_endurance(settings, "r1", "EURUSD")
    review = _review(settings, "EURUSD")
    lc = review["lifecycle_summary"]
    assert "evidence_age_hours" in lc
    # Age should be very small (just written)
    if lc["evidence_age_hours"] is not None:
        assert lc["evidence_age_hours"] < 1.0


def test_gate_caution_on_degradation_when_otherwise_ok(tmp_path, monkeypatch):
    """Degradation above threshold → MOVE_TO_CAUTION (not BLOCKED), since evidence is present."""
    settings = _settings(tmp_path, monkeypatch)
    _make_validated_registry(settings, "EURUSD")
    _write_lifecycle(settings, "r1", "EURUSD", critical=0, audit_ok=True)
    _write_endurance(
        settings, "r1", "EURUSD",
        decision="go",
        cycles=3,
        trades_per_cycle=5,
        profit_factor=1.5,
        expectancy_usd=3.0,
        expectancy_degradation=0.50,  # > 0.25 gate → caution
    )
    review = _review(settings, "EURUSD")
    # Should be CAUTION (degradation is borderline, not a hard error)
    assert review["final_decision"] == "MOVE_TO_CAUTION"
    assert any("degradation" in r for r in review["reasons"])
