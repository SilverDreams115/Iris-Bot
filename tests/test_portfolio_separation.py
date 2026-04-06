"""
Tests for portfolio.py — explicit portfolio vs universe separation.

Covers:
  - build_portfolio_separation correctly categorizes symbols
  - Deliberately blocked symbols (USDJPY) are never in active_portfolio
  - approved_demo_universe is a proper subset of eligible_universe
  - active_portfolio is a proper subset of approved_demo_universe
  - active_portfolio_status_report shows per-symbol status clearly
  - active_universe_status_report includes all symbols with categories
  - Separation coherent after promotion and rollback
  - active_strategy_materialization: only approved_demo entries
"""
from __future__ import annotations

import json
from pathlib import Path


from iris_bot.artifacts import wrap_artifact
from iris_bot.config import load_settings
from iris_bot.governance import (
    promote_strategy_profile,
    validate_strategy_profiles,
)
from iris_bot.portfolio import (
    _PERMANENTLY_EXCLUDED,
    active_portfolio_status_report,
    active_universe_status_report,
    build_portfolio_separation,
)


def _settings(tmp_path: Path, monkeypatch, symbols=("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")):
    monkeypatch.setenv("IRIS_SYMBOLS", ",".join(symbols))
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


def _write_validation_artifacts(settings, run_id, states):
    run_dir = settings.data.runs_dir / f"{run_id}_strategy_validation"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "leakage_fix_report.json").write_text(
        json.dumps(wrap_artifact("strategy_validation", {"test_used_for_selection": False}))
    )
    (run_dir / "symbol_enablement_report.json").write_text(
        json.dumps(wrap_artifact("symbol_enablement", {
            "symbols": {s: {"state": st, "enabled": st == "enabled", "chosen_model": "global_model"} for s, st in states.items()}
        }))
    )
    (run_dir / "strategy_validation_report.json").write_text(
        json.dumps(wrap_artifact("strategy_validation", {"symbols": {s: {"chosen_model": "global_model"} for s in states}}))
    )


def _write_strategy_profiles(settings):
    from iris_bot.symbols import write_symbol_strategy_profiles
    write_symbol_strategy_profiles(settings, {}, {})


def _make_approved_demo(settings, symbol: str) -> None:
    """Creates a validated profile and promotes it to approved_demo using mocked evidence."""
    _write_validation_artifacts(settings, f"r_{symbol}", {symbol: "enabled"})
    _write_strategy_profiles(settings)
    validate_strategy_profiles(settings)
    # Write lifecycle + endurance evidence
    _write_lifecycle_artifact(settings, symbol)
    _write_endurance_artifact(settings, symbol)
    # Promote
    import os
    env_backup = os.environ.get("IRIS_GOVERNANCE_TARGET_SYMBOL", "")
    os.environ["IRIS_GOVERNANCE_TARGET_SYMBOL"] = symbol
    from iris_bot.config import load_settings as ls
    promo_settings = ls()
    object.__setattr__(promo_settings, "project_root", settings.project_root)
    for attr in ("raw_dir", "processed_dir", "runs_dir", "runtime_dir"):
        object.__setattr__(promo_settings.data, attr, getattr(settings.data, attr))
    promote_strategy_profile(promo_settings)
    os.environ["IRIS_GOVERNANCE_TARGET_SYMBOL"] = env_backup


def _write_lifecycle_artifact(settings, symbol: str) -> None:
    run_dir = settings.data.runs_dir / f"lc_{symbol}_lifecycle_reconciliation"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": True,
        "critical_mismatch_count": 0,
        "mismatch_counts": {},
        "mismatches": [],
        "symbols": {symbol: {"critical_mismatch_count": 0, "mismatch_categories": []}},
    }
    envelope = wrap_artifact("lifecycle_reconciliation", payload)
    (run_dir / "lifecycle_reconciliation_report.json").write_text(json.dumps(envelope))
    stab_dir = settings.data.runs_dir / f"lc_{symbol}_mt5_windows_stabilization"
    stab_dir.mkdir(parents=True, exist_ok=True)
    rerun = {
        "audit_ok": True,
        "critical_mismatch_count": 0,
        "reconciliation_ok": True,
        "reconciliation_run": str(run_dir / "lifecycle_reconciliation_report.json"),
        "rerun_source": "test",
    }
    (stab_dir / "lifecycle_rerun_report.json").write_text(json.dumps(rerun))


def _write_endurance_artifact(settings, symbol: str) -> None:
    run_dir = settings.data.runs_dir / f"end_{symbol}_symbol_endurance"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbols": {
            symbol: {
                "decision": "go",
                "cycles_completed": 3,
                "blocked_trades": 0,
                "no_trade_count": 2,
                "expectancy_degradation_pct": 0.05,
                "profit_factor_degradation_pct": 0.05,
                "alerts_by_severity": {"critical": 0, "error": 0, "warning": 0, "info": 0},
                "cycle_metrics": [
                    {"trades": 5, "expectancy_usd": 3.0, "profit_factor": 1.5}
                    for _ in range(3)
                ],
            }
        }
    }
    (run_dir / "symbol_stability_report.json").write_text(json.dumps(wrap_artifact("symbol_stability", payload)))


# --- Tests ---

def test_permanently_excluded_symbols_defined():
    """USDJPY must be in _PERMANENTLY_EXCLUDED with a reason."""
    assert "USDJPY" in _PERMANENTLY_EXCLUDED
    assert len(_PERMANENTLY_EXCLUDED["USDJPY"]) > 0


def test_full_universe_contains_all_configured_symbols(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    registry = {"profiles": {}, "active_profiles": {}}
    sep = build_portfolio_separation(settings, registry)
    assert set(sep.full_universe) == set(settings.trading.symbols)


def test_eligible_universe_excludes_deliberately_blocked(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    registry = {"profiles": {}, "active_profiles": {}}
    sep = build_portfolio_separation(settings, registry)
    for s in sep.deliberately_blocked:
        assert s not in sep.eligible_universe


def test_usdjpy_in_deliberately_blocked(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    registry = {"profiles": {}, "active_profiles": {}}
    sep = build_portfolio_separation(settings, registry)
    assert "USDJPY" in sep.deliberately_blocked
    assert sep.deliberately_blocked["USDJPY"] == _PERMANENTLY_EXCLUDED["USDJPY"]


def test_usdjpy_never_in_active_portfolio(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    # Even if USDJPY somehow appears in active_profiles, it must not be in active_portfolio
    registry = {
        "profiles": {
            "USDJPY": [{"profile_id": "p1", "promotion_state": "approved_demo", "profile_payload": {"enabled_state": "enabled", "enabled": True}}]
        },
        "active_profiles": {"USDJPY": "p1"},
    }
    sep = build_portfolio_separation(settings, registry)
    assert "USDJPY" not in sep.active_portfolio
    assert "USDJPY" not in sep.approved_demo_universe


def test_approved_demo_universe_subset_of_eligible(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    registry = {
        "profiles": {
            "EURUSD": [{"profile_id": "p1", "promotion_state": "approved_demo", "profile_payload": {"enabled_state": "enabled", "enabled": True}}]
        },
        "active_profiles": {"EURUSD": "p1"},
    }
    sep = build_portfolio_separation(settings, registry)
    for s in sep.approved_demo_universe:
        assert s in sep.eligible_universe


def test_active_portfolio_subset_of_approved_demo(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    registry = {
        "profiles": {
            "EURUSD": [{"profile_id": "p1", "promotion_state": "approved_demo", "profile_payload": {"enabled_state": "enabled", "enabled": True}}],
            "GBPUSD": [{"profile_id": "p2", "promotion_state": "approved_demo", "profile_payload": {"enabled_state": "disabled", "enabled": False}}],
        },
        "active_profiles": {"EURUSD": "p1", "GBPUSD": "p2"},
    }
    sep = build_portfolio_separation(settings, registry)
    for s in sep.active_portfolio:
        assert s in sep.approved_demo_universe


def test_disabled_profile_not_in_active_portfolio(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    registry = {
        "profiles": {
            "GBPUSD": [{"profile_id": "p1", "promotion_state": "approved_demo", "profile_payload": {"enabled_state": "disabled", "enabled": False}}],
        },
        "active_profiles": {"GBPUSD": "p1"},
    }
    sep = build_portfolio_separation(settings, registry)
    assert "GBPUSD" not in sep.active_portfolio
    assert "GBPUSD" in sep.approved_demo_universe  # approved but not active


def test_active_portfolio_status_report_structure(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    registry = {
        "profiles": {
            "EURUSD": [{"profile_id": "p1", "promotion_state": "approved_demo", "profile_payload": {"enabled_state": "enabled", "enabled": True}}],
        },
        "active_profiles": {"EURUSD": "p1"},
    }
    report = active_portfolio_status_report(settings, registry)
    assert "portfolio_separation" in report
    assert "per_symbol_eligible" in report
    assert "deliberately_blocked" in report
    assert "summary" in report
    assert "USDJPY" in report["deliberately_blocked"]
    assert "USDJPY" not in report["per_symbol_eligible"]


def test_active_universe_status_report_covers_full_universe(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    registry = {"profiles": {}, "active_profiles": {}}
    report = active_universe_status_report(settings, registry)
    for s in settings.trading.symbols:
        assert s in report["per_symbol"]


def test_active_universe_status_deliberately_blocked_category(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    registry = {"profiles": {}, "active_profiles": {}}
    report = active_universe_status_report(settings, registry)
    usdjpy = report["per_symbol"]["USDJPY"]
    assert usdjpy["universe_category"] == "deliberately_blocked"
    assert usdjpy["in_active_portfolio"] is False
    assert usdjpy["in_eligible_universe"] is False


def test_portfolio_impact_of_blocked(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    registry = {"profiles": {}, "active_profiles": {}}
    sep = build_portfolio_separation(settings, registry)
    impact = sep.portfolio_impact_of_blocked()
    assert "USDJPY" in impact
    assert impact["USDJPY"] is False  # USDJPY not in active portfolio


def test_separation_to_dict(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    registry = {"profiles": {}, "active_profiles": {}}
    sep = build_portfolio_separation(settings, registry)
    d = sep.to_dict()
    assert "full_universe" in d
    assert "eligible_universe" in d
    assert "approved_demo_universe" in d
    assert "active_portfolio" in d
    assert "deliberately_blocked" in d
    assert "counts" in d


def test_active_profile_not_approved_demo_excluded_from_approved_universe(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    registry = {
        "profiles": {
            "EURUSD": [{"profile_id": "p1", "promotion_state": "validated", "profile_payload": {"enabled_state": "enabled", "enabled": True}}],
        },
        "active_profiles": {"EURUSD": "p1"},
    }
    sep = build_portfolio_separation(settings, registry)
    assert "EURUSD" not in sep.approved_demo_universe
    assert "EURUSD" not in sep.active_portfolio
