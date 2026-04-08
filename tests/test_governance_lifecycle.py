import csv
import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

from iris_bot.artifacts import read_artifact_payload, wrap_artifact
from iris_bot.config import MT5Config, load_settings
from iris_bot.governance import (
    active_strategy_status,
    diagnose_profile_activation,
    ingest_governance_evidence,
    load_strategy_profile_registry,
    promote_strategy_profile,
    review_approved_demo_readiness,
    resolve_active_profile_entry,
    rollback_strategy_profile,
    save_strategy_profile_registry,
    validate_strategy_profiles,
)
from iris_bot.evidence_store import evidence_store_status
from iris_bot.lifecycle import reconcile_lifecycle_records, run_lifecycle_reconciliation
from iris_bot.operational import AccountState, PaperEngineState, atomic_write_json
from iris_bot.paper import load_paper_context, run_paper_engine, PaperSessionConfig
from iris_bot.processed_dataset import ProcessedRow
from iris_bot.symbol_endurance import _endurance_consistency, run_symbol_endurance
from iris_bot.symbols import default_symbol_strategy_profile, write_symbol_strategy_profiles


def _settings(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("IRIS_PRIMARY_TIMEFRAME", "M15")
    settings = load_settings()
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    runs_dir = tmp_path / "runs"
    runtime_dir = tmp_path / "data" / "runtime"
    for path in (raw_dir, processed_dir, runs_dir, runtime_dir):
        path.mkdir(parents=True, exist_ok=True)
    object.__setattr__(settings.data, "raw_dir", raw_dir)
    object.__setattr__(settings.data, "processed_dir", processed_dir)
    object.__setattr__(settings.data, "runs_dir", runs_dir)
    object.__setattr__(settings.data, "runtime_dir", runtime_dir)
    object.__setattr__(settings.experiment, "_processed_dir", processed_dir)
    return settings


def _write_validation_artifacts(settings, run_id: str, states: dict[str, str]) -> Path:
    run_dir = settings.data.runs_dir / f"{run_id}_strategy_validation"
    run_dir.mkdir(parents=True, exist_ok=True)
    leakage = {"test_used_for_selection": False}
    enablement = {
        "symbols": {
            symbol: {
                "state": state,
                "enabled": state == "enabled",
                "chosen_model": "global_model",
                "chosen_exit_policy": "static",
                "selection_based": True,
            }
            for symbol, state in states.items()
        }
    }
    comparison = {
        "symbols": {
            symbol: {"chosen_model": "global_model", "chosen_exit_policy": "static"}
            for symbol in states
        }
    }
    (run_dir / "leakage_fix_report.json").write_text(json.dumps(wrap_artifact("strategy_validation", leakage)), encoding="utf-8")
    (run_dir / "symbol_enablement_report.json").write_text(json.dumps(wrap_artifact("symbol_enablement", enablement)), encoding="utf-8")
    (run_dir / "strategy_validation_report.json").write_text(json.dumps(wrap_artifact("strategy_validation", comparison)), encoding="utf-8")
    return run_dir


def _write_stability_artifact(settings, run_id: str, symbol: str, decision: str = "go") -> Path:
    run_dir = settings.data.runs_dir / f"{run_id}_symbol_endurance"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbols": {
            symbol: {
                "decision": decision,
                "cycles_completed": 3,
                "expectancy_degradation_pct": 0.05,
                "profit_factor_degradation_pct": 0.05,
                "alerts_by_severity": {"info": 0, "warning": 0, "error": 0, "critical": 0},
                "cycle_metrics": [
                    {"trades": 4, "expectancy_usd": 5.0, "profit_factor": 2.0, "net_pnl_usd": 20.0},
                    {"trades": 4, "expectancy_usd": 5.0, "profit_factor": 2.0, "net_pnl_usd": 20.0},
                    {"trades": 4, "expectancy_usd": 5.0, "profit_factor": 2.0, "net_pnl_usd": 20.0},
                ],
            }
        }
    }
    (run_dir / "symbol_stability_report.json").write_text(json.dumps(wrap_artifact("symbol_stability", payload)), encoding="utf-8")
    return run_dir


def _write_lifecycle_artifact(settings, run_id: str, symbol: str, critical: int = 0) -> Path:
    run_dir = settings.data.runs_dir / f"{run_id}_lifecycle_reconciliation"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "symbols": {
            symbol: {
                "critical_mismatch_count": critical,
                "mismatch_categories": [],
            }
        }
    }
    (run_dir / "lifecycle_reconciliation_report.json").write_text(json.dumps(wrap_artifact("lifecycle_reconciliation", payload)), encoding="utf-8")
    stabilization_dir = settings.data.runs_dir / f"{run_id}_mt5_windows_stabilization"
    stabilization_dir.mkdir(parents=True, exist_ok=True)
    rerun_report = {
        "audit_ok": True,
        "audit_run": str(run_dir / "lifecycle_reconciliation_report.json"),
        "critical_mismatch_count": critical,
        "reconciliation_ok": critical == 0,
        "reconciliation_run": str(run_dir / "lifecycle_reconciliation_report.json"),
        "rerun_source": "test_fixture",
        "workspace": str(settings.project_root),
    }
    (stabilization_dir / "lifecycle_rerun_report.json").write_text(json.dumps(rerun_report), encoding="utf-8")
    return run_dir


def _write_fake_soak_dir(base: Path, symbol: str, decision: str = "go") -> Path:
    base.mkdir(parents=True, exist_ok=True)
    cycle_summaries = base / "cycle_summaries"
    cycle_summaries.mkdir(parents=True, exist_ok=True)
    cycles = [
        {"cycle": 1, "status": decision, "alerts_by_severity": {"info": 0, "warning": 0, "error": 0, "critical": 0}},
        {"cycle": 2, "status": decision, "alerts_by_severity": {"info": 0, "warning": 0, "error": 0, "critical": 0}},
        {"cycle": 3, "status": decision, "alerts_by_severity": {"info": 0, "warning": 0, "error": 0, "critical": 0}},
    ]
    (base / "health_report.json").write_text(json.dumps({"cycles": cycles}), encoding="utf-8")
    (base / "soak_report.json").write_text(json.dumps({"cycles_completed": 3}), encoding="utf-8")
    (base / "go_no_go_report.json").write_text(json.dumps({"decision": decision}), encoding="utf-8")
    for cycle in cycles:
        (cycle_summaries / f"cycle_{cycle['cycle']:02d}.json").write_text(json.dumps(cycle), encoding="utf-8")
    with (base / "closed_trades.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "net_pnl_usd", "target_policy_details"])
        writer.writeheader()
        writer.writerow({"symbol": symbol, "net_pnl_usd": "10.0", "target_policy_details": json.dumps({"session": "london"})})
    with (base / "signal_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "status", "signal", "threshold"])
        writer.writeheader()
        writer.writerow({"symbol": symbol, "status": "generated", "signal": "1", "threshold": "0.60"})
    with (base / "execution_journal.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol"])
        writer.writeheader()
        writer.writerow({"symbol": symbol})
    return base


def _write_no_trade_soak_dir(base: Path, symbol: str, decision: str = "go") -> Path:
    base.mkdir(parents=True, exist_ok=True)
    cycle_summaries = base / "cycle_summaries"
    cycle_summaries.mkdir(parents=True, exist_ok=True)
    cycles = [
        {"cycle": 1, "status": decision, "alerts_by_severity": {"info": 0, "warning": 0, "error": 0, "critical": 0}},
        {"cycle": 2, "status": decision, "alerts_by_severity": {"info": 0, "warning": 0, "error": 0, "critical": 0}},
        {"cycle": 3, "status": decision, "alerts_by_severity": {"info": 0, "warning": 0, "error": 0, "critical": 0}},
    ]
    (base / "health_report.json").write_text(json.dumps({"cycles": cycles}), encoding="utf-8")
    (base / "soak_report.json").write_text(json.dumps({"cycles_completed": 3}), encoding="utf-8")
    (base / "go_no_go_report.json").write_text(json.dumps({"decision": decision}), encoding="utf-8")
    for cycle in cycles:
        (cycle_summaries / f"cycle_{cycle['cycle']:02d}.json").write_text(json.dumps(cycle), encoding="utf-8")
    with (base / "closed_trades.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "net_pnl_usd", "target_policy_details"])
        writer.writeheader()
    with (base / "signal_log.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "status", "signal", "threshold"])
        writer.writeheader()
        writer.writerow({"symbol": symbol, "status": "generated", "signal": "0", "threshold": "0.60"})
    with (base / "execution_journal.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol"])
        writer.writeheader()
    return base


class LifecycleFakeClient:
    def __init__(self, payload, *, connect_ok: bool = True):
        self.payload = payload
        self.config = replace(MT5Config(), enabled=True)
        self.connect_ok = connect_ok

    def connect(self) -> bool:
        return self.connect_ok

    def shutdown(self) -> None:
        return None

    def broker_lifecycle_snapshot(self, symbols, history_days):
        return self.payload


def test_run_symbol_endurance_filters_enabled_symbols(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    settings = replace(settings, trading=replace(settings.trading, symbols=("EURUSD", "GBPUSD")))
    settings = replace(settings, endurance=replace(settings.endurance, min_cycles_for_stability=1))
    write_symbol_strategy_profiles(
        settings,
        {},
        {
            "EURUSD": {"enabled_state": "enabled", "enabled": True},
            "GBPUSD": {"enabled_state": "disabled", "enabled": False},
        },
    )

    def fake_run_soak(scoped_settings, mode, require_broker):
        symbol = scoped_settings.trading.symbols[0]
        return 0, _write_fake_soak_dir(settings.data.runs_dir / f"{symbol}_fake_soak", symbol)

    monkeypatch.setattr("iris_bot.symbol_endurance.run_soak", fake_run_soak)
    exit_code, run_dir = run_symbol_endurance(settings, only_enabled=True)
    payload = read_artifact_payload(run_dir / "symbol_stability_report.json", expected_type="symbol_stability")
    assert exit_code == 0
    assert "EURUSD" in payload["symbols"]
    assert "GBPUSD" not in payload["symbols"]
    assert payload["symbols"]["EURUSD"]["cycles_completed"] == 3
    assert payload["symbols"]["EURUSD"]["source_of_truth"] == "health_report.cycles"
    assert payload["symbols"]["EURUSD"]["consistency_reports"][0]["ok"] is True


def test_default_mt5_ownership_mode_is_strict(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    assert settings.mt5.ownership_mode == "strict"


def test_ingest_governance_evidence_populates_canonical_store(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    object.__setattr__(settings, "project_root", tmp_path)
    registry = {
        "profiles": {
            "EURUSD": [
                {
                    "profile_id": "eurusd_approved",
                    "promotion_state": "approved_demo",
                    "enablement_state": "enabled",
                    "checksum": "abc",
                    "created_at": "2026-04-06T00:00:00+00:00",
                    "source_run_id": "20260406T000000Z_strategy_profile_promotion",
                    "symbol": "EURUSD",
                    "model_variant": "global_model",
                    "promotion_reason": "test",
                    "rollback_target": None,
                    "profile_payload": {
                        "symbol": "EURUSD",
                        "profile_id": "eurusd_approved",
                        "promotion_state": "approved_demo",
                        "enabled_state": "enabled",
                        "enabled": True,
                        "promotion_reason": "test",
                        "rollback_target": None,
                    },
                }
            ]
        },
        "active_profiles": {"EURUSD": "eurusd_approved"},
    }
    import hashlib
    import json as _json

    payload = dict(registry["profiles"]["EURUSD"][0]["profile_payload"])
    for key in ("profile_id", "promotion_state", "promotion_reason", "rollback_target"):
        payload.pop(key, None)
    registry["profiles"]["EURUSD"][0]["checksum"] = hashlib.sha256(_json.dumps(payload, sort_keys=True).encode()).hexdigest()
    save_strategy_profile_registry(settings, registry)

    _write_lifecycle_artifact(settings, "20260406T010000Z", "EURUSD", critical=0)
    _write_stability_artifact(settings, "20260406T020000Z", "EURUSD", decision="go")

    assert ingest_governance_evidence(settings) == 0
    status = evidence_store_status(settings)
    assert status["by_artifact_type"]["lifecycle_reconciliation"] >= 1
    assert status["by_artifact_type"]["symbol_stability"] >= 1
    latest_keys = status["latest_per_key"]
    assert "lifecycle_reconciliation__global" in latest_keys
    assert "symbol_stability__EURUSD" in latest_keys


def test_endurance_consistency_detects_cycle_mismatch(tmp_path: Path) -> None:
    soak_dir = _write_fake_soak_dir(tmp_path / "mismatch_soak", "EURUSD")
    (soak_dir / "soak_report.json").write_text(json.dumps({"cycles_completed": 2}), encoding="utf-8")
    report = _endurance_consistency(soak_dir)
    assert report["ok"] is False
    assert "soak_vs_health_cycle_count" in report["mismatches"]


def test_endurance_consistency_reports_correct_counts(tmp_path: Path) -> None:
    soak_dir = _write_fake_soak_dir(tmp_path / "clean_soak", "EURUSD")
    report = _endurance_consistency(soak_dir)
    assert report["ok"] is True
    assert report["health_cycle_count"] == 3
    assert report["reported_completed_cycles"] == 3
    assert report["cycle_summary_count"] == 3


def test_run_symbol_endurance_blocks_when_reporting_is_inconsistent(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    settings = replace(settings, trading=replace(settings.trading, symbols=("EURUSD",)))
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "enabled", "enabled": True}})

    def fake_run_soak(scoped_settings, mode, require_broker):
        soak_dir = _write_fake_soak_dir(settings.data.runs_dir / "broken_soak", "EURUSD")
        (soak_dir / "soak_report.json").write_text(json.dumps({"cycles_completed": 1}), encoding="utf-8")
        return 0, soak_dir

    monkeypatch.setattr("iris_bot.symbol_endurance.run_soak", fake_run_soak)
    exit_code, run_dir = run_symbol_endurance(settings, only_enabled=False)
    payload = read_artifact_payload(run_dir / "symbol_stability_report.json", expected_type="symbol_stability")
    assert exit_code == 2
    assert payload["symbols"]["EURUSD"]["decision"] == "blocked"
    assert "endurance_reporting_inconsistent" in payload["symbols"]["EURUSD"]["reasons"]


def test_run_symbol_endurance_uses_review_scope_for_blocked_symbols(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    settings = replace(settings, trading=replace(settings.trading, symbols=("EURUSD",)))
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "blocked", "enabled": False}})
    captured: dict[str, object] = {}

    def fake_run_soak(scoped_settings, mode, require_broker, allowed_profile_states=None):
        captured["symbols"] = scoped_settings.trading.symbols
        captured["runtime_dir"] = str(scoped_settings.data.runtime_dir)
        captured["allowed_profile_states"] = allowed_profile_states
        return 0, _write_fake_soak_dir(settings.data.runs_dir / "review_scope_soak", "EURUSD")

    monkeypatch.setattr("iris_bot.symbol_endurance.run_soak", fake_run_soak)
    exit_code, run_dir = run_symbol_endurance(settings, only_enabled=False)
    payload = read_artifact_payload(run_dir / "symbol_stability_report.json", expected_type="symbol_stability")
    assert exit_code == 2
    assert captured["symbols"] == ("EURUSD",)
    assert captured["runtime_dir"].endswith("/endurance_reviews/EURUSD")
    assert "blocked" in set(captured["allowed_profile_states"])
    assert payload["symbols"]["EURUSD"]["cycles_completed"] == 3
    assert payload["symbols"]["EURUSD"]["decision"] == "blocked"


def test_run_symbol_endurance_blocks_when_no_trades_execute(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    settings = replace(settings, trading=replace(settings.trading, symbols=("EURUSD",)))
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "blocked", "enabled": False}})

    def fake_run_soak(scoped_settings, mode, require_broker, allowed_profile_states=None):
        return 0, _write_no_trade_soak_dir(settings.data.runs_dir / "no_trade_soak", "EURUSD")

    monkeypatch.setattr("iris_bot.symbol_endurance.run_soak", fake_run_soak)
    exit_code, run_dir = run_symbol_endurance(settings, only_enabled=False)
    payload = read_artifact_payload(run_dir / "symbol_stability_report.json", expected_type="symbol_stability")
    assert exit_code == 2
    assert payload["symbols"]["EURUSD"]["decision"] == "blocked"
    assert "no_trades_executed" in payload["symbols"]["EURUSD"]["reasons"]


def test_run_symbol_endurance_blocks_disabled_profile_even_with_positive_metrics(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    settings = replace(settings, trading=replace(settings.trading, symbols=("EURUSD",)))
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "disabled", "enabled": False}})

    def fake_run_soak(scoped_settings, mode, require_broker, allowed_profile_states=None):
        return 0, _write_fake_soak_dir(settings.data.runs_dir / "positive_disabled_soak", "EURUSD")

    monkeypatch.setattr("iris_bot.symbol_endurance.run_soak", fake_run_soak)
    exit_code, run_dir = run_symbol_endurance(settings, only_enabled=False)
    payload = read_artifact_payload(run_dir / "symbol_stability_report.json", expected_type="symbol_stability")
    assert exit_code == 2
    assert payload["symbols"]["EURUSD"]["decision"] == "blocked"
    assert "profile_not_tradable" in payload["symbols"]["EURUSD"]["reasons"]


def test_run_symbol_endurance_blocks_non_positive_expectancy(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    settings = replace(settings, trading=replace(settings.trading, symbols=("EURUSD",)))
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "enabled", "enabled": True}})

    def fake_run_soak(scoped_settings, mode, require_broker, allowed_profile_states=None):
        soak_dir = _write_fake_soak_dir(settings.data.runs_dir / "negative_soak", "EURUSD")
        with (soak_dir / "closed_trades.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["symbol", "net_pnl_usd", "target_policy_details"])
            writer.writeheader()
            writer.writerow({"symbol": "EURUSD", "net_pnl_usd": "-10.0", "target_policy_details": json.dumps({"session": "london"})})
        return 0, soak_dir

    monkeypatch.setattr("iris_bot.symbol_endurance.run_soak", fake_run_soak)
    exit_code, run_dir = run_symbol_endurance(settings, only_enabled=False)
    payload = read_artifact_payload(run_dir / "symbol_stability_report.json", expected_type="symbol_stability")
    assert exit_code == 2
    assert payload["symbols"]["EURUSD"]["decision"] == "blocked"
    assert "non_positive_expectancy" in payload["symbols"]["EURUSD"]["reasons"]


def test_validate_promote_and_rollback_strategy_profile(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(
        settings,
        {},
        {
            "EURUSD": {"enabled_state": "enabled", "enabled": True, "threshold": 0.6},
            "GBPUSD": {"enabled_state": "disabled", "enabled": False},
        },
    )
    _write_validation_artifacts(settings, "20260401T000000Z", {"EURUSD": "enabled", "GBPUSD": "disabled"})
    assert validate_strategy_profiles(settings) == 0
    registry = load_strategy_profile_registry(settings)
    eurusd_entry = registry["profiles"]["EURUSD"][0]
    assert eurusd_entry["promotion_state"] == "validated"

    _write_stability_artifact(settings, "20260401T010000Z", "EURUSD", decision="go")
    _write_lifecycle_artifact(settings, "20260401T020000Z", "EURUSD", critical=0)
    settings = replace(settings, governance=replace(settings.governance, target_symbol="EURUSD"))
    assert promote_strategy_profile(settings) == 0
    registry = load_strategy_profile_registry(settings)
    active_id = registry["active_profiles"]["EURUSD"]
    assert registry["profiles"]["EURUSD"][0]["promotion_state"] == "approved_demo"

    second_profile = dict(registry["profiles"]["EURUSD"][0])
    second_profile["profile_id"] = "EURUSD-manual-older"
    second_profile["promotion_state"] = "deprecated"
    second_profile["profile_payload"] = dict(second_profile["profile_payload"], profile_id="EURUSD-manual-older", promotion_state="deprecated")
    registry["profiles"]["EURUSD"].append(second_profile)
    registry["profiles"]["EURUSD"][0]["rollback_target"] = "EURUSD-manual-older"
    from iris_bot.governance import save_strategy_profile_registry

    save_strategy_profile_registry(settings, registry)
    assert rollback_strategy_profile(settings) == 0
    registry = load_strategy_profile_registry(settings)
    assert registry["active_profiles"]["EURUSD"] == "EURUSD-manual-older"
    assert active_id != registry["active_profiles"]["EURUSD"]


def test_validate_strategy_profile_rejects_incompatible_checksum(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    path = settings.data.runtime_dir / settings.strategy.profiles_filename
    broken = wrap_artifact("strategy_profiles", {"common": {}, "symbols": {}})
    broken["checksum"] = "bad"
    path.write_text(json.dumps(broken), encoding="utf-8")
    _write_validation_artifacts(settings, "20260401T000000Z", {"EURUSD": "enabled"})
    assert validate_strategy_profiles(settings) == 1


def test_active_strategy_status_blocks_missing_active_profile(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "enabled", "enabled": True}})
    assert active_strategy_status(settings) == 2


def test_resolve_active_profile_entry_reports_exact_root_cause(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "disabled", "enabled": False}})
    _write_validation_artifacts(settings, "20260401T000000Z", {"EURUSD": "disabled"})
    assert validate_strategy_profiles(settings) == 0
    status = resolve_active_profile_entry(settings, "EURUSD")
    assert status["ok"] is False
    assert "no_active_profile_registered" in status["reasons"]


def test_promoted_profile_resolves_valid_active_profile(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "enabled", "enabled": True}})
    _write_validation_artifacts(settings, "20260401T000000Z", {"EURUSD": "enabled"})
    assert validate_strategy_profiles(settings) == 0
    _write_stability_artifact(settings, "20260401T010000Z", "EURUSD", decision="go")
    _write_lifecycle_artifact(settings, "20260401T020000Z", "EURUSD", critical=0)
    settings = replace(settings, governance=replace(settings.governance, target_symbol="EURUSD"))
    assert promote_strategy_profile(settings) == 0
    status = resolve_active_profile_entry(settings, "EURUSD")
    assert status["ok"] is True
    assert status["active_profile_id"].startswith("EURUSD-")
    assert status["resolved_profile"] is not None


def test_first_time_promotion_creates_reversible_rollback_target(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "enabled", "enabled": True}})
    _write_validation_artifacts(settings, "20260401T000000Z", {"EURUSD": "enabled"})
    assert validate_strategy_profiles(settings) == 0
    _write_stability_artifact(settings, "20260401T010000Z", "EURUSD", decision="go")
    _write_lifecycle_artifact(settings, "20260401T020000Z", "EURUSD", critical=0)
    settings = replace(settings, governance=replace(settings.governance, target_symbol="EURUSD"))
    assert promote_strategy_profile(settings) == 0
    registry = load_strategy_profile_registry(settings)
    active_id = registry["active_profiles"]["EURUSD"]
    active_entry = next(item for item in registry["profiles"]["EURUSD"] if item["profile_id"] == active_id)
    assert active_entry["rollback_target"]
    rollback_entry = next(item for item in registry["profiles"]["EURUSD"] if item["profile_id"] == active_entry["rollback_target"])
    assert rollback_entry["promotion_state"] == "deprecated"


def test_repromotion_of_same_profile_id_creates_distinct_rollback_target(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "enabled", "enabled": True}})
    _write_validation_artifacts(settings, "20260401T000000Z", {"EURUSD": "enabled"})
    assert validate_strategy_profiles(settings) == 0
    _write_stability_artifact(settings, "20260401T010000Z", "EURUSD", decision="go")
    _write_lifecycle_artifact(settings, "20260401T020000Z", "EURUSD", critical=0)
    settings = replace(settings, governance=replace(settings.governance, target_symbol="EURUSD"))
    assert promote_strategy_profile(settings) == 0
    assert validate_strategy_profiles(settings) == 0
    assert promote_strategy_profile(settings) == 0
    registry = load_strategy_profile_registry(settings)
    active_id = registry["active_profiles"]["EURUSD"]
    active_entry = next(item for item in registry["profiles"]["EURUSD"] if item["profile_id"] == active_id)
    assert active_entry["rollback_target"]
    assert active_entry["rollback_target"] != active_id


def test_review_approved_demo_readiness_keeps_validated_without_endurance(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "enabled", "enabled": True}})
    _write_validation_artifacts(settings, "20260401T000000Z", {"EURUSD": "enabled"})
    assert validate_strategy_profiles(settings) == 0
    settings = replace(settings, governance=replace(settings.governance, target_symbol="EURUSD"))
    assert review_approved_demo_readiness(settings) == 2
    latest = sorted(settings.data.runs_dir.glob("*_review_approved_demo_readiness"))[-1]
    payload = read_artifact_payload(latest / "approved_demo_readiness_report.json", expected_type="strategy_profile_promotion")
    assert payload["symbols"]["EURUSD"]["final_decision"] == "KEEP_VALIDATED"
    assert "missing_endurance_validation" in payload["symbols"]["EURUSD"]["reasons"]


def test_review_approved_demo_readiness_moves_to_caution_on_degradation(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "enabled", "enabled": True}})
    _write_validation_artifacts(settings, "20260401T000000Z", {"EURUSD": "enabled"})
    assert validate_strategy_profiles(settings) == 0
    _write_stability_artifact(settings, "20260401T010000Z", "EURUSD", decision="caution")
    _write_lifecycle_artifact(settings, "20260401T020000Z", "EURUSD", critical=0)
    settings = replace(settings, governance=replace(settings.governance, target_symbol="EURUSD"))
    assert review_approved_demo_readiness(settings) == 2
    latest = sorted(settings.data.runs_dir.glob("*_review_approved_demo_readiness"))[-1]
    payload = read_artifact_payload(latest / "approved_demo_readiness_report.json", expected_type="strategy_profile_promotion")
    assert payload["symbols"]["EURUSD"]["final_decision"] == "MOVE_TO_CAUTION"
    assert "endurance_decision=caution" in payload["symbols"]["EURUSD"]["reasons"]


def test_review_approved_demo_readiness_uses_latest_suffixed_symbol_endurance_run(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {}, {"GBPUSD": {"enabled_state": "enabled", "enabled": True}})
    _write_validation_artifacts(settings, "20260401T000000Z", {"GBPUSD": "enabled"})
    assert validate_strategy_profiles(settings) == 0
    older = settings.data.runs_dir / "20260401T010000Z_enabled_symbols_soak"
    older.mkdir(parents=True, exist_ok=True)
    older_payload = {
        "symbols": {
            "GBPUSD": {
                "decision": "blocked",
                "cycles_completed": 5,
                "alerts_by_severity": {"info": 0, "warning": 0, "error": 0, "critical": 0},
                "cycle_metrics": [],
                "expectancy_degradation_pct": 0.0,
                "profit_factor_degradation_pct": 0.0,
            }
        }
    }
    (older / "symbol_stability_report.json").write_text(json.dumps(wrap_artifact("symbol_stability", older_payload)), encoding="utf-8")
    _write_stability_artifact(settings, "20260401T020000Z", "GBPUSD", decision="go")
    newer = settings.data.runs_dir / "20260401T020000Z_symbol_endurance_01"
    base = settings.data.runs_dir / "20260401T020000Z_symbol_endurance"
    if base.exists() and not newer.exists():
        base.rename(newer)
    _write_lifecycle_artifact(settings, "20260401T030000Z", "GBPUSD", critical=0)
    settings = replace(settings, governance=replace(settings.governance, target_symbol="GBPUSD"))
    assert review_approved_demo_readiness(settings) == 0
    latest = sorted(settings.data.runs_dir.glob("*_review_approved_demo_readiness"))[-1]
    payload = read_artifact_payload(latest / "approved_demo_readiness_report.json", expected_type="strategy_profile_promotion")
    assert payload["symbols"]["GBPUSD"]["final_decision"] == "APPROVED_DEMO"
    assert payload["symbols"]["GBPUSD"]["endurance_summary"]["source_run"].endswith("_symbol_endurance_01")


def test_promote_strategy_profile_reverts_to_blocked_on_lifecycle_inconsistency(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "enabled", "enabled": True}})
    _write_validation_artifacts(settings, "20260401T000000Z", {"EURUSD": "enabled"})
    assert validate_strategy_profiles(settings) == 0
    _write_stability_artifact(settings, "20260401T010000Z", "EURUSD", decision="go")
    _write_lifecycle_artifact(settings, "20260401T020000Z", "EURUSD", critical=1)
    settings = replace(settings, governance=replace(settings.governance, target_symbol="EURUSD"))
    assert promote_strategy_profile(settings) == 2
    registry = load_strategy_profile_registry(settings)
    latest = registry["profiles"]["EURUSD"][-1]
    assert latest["promotion_state"] == "blocked"
    assert latest["profile_payload"]["promotion_state"] == "blocked"


def test_review_approved_demo_readiness_never_promotes_usdjpy(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {}, {"USDJPY": {"enabled_state": "disabled", "enabled": False}})
    _write_validation_artifacts(settings, "20260401T000000Z", {"USDJPY": "disabled"})
    assert validate_strategy_profiles(settings) == 0
    _write_stability_artifact(settings, "20260401T010000Z", "USDJPY", decision="go")
    _write_lifecycle_artifact(settings, "20260401T020000Z", "USDJPY", critical=0)
    settings = replace(settings, governance=replace(settings.governance, target_symbol="USDJPY"))
    assert review_approved_demo_readiness(settings) == 2
    latest = sorted(settings.data.runs_dir.glob("*_review_approved_demo_readiness"))[-1]
    payload = read_artifact_payload(latest / "approved_demo_readiness_report.json", expected_type="strategy_profile_promotion")
    assert payload["symbols"]["USDJPY"]["final_decision"] == "REVERT_TO_BLOCKED"
    assert "symbol_out_of_scope_for_promotion" in payload["symbols"]["USDJPY"]["reasons"]


def test_review_approved_demo_readiness_artifact_includes_policy_provenance(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "enabled", "enabled": True}})
    _write_validation_artifacts(settings, "20260401T000000Z", {"EURUSD": "enabled"})
    assert validate_strategy_profiles(settings) == 0
    _write_stability_artifact(settings, "20260401T010000Z", "EURUSD", decision="go")
    _write_lifecycle_artifact(settings, "20260401T020000Z", "EURUSD", critical=0)
    settings = replace(settings, governance=replace(settings.governance, target_symbol="EURUSD"))
    assert review_approved_demo_readiness(settings) == 0
    latest = sorted(settings.data.runs_dir.glob("*_review_approved_demo_readiness"))[-1]
    raw = json.loads((latest / "approved_demo_readiness_report.json").read_text(encoding="utf-8"))
    assert raw["provenance"]["correlation_keys"]["command"] == "review_approved_demo_readiness"
    assert raw["provenance"]["policy_version"] == "governance_policy.v1"
    assert raw["provenance"]["references"]["registry_path"].endswith("strategy_profile_registry.json")


def test_checksum_mismatch_blocks_active_profile_resolution(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "enabled", "enabled": True}})
    _write_validation_artifacts(settings, "20260401T000000Z", {"EURUSD": "enabled"})
    assert validate_strategy_profiles(settings) == 0
    _write_stability_artifact(settings, "20260401T010000Z", "EURUSD", decision="go")
    _write_lifecycle_artifact(settings, "20260401T020000Z", "EURUSD", critical=0)
    settings = replace(settings, governance=replace(settings.governance, target_symbol="EURUSD"))
    assert promote_strategy_profile(settings) == 0
    registry = load_strategy_profile_registry(settings)
    active_id = registry["active_profiles"]["EURUSD"]
    entry = next(item for item in registry["profiles"]["EURUSD"] if item["profile_id"] == active_id)
    entry["profile_payload"]["threshold"] = 0.99
    from iris_bot.governance import save_strategy_profile_registry

    save_strategy_profile_registry(settings, registry)
    status = resolve_active_profile_entry(settings, "EURUSD")
    assert status["ok"] is False
    assert "registry_profile_checksum_mismatch" in status["reasons"]


def test_rollback_clears_and_restores_active_profile_safely(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "enabled", "enabled": True}})
    _write_validation_artifacts(settings, "20260401T000000Z", {"EURUSD": "enabled"})
    assert validate_strategy_profiles(settings) == 0
    registry = load_strategy_profile_registry(settings)
    first = registry["profiles"]["EURUSD"][0]
    older = dict(first)
    older["profile_id"] = "EURUSD-rollback-target"
    older["promotion_state"] = "validated"
    older["profile_payload"] = dict(older["profile_payload"], profile_id="EURUSD-rollback-target", promotion_state="validated")
    older["checksum"] = older["checksum"]
    registry["profiles"]["EURUSD"].append(older)
    from iris_bot.governance import save_strategy_profile_registry

    save_strategy_profile_registry(settings, registry)
    _write_stability_artifact(settings, "20260401T010000Z", "EURUSD", decision="go")
    _write_lifecycle_artifact(settings, "20260401T020000Z", "EURUSD", critical=0)
    settings = replace(settings, governance=replace(settings.governance, target_symbol="EURUSD"))
    assert promote_strategy_profile(settings) == 0
    registry = load_strategy_profile_registry(settings)
    active_id = registry["active_profiles"]["EURUSD"]
    current = next(item for item in registry["profiles"]["EURUSD"] if item["profile_id"] == active_id)
    current["rollback_target"] = "EURUSD-rollback-target"
    current["profile_payload"]["rollback_target"] = "EURUSD-rollback-target"
    save_strategy_profile_registry(settings, registry)
    assert rollback_strategy_profile(settings) == 0
    status = resolve_active_profile_entry(settings, "EURUSD")
    assert status["ok"] is True
    assert status["active_profile_id"] == "EURUSD-rollback-target"


def test_load_paper_context_blocks_invalid_active_profile(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    settings = replace(settings, trading=replace(settings.trading, symbols=("EURUSD",)))
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "disabled", "enabled": False}})
    _write_validation_artifacts(settings, "20260401T000000Z", {"EURUSD": "disabled"})
    assert validate_strategy_profiles(settings) == 0

    class FakeReference:
        test_start_timestamp = "2026-01-01T00:00:00"
        test_end_timestamp = "2026-01-01T00:15:00"
        model_path = Path("/tmp/fake-model.json")
        feature_names = ("atr_5",)

    class FakeDataset:
        rows = [
            ProcessedRow(datetime(2026, 1, 1, 0, 0, 0), "EURUSD", "M15", 1.10, 1.11, 1.09, 1.105, 100.0, 1, "x", "2026-01-01T00:15:00", {"atr_5": 0.001}),
        ]

    monkeypatch.setattr("iris_bot.paper._locate_experiment_reference", lambda settings: FakeReference())
    monkeypatch.setattr("iris_bot.paper.load_processed_dataset", lambda *args, **kwargs: FakeDataset())
    try:
        load_paper_context(settings)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "Invalid active profiles for operation" in str(exc)


def test_load_paper_context_ignores_deliberately_blocked_symbol_without_active_profile(
    tmp_path: Path, monkeypatch
) -> None:
    settings = _settings(tmp_path, monkeypatch)
    settings = replace(settings, trading=replace(settings.trading, symbols=("EURUSD", "USDJPY")))
    registry = {
        "profiles": {
            "EURUSD": [{
                "profile_id": "EURUSD-approved",
                "symbol": "EURUSD",
                "created_at": "2026-04-01T00:00:00+00:00",
                "promotion_state": "approved_demo",
                "promotion_reason": "test",
                "checksum": "e3b0c44298fc1c149afbf4c8996fb924",
                "profile_payload": {
                    "enabled_state": "enabled",
                },
            }],
        },
        "active_profiles": {
            "EURUSD": "EURUSD-approved",
        },
    }
    from iris_bot.governance import save_strategy_profile_registry

    save_strategy_profile_registry(settings, registry)

    class FakeReference:
        test_start_timestamp = "2026-01-01T00:00:00"
        test_end_timestamp = "2026-01-01T00:15:00"
        model_path = Path("/tmp/fake-model.json")
        feature_names = ("atr_5",)

    class FakeDataset:
        rows = [
            ProcessedRow(datetime(2026, 1, 1, 0, 0, 0), "EURUSD", "M15", 1.10, 1.11, 1.09, 1.105, 100.0, 1, "x", "2026-01-01T00:15:00", {"atr_5": 0.001}),
        ]

    monkeypatch.setattr("iris_bot.paper._locate_experiment_reference", lambda settings: FakeReference())
    monkeypatch.setattr("iris_bot.paper.load_processed_dataset", lambda *args, **kwargs: FakeDataset())
    monkeypatch.setattr(
        "iris_bot.paper.resolve_active_profiles",
        lambda settings: (
            {"EURUSD": default_symbol_strategy_profile(settings, "EURUSD")},
            {
                "symbols": {
                    "EURUSD": {"ok": True, "reasons": []},
                    "USDJPY": {"ok": False, "reasons": ["no_active_profile_registered"]},
                },
            },
        ),
    )
    monkeypatch.setattr("iris_bot.paper.compute_signal_probabilities", lambda *args, **kwargs: [{1: 0.8, 0: 0.1, -1: 0.1}])
    monkeypatch.setattr("iris_bot.paper.XGBoostMultiClassModel.load", lambda self, path: None)

    reference, rows, probabilities = load_paper_context(settings)
    assert reference.model_path == Path("/tmp/fake-model.json")
    assert len(rows) == 1
    assert probabilities[0][1] == 0.8


def test_diagnose_profile_activation_writes_reports(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    write_symbol_strategy_profiles(settings, {}, {"EURUSD": {"enabled_state": "disabled", "enabled": False}})
    _write_validation_artifacts(settings, "20260401T000000Z", {"EURUSD": "disabled"})
    assert validate_strategy_profiles(settings) == 0
    code = diagnose_profile_activation(settings)
    assert code == 2
    latest = sorted(settings.data.runs_dir.glob("*_diagnose_profile_activation"))[-1]
    assert (latest / "profile_activation_diagnostic_report.json").exists()
    assert (latest / "active_profile_resolution_report.json").exists()
    assert (latest / "governance_consistency_report.json").exists()
    assert (latest / "technical_debt_avoidance_report.json").exists()
    assert (latest / "symbol_reactivation_readiness_report.json").exists()


def test_run_paper_engine_traces_active_profile_metadata() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [
        ProcessedRow(start, "EURUSD", "M15", 1.10, 1.11, 1.09, 1.105, 100.0, 1, "x", (start + timedelta(minutes=15)).isoformat(), {"atr_5": 0.001, "session_asia": 1.0, "session_london": 0.0, "session_new_york": 0.0}),
        ProcessedRow(start + timedelta(minutes=15), "EURUSD", "M15", 1.105, 1.115, 1.095, 1.112, 100.0, 1, "x", (start + timedelta(minutes=30)).isoformat(), {"atr_5": 0.001, "session_asia": 1.0, "session_london": 0.0, "session_new_york": 0.0}),
    ]
    probabilities = [{1: 0.9, 0: 0.05, -1: 0.05}, {1: 0.1, 0: 0.8, -1: 0.1}]
    from iris_bot.symbols import SymbolStrategyProfile
    from iris_bot.exits import SymbolExitProfile
    settings = load_settings()
    profile = SymbolStrategyProfile(
        symbol="EURUSD",
        enabled_state="enabled",
        allowed_timeframes=("M15",),
        allowed_sessions=("asia",),
        threshold=0.6,
        allow_long=True,
        allow_short=True,
        risk_multiplier=1.0,
        max_open_positions=1,
        stop_policy="static",
        target_policy="static",
        stop_atr_multiplier=1.5,
        target_atr_multiplier=3.0,
        stop_min_pct=0.001,
        stop_max_pct=0.01,
        target_min_pct=0.0015,
        target_max_pct=0.02,
        no_trade_min_expectancy_usd=0.0,
        profile_id="EURUSD-approved",
        model_variant="global_model",
        source_run_id="run123",
        promotion_state="approved_demo",
        promotion_reason="ok",
    )
    artifacts = run_paper_engine(
        PaperSessionConfig(
            mode="paper",
            threshold=0.6,
            trading_symbols=("EURUSD",),
            one_position_per_symbol=True,
            allow_long=True,
            allow_short=True,
            backtest=settings.backtest,
            risk=settings.risk,
            symbol_exit_profiles={"EURUSD": SymbolExitProfile("static", "static", 1.5, 3.0, None, None, None, None)},
            symbol_strategy_profiles={"EURUSD": profile},
            threshold_by_symbol={"EURUSD": 0.6},
        ),
        rows,
        probabilities,
    )
    generated = [row for row in artifacts.signal_rows if row["status"] == "generated"][0]
    assert generated["active_profile_id"] == "EURUSD-approved"
    assert generated["promotion_state"] == "approved_demo"


def test_reconcile_lifecycle_records_classifies_core_mismatches() -> None:
    local_intents = [{"symbol": "EURUSD", "signal_timestamp": "2026-01-01T00:00:00", "side": "buy", "volume_lots": 0.10}]
    broker_trace = {
        "orders": [{"ticket": 10, "symbol": "GBPUSD", "volume_initial": 0.10, "volume_current": 0.05}],
        "deals": [{"ticket": 100, "order": 10, "symbol": "GBPUSD", "volume": 0.10}],
        "positions": [],
    }
    payload = reconcile_lifecycle_records(local_intents, broker_trace)
    assert payload["critical_mismatch_count"] >= 1
    assert payload["mismatch_counts"]["local_intent_without_broker_evidence"] == 1
    assert payload["mismatch_counts"]["broker_event_without_local_intent"] == 1
    assert payload["mismatch_counts"]["partial_fill_real"] == 1


def test_reconcile_lifecycle_records_accepts_recent_closed_trade_evidence() -> None:
    local_intents = [
        {
            "symbol": "EURUSD",
            "signal_timestamp": "2026-01-01T00:00:00+00:00",
            "side": "buy",
            "volume_lots": 0.01,
            "local_closed_trade": True,
            "exit_timestamp": "2026-01-01T00:05:00+00:00",
            "exit_reason": "demo_live_probe_close",
        }
    ]
    broker_trace = {
        "orders": [{"ticket": 10, "symbol": "EURUSD", "volume_initial": 0.01, "volume_current": 0.01}],
        "deals": [{"ticket": 100, "order": 10, "symbol": "EURUSD", "volume": 0.01}],
        "positions": [],
    }
    payload = reconcile_lifecycle_records(local_intents, broker_trace)
    assert payload["critical_mismatch_count"] == 0
    assert payload["mismatch_counts"] == {}


def test_run_lifecycle_reconciliation_writes_reports(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    state = PaperEngineState(account_state=AccountState(1000.0, 1000.0, 1000.0))
    state.pending_intents = []
    atomic_write_json(
        settings.data.runtime_dir / settings.operational.persistence_state_filename,
        {"saved_at": "2026-01-01T00:00:00", "state": state.to_dict(), "latest_broker_sync_result": {}},
    )
    payload = {"connected": True, "orders": [], "deals": [], "positions": [], "scope_report": {"ownership_filter_active": True}}
    exit_code, run_dir = run_lifecycle_reconciliation(settings, client_factory=lambda: LifecycleFakeClient(payload))
    assert exit_code == 0
    assert (run_dir / "lifecycle_reconciliation_report.json").exists()


def test_run_lifecycle_reconciliation_does_not_promote_audit_visible_scope_only_records(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    state = PaperEngineState(account_state=AccountState(1000.0, 1000.0, 1000.0))
    state.pending_intents = []
    atomic_write_json(
        settings.data.runtime_dir / settings.operational.persistence_state_filename,
        {"saved_at": "2026-01-01T00:00:00", "state": state.to_dict(), "latest_broker_sync_result": {}},
    )
    payload = {
        "connected": True,
        "orders": [],
        "deals": [],
        "positions": [],
        "scope_report": {
            "ownership_filter_active": True,
            "ownership_mode": "strict",
            "ownership_policy_version": 1,
            "audit_visible_positions": [
                {
                    "symbol": "EURUSD",
                    "ticket": 999,
                    "owned_by_bot": False,
                    "in_symbol_scope": True,
                    "visible_for_audit": True,
                    "ownership_reason": "symbol_scope_only",
                }
            ],
            "audit_visible_orders": [],
            "audit_visible_deals": [],
        },
    }

    exit_code, run_dir = run_lifecycle_reconciliation(settings, client_factory=lambda: LifecycleFakeClient(payload))

    assert exit_code == 0
    report = read_artifact_payload(run_dir / "lifecycle_reconciliation_report.json", expected_type="lifecycle_reconciliation")
    assert report["critical_mismatch_count"] == 0
    assert report["scope_report"]["audit_visible_positions"][0]["ownership_reason"] == "symbol_scope_only"


def test_run_lifecycle_reconciliation_blocks_when_connect_fails(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path, monkeypatch)
    state = PaperEngineState(account_state=AccountState(1000.0, 1000.0, 1000.0))
    atomic_write_json(
        settings.data.runtime_dir / settings.operational.persistence_state_filename,
        {"saved_at": "2026-01-01T00:00:00", "state": state.to_dict(), "latest_broker_sync_result": {}},
    )
    payload = {"connected": False, "orders": [], "deals": [], "positions": [], "scope_report": {"ownership_filter_active": True}}

    exit_code, run_dir = run_lifecycle_reconciliation(
        settings,
        client_factory=lambda: LifecycleFakeClient(payload, connect_ok=False),
    )

    assert exit_code == 2
    assert not (run_dir / "lifecycle_reconciliation_report.json").exists()
