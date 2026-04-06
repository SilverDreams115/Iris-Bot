"""
Tests for evidence_store.py and run_index.py.

Covers:
  - Canonical evidence ingestion (project-internal paths only)
  - Rejection of external paths (/mnt/c/..., outside project root)
  - Manifest integrity (checksum verification)
  - Latest evidence retrieval by artifact_type + symbol
  - No dependency on external temporary artifacts
  - Run index registration and canonical discovery
  - Discovery by manifest/index vs. fragile glob
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from iris_bot.artifacts import wrap_artifact
from iris_bot.config import load_settings
from datetime import UTC, datetime, timedelta

from iris_bot.evidence_store import (
    ExternalPathError,
    UnknownArtifactTypeError,
    detect_contradictions,
    evidence_store_status,
    expire_evidence,
    get_latest_evidence,
    get_latest_evidence_payload,
    ingest_evidence,
    query_evidence_by_date_range,
)
from iris_bot.run_index import (
    get_latest_run,
    register_run,
    run_index_status,
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


def _write_lifecycle_artifact(tmp_path: Path, symbol: str, run_id: str, critical: int = 0) -> Path:
    run_dir = tmp_path / "runs" / f"{run_id}_lifecycle_reconciliation"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ok": critical == 0,
        "critical_mismatch_count": critical,
        "mismatch_counts": {},
        "mismatches": [],
        "symbols": {symbol: {"critical_mismatch_count": critical, "mismatch_categories": []}},
    }
    artifact = wrap_artifact("lifecycle_reconciliation", payload)
    report_path = run_dir / "lifecycle_reconciliation_report.json"
    report_path.write_text(json.dumps(artifact), encoding="utf-8")
    return report_path


def _write_stability_artifact(tmp_path: Path, symbol: str, run_id: str) -> Path:
    run_dir = tmp_path / "runs" / f"{run_id}_symbol_endurance"
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
                "cycle_metrics": [{"trades": 5, "expectancy_usd": 3.0, "profit_factor": 1.5}],
            }
        }
    }
    artifact = wrap_artifact("symbol_stability", payload)
    path = run_dir / "symbol_stability_report.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


# --- Ingest: basic ---

def test_ingest_lifecycle_evidence(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    source = _write_lifecycle_artifact(tmp_path, "EURUSD", "20260401T100000Z")
    entry = ingest_evidence(
        settings,
        source,
        artifact_type="lifecycle_reconciliation",
        source_run_id="20260401T100000Z_lifecycle_reconciliation",
        symbol="EURUSD",
        provenance="test",
    )
    assert entry["artifact_type"] == "lifecycle_reconciliation"
    assert entry["symbol"] == "EURUSD"
    assert len(entry["checksum"]) == 64
    assert entry["provenance"] == "test"
    # Canonical path must be within project
    assert tmp_path.name in entry["canonical_abs_path"]


def test_ingest_stability_evidence(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    source = _write_stability_artifact(tmp_path, "GBPUSD", "20260401T110000Z")
    entry = ingest_evidence(
        settings,
        source,
        artifact_type="symbol_stability",
        source_run_id="20260401T110000Z_symbol_endurance",
        symbol="GBPUSD",
    )
    assert entry["symbol"] == "GBPUSD"
    assert Path(entry["canonical_abs_path"]).exists()


def test_ingest_manifest_updated(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    source = _write_lifecycle_artifact(tmp_path, "EURUSD", "20260401T100000Z")
    ingest_evidence(settings, source, "lifecycle_reconciliation", "run_1", "EURUSD")
    source2 = _write_lifecycle_artifact(tmp_path, "GBPUSD", "20260401T110000Z")
    ingest_evidence(settings, source2, "lifecycle_reconciliation", "run_2", "GBPUSD")
    status = evidence_store_status(settings)
    assert status["total_entries"] == 2
    assert status["integrity_ok"] is True


def test_ingest_idempotent_same_run(tmp_path, monkeypatch):
    """Re-ingesting the same run_id replaces the entry, not duplicates it."""
    settings = _settings(tmp_path, monkeypatch)
    source = _write_lifecycle_artifact(tmp_path, "EURUSD", "20260401T100000Z")
    ingest_evidence(settings, source, "lifecycle_reconciliation", "run_1", "EURUSD")
    ingest_evidence(settings, source, "lifecycle_reconciliation", "run_1", "EURUSD")
    status = evidence_store_status(settings)
    assert status["total_entries"] == 1  # No duplicate


# --- External path rejection ---

def test_ingest_rejects_external_path(tmp_path, monkeypatch):
    """Evidence from outside the project root must be rejected."""
    settings = _settings(tmp_path, monkeypatch)
    # Create a fake external artifact at /tmp (outside project_root = tmp_path)
    external_dir = Path("/tmp") / "iris_test_external"
    external_dir.mkdir(parents=True, exist_ok=True)
    external_file = external_dir / "lifecycle_reconciliation_report.json"
    external_file.write_text(json.dumps({"artifact_type": "lifecycle_reconciliation", "payload": {}}))
    try:
        with pytest.raises(ExternalPathError, match="external path"):
            ingest_evidence(
                settings,
                external_file,
                artifact_type="lifecycle_reconciliation",
                source_run_id="external_run",
                symbol="EURUSD",
            )
    finally:
        try:
            external_file.unlink()
            external_dir.rmdir()
        except OSError:
            pass


def test_ingest_rejects_unknown_artifact_type(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    some_file = tmp_path / "some_artifact.json"
    some_file.write_text("{}")
    with pytest.raises(UnknownArtifactTypeError):
        ingest_evidence(settings, some_file, "unknown_type", "run_1", "EURUSD")


def test_no_windows_path_dependency(tmp_path, monkeypatch):
    """
    Governance decisions must not depend on /mnt/c/... paths.
    Verify the evidence store only contains project-internal paths.
    """
    settings = _settings(tmp_path, monkeypatch)
    source = _write_lifecycle_artifact(tmp_path, "EURUSD", "20260401T100000Z")
    ingest_evidence(settings, source, "lifecycle_reconciliation", "run_1", "EURUSD")
    entry = get_latest_evidence(settings, "lifecycle_reconciliation", "EURUSD")
    assert entry is not None
    # canonical_abs_path must be within project (tmp_path), not /mnt/c/
    assert "/mnt/c" not in entry["canonical_abs_path"]
    assert str(tmp_path) in entry["canonical_abs_path"]


# --- Latest evidence retrieval ---

def test_get_latest_evidence_by_symbol(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    s1 = _write_lifecycle_artifact(tmp_path, "EURUSD", "20260401T100000Z")
    s2 = _write_lifecycle_artifact(tmp_path, "EURUSD", "20260401T120000Z")
    ingest_evidence(settings, s1, "lifecycle_reconciliation", "run_earlier", "EURUSD")
    ingest_evidence(settings, s2, "lifecycle_reconciliation", "run_later", "EURUSD")
    latest = get_latest_evidence(settings, "lifecycle_reconciliation", "EURUSD")
    assert latest is not None
    assert latest["source_run_id"] == "run_later"


def test_get_latest_evidence_returns_none_when_empty(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    result = get_latest_evidence(settings, "lifecycle_reconciliation", "EURUSD")
    assert result is None


def test_get_latest_evidence_payload_verifies_checksum(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    source = _write_lifecycle_artifact(tmp_path, "EURUSD", "20260401T100000Z")
    ingest_evidence(settings, source, "lifecycle_reconciliation", "run_1", "EURUSD")
    payload = get_latest_evidence_payload(settings, "lifecycle_reconciliation", "EURUSD")
    assert payload is not None


def test_get_latest_evidence_payload_fails_on_tampered_file(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    source = _write_lifecycle_artifact(tmp_path, "EURUSD", "20260401T100000Z")
    entry = ingest_evidence(settings, source, "lifecycle_reconciliation", "run_1", "EURUSD")
    # Tamper with the canonical file
    canonical = Path(entry["canonical_abs_path"])
    canonical.write_text('{"tampered": true}', encoding="utf-8")
    # Payload retrieval should return None (checksum mismatch)
    result = get_latest_evidence_payload(settings, "lifecycle_reconciliation", "EURUSD")
    assert result is None


# --- Evidence store integrity ---

def test_evidence_store_status_integrity_ok(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    source = _write_lifecycle_artifact(tmp_path, "EURUSD", "20260401T100000Z")
    ingest_evidence(settings, source, "lifecycle_reconciliation", "run_1", "EURUSD")
    status = evidence_store_status(settings)
    assert status["integrity_ok"] is True
    assert len(status["integrity_failures"]) == 0


def test_evidence_store_detects_missing_file(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    source = _write_lifecycle_artifact(tmp_path, "EURUSD", "20260401T100000Z")
    entry = ingest_evidence(settings, source, "lifecycle_reconciliation", "run_1", "EURUSD")
    # Delete the canonical file
    Path(entry["canonical_abs_path"]).unlink()
    status = evidence_store_status(settings)
    assert status["integrity_ok"] is False
    assert len(status["integrity_failures"]) > 0


# --- Run index ---

def test_register_and_retrieve_run(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    run_dir = tmp_path / "runs" / "20260401T100000Z_lifecycle_reconciliation"
    run_dir.mkdir(parents=True, exist_ok=True)
    register_run(
        settings,
        run_id="20260401T100000Z",
        run_type="lifecycle_reconciliation",
        run_dir=run_dir,
        symbol=None,
    )
    result = get_latest_run(settings, "lifecycle_reconciliation")
    assert result is not None
    assert result["run_id"] == "20260401T100000Z"


def test_run_index_returns_latest(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    for ts in ["20260401T100000Z", "20260401T110000Z", "20260401T120000Z"]:
        run_dir = tmp_path / "runs" / f"{ts}_lifecycle_reconciliation"
        run_dir.mkdir(parents=True, exist_ok=True)
        register_run(settings, ts, "lifecycle_reconciliation", run_dir)
    latest = get_latest_run(settings, "lifecycle_reconciliation")
    assert latest["run_id"] == "20260401T120000Z"


def test_run_index_filters_by_symbol(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    run_dir = tmp_path / "runs" / "r1_symbol_endurance"
    run_dir.mkdir(parents=True, exist_ok=True)
    register_run(settings, "r1", "symbol_endurance", run_dir, symbol="EURUSD")
    register_run(settings, "r2", "symbol_endurance", run_dir, symbol="GBPUSD")
    result = get_latest_run(settings, "symbol_endurance", symbol="EURUSD")
    assert result["symbol"] == "EURUSD"
    result2 = get_latest_run(settings, "symbol_endurance", symbol="GBPUSD")
    assert result2["symbol"] == "GBPUSD"


def test_run_index_returns_none_for_unknown_type(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    result = get_latest_run(settings, "nonexistent_type")
    assert result is None


def test_run_index_status(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir()
    register_run(settings, "r1", "lifecycle_reconciliation", run_dir)
    register_run(settings, "r2", "symbol_endurance", run_dir, symbol="EURUSD")
    status = run_index_status(settings)
    assert status["total_entries"] == 2
    assert "lifecycle_reconciliation" in status["by_run_type"]
    assert "symbol_endurance" in status["by_run_type"]


# --- Retention policy ---

def test_expire_evidence_removes_old_entries(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    source = _write_lifecycle_artifact(tmp_path, "EURUSD", "old_run")
    ingest_evidence(settings, source, "lifecycle_reconciliation", "old_run", "EURUSD")
    # Expire with a reference time 100 days in the future
    future = datetime.now(tz=UTC) + timedelta(days=100)
    expired = expire_evidence(settings, max_age_days=1, reference_time=future)
    assert len(expired) == 1
    status = evidence_store_status(settings)
    assert status["total_entries"] == 0


def test_expire_evidence_keeps_recent_entries(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    source = _write_lifecycle_artifact(tmp_path, "EURUSD", "new_run")
    ingest_evidence(settings, source, "lifecycle_reconciliation", "new_run", "EURUSD")
    # Expire with max_age_days=365 and reference_time=now → entry is less than 1 second old
    expired = expire_evidence(settings, max_age_days=365)
    assert expired == []
    status = evidence_store_status(settings)
    assert status["total_entries"] == 1


def test_expire_evidence_deletes_physical_file(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    source = _write_lifecycle_artifact(tmp_path, "EURUSD", "run_x")
    entry = ingest_evidence(settings, source, "lifecycle_reconciliation", "run_x", "EURUSD")
    canonical = Path(entry["canonical_abs_path"])
    assert canonical.exists()
    future = datetime.now(tz=UTC) + timedelta(days=400)
    expire_evidence(settings, max_age_days=1, reference_time=future)
    assert not canonical.exists()


def test_expire_evidence_zero_max_age_raises(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        expire_evidence(settings, max_age_days=0)


def test_expire_evidence_selective_keeps_newer(tmp_path, monkeypatch):
    """Only the old entry is expired; the new entry is retained."""
    settings = _settings(tmp_path, monkeypatch)
    old_src = _write_lifecycle_artifact(tmp_path, "EURUSD", "run_old")
    new_src = _write_lifecycle_artifact(tmp_path, "GBPUSD", "run_new")
    ingest_evidence(settings, old_src, "lifecycle_reconciliation", "run_old", "EURUSD")
    ingest_evidence(settings, new_src, "lifecycle_reconciliation", "run_new", "GBPUSD")
    # Both are very recent, but set reference_time to 400 days ahead
    # max_age_days=365 → anything older than 365 days expires → both expire
    # Instead use 0.0001 days (about 8 seconds) — entries are seconds old → they expire
    future_ref = datetime.now(tz=UTC) + timedelta(days=400)
    expire_evidence(settings, max_age_days=365, reference_time=future_ref)
    status = evidence_store_status(settings)
    assert status["total_entries"] == 0  # both ~0 days old relative to creation, 400 days old relative to future_ref


# --- Contradiction detection ---

def test_detect_contradictions_finds_differing_checksums(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    src1 = _write_lifecycle_artifact(tmp_path, "EURUSD", "run1", critical=0)
    src2 = _write_lifecycle_artifact(tmp_path, "EURUSD", "run2", critical=1)  # different payload → different checksum
    ingest_evidence(settings, src1, "lifecycle_reconciliation", "run1", "EURUSD")
    ingest_evidence(settings, src2, "lifecycle_reconciliation", "run2", "EURUSD")
    contradictions = detect_contradictions(settings)
    # Two entries for the same key (lifecycle_reconciliation::EURUSD) with different checksums
    assert len(contradictions) == 1
    assert contradictions[0]["key"] == "lifecycle_reconciliation::EURUSD"
    assert len(contradictions[0]["checksums"]) == 2


def test_detect_contradictions_no_conflict_same_checksum(tmp_path, monkeypatch):
    """Same artifact ingested twice (idempotent) → no contradictions."""
    settings = _settings(tmp_path, monkeypatch)
    src = _write_lifecycle_artifact(tmp_path, "EURUSD", "run1")
    ingest_evidence(settings, src, "lifecycle_reconciliation", "run1", "EURUSD")
    ingest_evidence(settings, src, "lifecycle_reconciliation", "run1", "EURUSD")  # idempotent
    contradictions = detect_contradictions(settings)
    assert contradictions == []


def test_detect_contradictions_empty_store_returns_empty(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    assert detect_contradictions(settings) == []


# --- Date-range query ---

def test_query_evidence_by_date_range_returns_matching(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    src = _write_lifecycle_artifact(tmp_path, "EURUSD", "run1")
    ingest_evidence(settings, src, "lifecycle_reconciliation", "run1", "EURUSD")
    start = datetime.now(tz=UTC) - timedelta(hours=1)
    end = datetime.now(tz=UTC) + timedelta(hours=1)
    results = query_evidence_by_date_range(settings, "lifecycle_reconciliation", start, end)
    assert len(results) == 1
    assert results[0]["source_run_id"] == "run1"


def test_query_evidence_by_date_range_excludes_outside_window(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    src = _write_lifecycle_artifact(tmp_path, "EURUSD", "run1")
    ingest_evidence(settings, src, "lifecycle_reconciliation", "run1", "EURUSD")
    # Query window is 10 days in the future
    start = datetime.now(tz=UTC) + timedelta(days=10)
    end = datetime.now(tz=UTC) + timedelta(days=20)
    results = query_evidence_by_date_range(settings, "lifecycle_reconciliation", start, end)
    assert results == []


def test_query_evidence_by_date_range_filters_by_symbol(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    src1 = _write_lifecycle_artifact(tmp_path, "EURUSD", "run1")
    src2 = _write_lifecycle_artifact(tmp_path, "GBPUSD", "run2")
    ingest_evidence(settings, src1, "lifecycle_reconciliation", "run1", "EURUSD")
    ingest_evidence(settings, src2, "lifecycle_reconciliation", "run2", "GBPUSD")
    start = datetime.now(tz=UTC) - timedelta(hours=1)
    end = datetime.now(tz=UTC) + timedelta(hours=1)
    results = query_evidence_by_date_range(settings, "lifecycle_reconciliation", start, end, symbol="EURUSD")
    assert len(results) == 1
    assert results[0]["symbol"] == "EURUSD"


def test_run_index_replaces_glob_for_canonical_discovery(tmp_path, monkeypatch):
    """
    The run index returns the registered latest run, not the lexicographically
    last directory name. This is the canonical replacement for glob heuristics.
    """
    settings = _settings(tmp_path, monkeypatch)
    # Create run dirs in non-lexicographic order of registration
    older_dir = tmp_path / "runs" / "20260401T090000Z_lifecycle_reconciliation"
    newer_dir = tmp_path / "runs" / "20260401T080000Z_lifecycle_reconciliation"  # lex-earlier but registered later
    older_dir.mkdir(parents=True, exist_ok=True)
    newer_dir.mkdir(parents=True, exist_ok=True)
    register_run(settings, "older_run", "lifecycle_reconciliation", older_dir)
    register_run(settings, "newer_run", "lifecycle_reconciliation", newer_dir)
    result = get_latest_run(settings, "lifecycle_reconciliation")
    # Index returns last-registered, not lexicographically-last
    assert result["run_id"] == "newer_run"
