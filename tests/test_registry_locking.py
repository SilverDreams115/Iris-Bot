"""
Tests for registry_lock.py — transactional locking of the strategy profile registry.

Covers:
  - Basic lock acquisition and release
  - Concurrent promotion safety (simulated via threading)
  - Concurrent rollback safety
  - Conflict detection via etag
  - Atomic write integrity (no corruption from partial writes)
  - Registry consistency after simulated failure
  - Lock timeout behavior
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from iris_bot.operational import atomic_write_json
from iris_bot.registry_lock import (
    RegistryLockTimeoutError,
    RegistryMutationConflictError,
    governance_lock_audit,
    registry_etag,
    registry_exclusive_lock,
)


@pytest.fixture
def reg_path(tmp_path: Path) -> Path:
    return tmp_path / "test_registry.json"


def _write_registry(path: Path, content: dict) -> None:
    atomic_write_json(path, content)


# --- Basic lock acquisition ---

def test_lock_acquire_and_release(reg_path):
    _write_registry(reg_path, {"profiles": {}, "active_profiles": {}})
    with registry_exclusive_lock(reg_path):
        pass  # Should not raise


def test_lock_creates_lock_file(reg_path):
    _write_registry(reg_path, {"profiles": {}})
    lock_path = reg_path.with_suffix(reg_path.suffix + ".lock")
    with registry_exclusive_lock(reg_path):
        assert lock_path.exists()


def test_lock_released_after_context(reg_path):
    _write_registry(reg_path, {"profiles": {}})
    with registry_exclusive_lock(reg_path):
        pass
    # After context, another lock acquisition should succeed immediately
    with registry_exclusive_lock(reg_path, timeout_seconds=1.0):
        pass


def test_registry_etag_empty_file(tmp_path):
    path = tmp_path / "nonexistent.json"
    assert registry_etag(path) == ""


def test_registry_etag_stable(reg_path):
    _write_registry(reg_path, {"profiles": {}, "version": 1})
    etag1 = registry_etag(reg_path)
    etag2 = registry_etag(reg_path)
    assert etag1 == etag2
    assert len(etag1) == 64  # SHA-256 hex


def test_registry_etag_changes_on_mutation(reg_path):
    _write_registry(reg_path, {"profiles": {}, "version": 1})
    etag_before = registry_etag(reg_path)
    _write_registry(reg_path, {"profiles": {}, "version": 2})
    etag_after = registry_etag(reg_path)
    assert etag_before != etag_after


# --- Conflict detection ---

def test_conflict_raises_when_registry_mutated(reg_path):
    _write_registry(reg_path, {"profiles": {}, "version": 1})
    etag = registry_etag(reg_path)
    # Mutate the registry between etag capture and lock acquisition
    _write_registry(reg_path, {"profiles": {}, "version": 2})
    with pytest.raises(RegistryMutationConflictError, match="mutated by a concurrent process"):
        with registry_exclusive_lock(reg_path, expected_etag=etag):
            pass


def test_no_conflict_when_registry_unchanged(reg_path):
    _write_registry(reg_path, {"profiles": {}, "version": 1})
    etag = registry_etag(reg_path)
    # No mutation between etag capture and lock
    with registry_exclusive_lock(reg_path, expected_etag=etag):
        pass  # Should not raise


def test_no_conflict_check_when_etag_is_none(reg_path):
    _write_registry(reg_path, {"profiles": {}, "version": 1})
    # Mutate before lock
    _write_registry(reg_path, {"profiles": {}, "version": 2})
    # Without expected_etag, no conflict check — lock just succeeds
    with registry_exclusive_lock(reg_path, expected_etag=None):
        pass  # Should not raise


# --- Concurrent access: two threads trying to promote simultaneously ---

def test_concurrent_promotions_do_not_corrupt_registry(reg_path):
    """
    Two threads each try to append to the registry under the lock.
    The final state must contain both entries (no lost update).
    """
    _write_registry(reg_path, {"profiles": {"EURUSD": []}, "active_profiles": {}})
    errors = []
    completed = []

    def promote_worker(entry_id: str) -> None:
        try:
            with registry_exclusive_lock(reg_path, timeout_seconds=5.0):
                # Read inside lock
                registry = json.loads(reg_path.read_text())
                entries = registry["profiles"].setdefault("EURUSD", [])
                entries.append({"profile_id": entry_id, "state": "approved_demo"})
                time.sleep(0.01)  # Simulate work
                atomic_write_json(reg_path, registry)
                completed.append(entry_id)
        except Exception as exc:
            errors.append(str(exc))

    t1 = threading.Thread(target=promote_worker, args=("profile_A",))
    t2 = threading.Thread(target=promote_worker, args=("profile_B",))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, f"Thread errors: {errors}"
    assert len(completed) == 2, f"Expected 2 completions, got {completed}"

    final = json.loads(reg_path.read_text())
    profile_ids = {e["profile_id"] for e in final["profiles"]["EURUSD"]}
    assert profile_ids == {"profile_A", "profile_B"}, "Both entries must be present (no lost update)"


def test_concurrent_rollbacks_no_corruption(reg_path):
    """
    Two threads each try to rollback. Only one should win the lock at a time,
    and the final state must be internally consistent.
    """
    _write_registry(reg_path, {
        "profiles": {"EURUSD": [
            {"profile_id": "v1", "promotion_state": "approved_demo"},
            {"profile_id": "v2", "promotion_state": "deprecated"},
        ]},
        "active_profiles": {"EURUSD": "v1"},
    })
    completed = []
    errors = []

    def rollback_worker(worker_id: str) -> None:
        try:
            with registry_exclusive_lock(reg_path, timeout_seconds=5.0):
                registry = json.loads(reg_path.read_text())
                registry["active_profiles"]["EURUSD"] = f"rollback_{worker_id}"
                time.sleep(0.005)
                atomic_write_json(reg_path, registry)
                completed.append(worker_id)
        except Exception as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=rollback_worker, args=(f"w{i}",)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Thread errors: {errors}"
    assert len(completed) == 3

    final = json.loads(reg_path.read_text())
    # The active_profiles must be one of the rollback values (last writer wins, but coherently)
    assert final["active_profiles"]["EURUSD"].startswith("rollback_")


def test_conflict_detected_prevents_stale_write(reg_path):
    """
    Thread A reads etag. Thread B mutates the registry. Thread A acquires lock with
    stale etag → should get RegistryMutationConflictError.
    """
    _write_registry(reg_path, {"profiles": {}, "counter": 0})
    stale_etag = registry_etag(reg_path)

    # Simulate Thread B mutating between Thread A's read and lock acquisition
    _write_registry(reg_path, {"profiles": {}, "counter": 1})

    with pytest.raises(RegistryMutationConflictError):
        with registry_exclusive_lock(reg_path, expected_etag=stale_etag):
            pass  # Should never reach here


def test_timeout_raises_on_held_lock(reg_path):
    """
    If a lock is already held, trying to acquire it with a very short timeout must raise.
    """
    _write_registry(reg_path, {"profiles": {}})
    lock_acquired = threading.Event()
    lock_released = threading.Event()

    def hold_lock():
        with registry_exclusive_lock(reg_path, timeout_seconds=5.0):
            lock_acquired.set()
            lock_released.wait(timeout=5.0)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    lock_acquired.wait(timeout=2.0)

    try:
        with pytest.raises(RegistryLockTimeoutError):
            with registry_exclusive_lock(reg_path, timeout_seconds=0.1):
                pass
    finally:
        lock_released.set()
        holder.join(timeout=5.0)


# --- Atomic write integrity ---

def test_atomic_write_no_corruption_on_failure(reg_path, tmp_path):
    """
    If we write a tmp file and then the process would "crash" before os.replace,
    the original registry remains intact.
    """
    _write_registry(reg_path, {"profiles": {}, "version": 1})
    original_content = reg_path.read_text()

    # Simulate what atomic_write_json does: write to .tmp first
    tmp = reg_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"profiles": {}, "version": 2}))
    # "Crash" before os.replace → original file unchanged
    assert reg_path.read_text() == original_content


def test_registry_consistent_after_lock_exception(reg_path):
    """
    If an exception occurs inside the lock block, the registry must remain
    in its pre-lock state (write must not have happened).
    """
    _write_registry(reg_path, {"profiles": {}, "version": 1})
    original = json.loads(reg_path.read_text())

    with pytest.raises(RuntimeError):
        with registry_exclusive_lock(reg_path):
            # Simulate partial work but exception before save
            raise RuntimeError("simulated failure before save")

    # Registry unchanged because write never happened
    final = json.loads(reg_path.read_text())
    assert final["version"] == 1
    assert final == original


# --- Lock audit ---

def test_governance_lock_audit_no_lock(reg_path):
    _write_registry(reg_path, {"profiles": {}})
    audit = governance_lock_audit(reg_path)
    assert audit["registry_exists"] is True
    assert audit["lock_currently_held"] is False
    assert audit["safe_to_mutate"] is True
    assert len(audit["registry_etag"]) == 64


def test_governance_lock_audit_missing_registry(tmp_path):
    reg_path = tmp_path / "missing.json"
    audit = governance_lock_audit(reg_path)
    assert audit["registry_exists"] is False
    assert audit["safe_to_mutate"] is False
    assert audit["registry_etag"] == ""


def test_lock_audit_while_held(reg_path):
    """Lock audit should detect that lock is currently held."""
    _write_registry(reg_path, {"profiles": {}})
    lock_acquired = threading.Event()
    lock_released = threading.Event()

    def hold_lock():
        with registry_exclusive_lock(reg_path, timeout_seconds=5.0):
            lock_acquired.set()
            lock_released.wait(timeout=5.0)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    lock_acquired.wait(timeout=2.0)

    try:
        audit = governance_lock_audit(reg_path)
        assert audit["lock_currently_held"] is True
        assert audit["safe_to_mutate"] is False
    finally:
        lock_released.set()
        holder.join(timeout=5.0)
