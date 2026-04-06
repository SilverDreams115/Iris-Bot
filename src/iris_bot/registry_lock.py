"""
Transactional locking for the strategy profile registry.

Guarantees:
  - Only one process can mutate the registry at a time (fcntl.flock, exclusive)
  - Callers pass a pre-lock etag; if the file changed between the caller's read
    and lock acquisition, RegistryMutationConflictError is raised and the
    operation is aborted cleanly — no silent lost update
  - Lock file is a separate .lock sidecar, never the registry itself
  - Timeout is enforced; RegistryLockTimeoutError on expiry

Usage pattern (all registry mutations must follow this):

    etag = registry_etag(reg_path)
    # ... do expensive read-only pre-computation outside lock ...
    with registry_exclusive_lock(reg_path, expected_etag=etag):
        registry = load_strategy_profile_registry(settings)  # re-read inside
        # apply mutation
        save_strategy_profile_registry(settings, registry)
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import time
from pathlib import Path
from typing import Generator

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class RegistryMutationConflictError(Exception):
    """Raised when a concurrent process mutated the registry before our lock was acquired."""


class RegistryLockTimeoutError(Exception):
    """Raised when the exclusive lock cannot be acquired within the configured timeout."""


def _try_lock_nonblocking(handle: object) -> bool:
    if os.name == "nt":
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _unlock(handle: object) -> None:
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def registry_etag(path: Path) -> str:
    """
    Returns a stable content hash of the registry file.

    Call BEFORE starting any pre-computation that will lead to a mutation.
    Pass the result as expected_etag to registry_exclusive_lock.
    Returns "" if the file does not exist (creation case is also protected).
    """
    if not path.exists():
        return ""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


@contextlib.contextmanager
def registry_exclusive_lock(
    registry_path: Path,
    timeout_seconds: float = 15.0,
    expected_etag: str | None = None,
) -> Generator[None, None, None]:
    """
    Acquires an exclusive write lock on {registry_path}.lock.

    Args:
        registry_path: Path to the registry JSON file being mutated.
        timeout_seconds: How long to spin before raising RegistryLockTimeoutError.
        expected_etag: If provided, the file content hash captured BEFORE the
            caller's pre-computation. After acquiring the lock, the current hash
            is verified. If different, RegistryMutationConflictError is raised
            and the lock is immediately released.

    Guarantees inside the `with` block:
        - No other process holds the lock on this registry
        - The registry file matches expected_etag (if provided)
        - Caller MUST re-read the registry inside the block to get authoritative state
    """
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    fh = lock_path.open("a+", encoding="utf-8")
    try:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                locked = _try_lock_nonblocking(fh)
                if not locked:
                    raise BlockingIOError
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RegistryLockTimeoutError(
                        f"Could not acquire exclusive lock on {lock_path} "
                        f"within {timeout_seconds}s. Another process may be holding it."
                    )
                time.sleep(min(0.05, remaining))

        # Lock acquired — write holder metadata for auditability and stale detection
        fh.seek(0)
        fh.truncate()
        fh.write(f"locked_at={time.time():.6f}\npid={os.getpid()}\n")
        fh.flush()

        # Conflict detection: verify registry has not been mutated since caller's read
        if expected_etag is not None:
            current_etag = registry_etag(registry_path)
            if current_etag != expected_etag:
                # Release before raising so other waiters can proceed
                _unlock(fh)
                raise RegistryMutationConflictError(
                    f"Registry {registry_path.name} was mutated by a concurrent process "
                    f"between pre-computation and lock acquisition. "
                    f"Pre-computation etag: {expected_etag[:16]!r}, "
                    f"current etag: {current_etag[:16]!r}. "
                    "Aborting to prevent lost update. Retry the operation."
                )
        try:
            yield
        finally:
            fh.seek(0)
            fh.truncate()
            fh.write(f"released_at={time.time():.6f}\n")
            fh.flush()
            _unlock(fh)
    finally:
        fh.close()


def governance_lock_audit(registry_path: Path) -> dict:
    """
    Returns audit information about the lock state for the registry.

    Checks:
        - Whether the lock file exists
        - Whether the lock is currently held (non-blocking trylock attempt)
        - Whether the registry file exists and has a valid checksum
    """
    lock_path = registry_path.with_suffix(registry_path.suffix + ".lock")
    lock_exists = lock_path.exists()
    lock_held = False
    lock_content = ""
    lock_pid: int | None = None
    lock_stale = False

    if lock_exists:
        try:
            lock_content = lock_path.read_text(encoding="utf-8").strip()
            if os.name == "nt" and lock_content.startswith("released_at=") and "pid=" not in lock_content:
                lock_held = False
                lock_pid = None
                lock_stale = False
                return {
                    "registry_path": str(registry_path),
                    "lock_path": str(lock_path),
                    "lock_file_exists": lock_exists,
                    "lock_currently_held": lock_held,
                    "lock_pid": lock_pid,
                    "lock_stale": lock_stale,
                    "lock_content": lock_content,
                    "registry_exists": registry_path.exists(),
                    "registry_etag": registry_etag(registry_path) if registry_path.exists() else "",
                    "safe_to_mutate": registry_path.exists() and not lock_held,
                }
            # Parse PID from lock file content
            for line in lock_content.splitlines():
                if line.startswith("pid="):
                    try:
                        lock_pid = int(line.split("=", 1)[1])
                    except ValueError:
                        pass
            fh = lock_path.open("r", encoding="utf-8")
            try:
                if _try_lock_nonblocking(fh):
                    _unlock(fh)
                    lock_held = False
                else:
                    lock_held = True
                    # Check if the holding process is still alive
                    if lock_pid is not None:
                        try:
                            os.kill(lock_pid, 0)
                        except ProcessLookupError:
                            lock_stale = True
                        except PermissionError:
                            pass  # Process exists but we can't signal it
            finally:
                fh.close()
        except OSError:
            pass

    registry_exists = registry_path.exists()
    registry_etag_value = registry_etag(registry_path) if registry_exists else ""

    return {
        "registry_path": str(registry_path),
        "lock_path": str(lock_path),
        "lock_file_exists": lock_exists,
        "lock_currently_held": lock_held,
        "lock_pid": lock_pid,
        "lock_stale": lock_stale,
        "lock_content": lock_content,
        "registry_exists": registry_exists,
        "registry_etag": registry_etag_value,
        "safe_to_mutate": registry_exists and not lock_held,
    }
