from __future__ import annotations

import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any
import json


def _fsync_directory(path: Path) -> None:
    if platform.system() == "Windows":
        # Windows does not support opening directories with os.open(..., O_RDONLY)
        # in the same way as POSIX for directory fsync.
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def durable_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def durable_write_text(path: Path, payload: str, *, encoding: str = "utf-8") -> None:
    durable_write_bytes(path, payload.encode(encoding))


def durable_write_json(path: Path, payload: dict[str, Any]) -> None:
    durable_write_text(path, json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def durable_copy_file(source_path: Path, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{dest_path.name}.", suffix=".tmp", dir=dest_path.parent)
    tmp_path = Path(tmp_name)
    try:
        with source_path.open("rb") as source_handle, os.fdopen(tmp_fd, "wb") as dest_handle:
            shutil.copyfileobj(source_handle, dest_handle)
            dest_handle.flush()
            os.fsync(dest_handle.fileno())
        os.replace(tmp_path, dest_path)
        _fsync_directory(dest_path.parent)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
