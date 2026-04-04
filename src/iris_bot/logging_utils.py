from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path


def build_run_directory(runs_dir: Path, command_name: str) -> Path:
    run_id = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    base_dir = runs_dir / f"{run_id}_{command_name}"
    run_dir = base_dir
    suffix = 1
    while run_dir.exists():
        run_dir = runs_dir / f"{run_id}_{command_name}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def configure_logging(run_dir: Path, level: str) -> logging.Logger:
    logger = logging.getLogger("iris_bot")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def write_json_report(run_dir: Path, filename: str, payload: dict[str, object]) -> Path:
    path = run_dir / filename
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
