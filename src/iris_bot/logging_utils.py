from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from iris_bot.durable_io import durable_write_json

class _JsonLineFormatter(logging.Formatter):
    """Emit one JSON object per line (JSON Lines format).

    Each record contains: timestamp (ISO-8601), level, logger, message, and
    any exc_info as 'exception' when present.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


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


def configure_logging(run_dir: Path, level: str, log_format: str = "text") -> logging.Logger:
    logger = logging.getLogger("iris_bot")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    text_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(text_formatter)

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(text_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    if log_format == "json":
        jsonl_handler = logging.FileHandler(run_dir / "run.jsonl", encoding="utf-8")
        jsonl_handler.setFormatter(_JsonLineFormatter())
        logger.addHandler(jsonl_handler)

    logger.propagate = False
    return logger


def write_json_report(run_dir: Path, filename: str, payload: dict[str, object]) -> Path:
    path = run_dir / filename
    durable_write_json(path, payload)
    return path
