import json
import logging

from iris_bot.logging_utils import configure_logging


def test_text_format_writes_log_file(tmp_path) -> None:
    logger = configure_logging(tmp_path, "INFO", log_format="text")
    logger.info("hello text")
    for h in logger.handlers:
        h.flush()

    log_file = tmp_path / "run.log"
    assert log_file.exists()
    assert "hello text" in log_file.read_text(encoding="utf-8")
    assert not (tmp_path / "run.jsonl").exists()


def test_json_format_writes_jsonl_file(tmp_path) -> None:
    logger = configure_logging(tmp_path, "DEBUG", log_format="json")
    logger.warning("hello json", )
    for h in logger.handlers:
        h.flush()

    jsonl_file = tmp_path / "run.jsonl"
    assert jsonl_file.exists()
    lines = [l for l in jsonl_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) >= 1
    record = json.loads(lines[0])
    assert record["level"] == "WARNING"
    assert record["message"] == "hello json"
    assert "timestamp" in record


def test_json_format_also_writes_text_log(tmp_path) -> None:
    logger = configure_logging(tmp_path, "INFO", log_format="json")
    logger.info("dual write")
    for h in logger.handlers:
        h.flush()

    assert (tmp_path / "run.log").exists()
    assert (tmp_path / "run.jsonl").exists()


def test_default_format_is_text(tmp_path) -> None:
    logger = configure_logging(tmp_path, "INFO")
    logger.info("default")
    for h in logger.handlers:
        h.flush()

    assert (tmp_path / "run.log").exists()
    assert not (tmp_path / "run.jsonl").exists()
