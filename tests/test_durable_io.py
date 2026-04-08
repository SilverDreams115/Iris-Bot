from __future__ import annotations

import json
from pathlib import Path

from iris_bot.durable_io import durable_write_json
from iris_bot.logging_utils import write_json_report
from iris_bot.operational import atomic_write_json


def test_durable_write_json_commits_payload_without_tmp_leftovers(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    durable_write_json(path, {"ok": True, "value": 3})

    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True, "value": 3}
    assert list(tmp_path.glob(".report.json.*.tmp")) == []


def test_write_json_report_uses_durable_write_json(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[Path, dict[str, object]]] = []

    def fake_durable_write_json(path: Path, payload: dict[str, object]) -> None:
        calls.append((path, payload))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    monkeypatch.setattr("iris_bot.logging_utils.durable_write_json", fake_durable_write_json)

    written = write_json_report(tmp_path, "x.json", {"result": "ok"})

    assert written == tmp_path / "x.json"
    assert calls == [(tmp_path / "x.json", {"result": "ok"})]


def test_atomic_write_json_uses_durable_write_json(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[Path, dict[str, object]]] = []

    def fake_durable_write_json(path: Path, payload: dict[str, object]) -> None:
        calls.append((path, payload))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    monkeypatch.setattr("iris_bot.operational.durable_write_json", fake_durable_write_json)

    atomic_write_json(tmp_path / "state.json", {"state": "ok"})

    assert calls == [(tmp_path / "state.json", {"state": "ok"})]
