from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from iris_bot.config import load_settings
from iris_bot.demo_live_checklist import generate_demo_live_checklist_report
from iris_bot.mt5 import MT5Client, MT5ValidationReport


class FakeChecklistClient(MT5Client):
    def __init__(self, *, connect_ok: bool = True, report_ok: bool = True) -> None:
        super().__init__(replace(load_settings().mt5, enabled=True), mt5_module=object())
        self._connect_ok = connect_ok
        self._report_ok = report_ok

    def connect(self) -> bool:
        return self._connect_ok

    def check(self, symbols: tuple[str, ...]) -> MT5ValidationReport:  # type: ignore[override]
        return MT5ValidationReport(
            ok=self._report_ok,
            connected=self._connect_ok,
            terminal_initialized=self._connect_ok,
            issues=[] if self._report_ok else [],
            symbols={symbol: {"ok": self._report_ok} for symbol in symbols},
        )

    def shutdown(self) -> None:
        return None


def _settings(tmp_path: Path):
    settings = load_settings()
    return replace(
        settings,
        mt5=replace(
            settings.mt5,
            enabled=True,
            login=123456,
            password="secret",
            server="MetaQuotes-Demo",
            path="C:/MT5/terminal64.exe",
        ),
        data=replace(settings.data, runs_dir=tmp_path / "runs"),
    )


def _write_probe_report(settings, ok: bool, reason: str = "ok") -> None:
    run_dir = settings.data.runs_dir / "20260405T000000Z_demo_live_probe"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "demo_live_probe_report.json").write_text(
        json.dumps({"ok": ok, "reason": reason}),
        encoding="utf-8",
    )


def test_demo_live_checklist_not_ready_without_probe(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    report = generate_demo_live_checklist_report(
        settings,
        client_factory=lambda: FakeChecklistClient(connect_ok=True, report_ok=True),
    )

    assert report.decision == "not_ready"
    assert "latest_demo_live_probe" in report.failed_checks


def test_demo_live_checklist_ready_with_connectivity_and_probe(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    _write_probe_report(settings, ok=True)
    monkeypatch.setattr(
        "iris_bot.demo_live_checklist.generate_demo_execution_readiness_report",
        lambda _settings: {"decision": "ready_for_next_phase"},
    )

    report = generate_demo_live_checklist_report(
        settings,
        client_factory=lambda: FakeChecklistClient(connect_ok=True, report_ok=True),
    )

    assert report.decision == "ready"
    assert report.all_required_ok is True
    assert report.failed_checks == []
