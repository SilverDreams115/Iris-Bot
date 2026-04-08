from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from iris_bot.config import load_settings
from iris_bot.windows_mt5_bridge import requires_windows_mt5_bridge, run_windows_mt5_bridge


def _settings(tmp_path: Path):
    settings = load_settings()
    return replace(
        settings,
        data=replace(settings.data, runs_dir=tmp_path / "runs"),
    )


def test_requires_windows_mt5_bridge_only_for_wsl_without_metatrader5(monkeypatch: pytest.MonkeyPatch) -> None:
    import iris_bot.windows_mt5_bridge as bridge

    monkeypatch.setattr(bridge, "is_wsl_runtime", lambda: True)
    monkeypatch.setattr(bridge, "metatrader5_importable", lambda: False)

    assert requires_windows_mt5_bridge("mt5-check") is True
    assert requires_windows_mt5_bridge("reconcile-lifecycle") is True
    assert requires_windows_mt5_bridge("run-paper") is False


def test_run_windows_mt5_bridge_invokes_backend_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iris_bot.windows_mt5_bridge as bridge

    settings = _settings(tmp_path)
    script_dir = tmp_path / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / "run_mt5_research_windows.sh"
    script_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    object.__setattr__(settings, "project_root", tmp_path)

    calls: list[tuple[list[str], Path]] = []

    def _fake_run(cmd: list[str], cwd: Path, check: bool) -> SimpleNamespace:
        calls.append((cmd, cwd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(bridge.subprocess, "run", _fake_run)

    logger = SimpleNamespace(info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    exit_code = run_windows_mt5_bridge(settings, "mt5-check", logger)

    assert exit_code == 0
    assert calls == [(["bash", str(script_path), "mt5-check"], tmp_path)]
