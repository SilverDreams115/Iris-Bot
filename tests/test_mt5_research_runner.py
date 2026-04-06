from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _scoped_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    runtime_dir = tmp_path / "data" / "runtime"
    runs_dir = tmp_path / "runs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    from iris_bot.config import load_settings

    settings = load_settings()
    object.__setattr__(settings.data, "raw_dir", raw_dir)
    object.__setattr__(settings.data, "processed_dir", processed_dir)
    object.__setattr__(settings.data, "runtime_dir", runtime_dir)
    object.__setattr__(settings.data, "runs_dir", runs_dir)
    object.__setattr__(settings.experiment, "_processed_dir", processed_dir)
    return settings


class _FakeClient:
    def __init__(self, _config, *, connect_ok: bool = True, terminal_ok: bool = True, account_ok: bool = True) -> None:
        self._connect_ok = connect_ok
        self._terminal_ok = terminal_ok
        self._account_ok = account_ok
        self._mt5 = SimpleNamespace(terminal_info=lambda: {"path": r"C:\Program Files\MetaTrader 5\terminal64.exe"})

    def connect(self) -> bool:
        return self._connect_ok

    def session_health(self):
        return SimpleNamespace(terminal_available=self._terminal_ok, account_accessible=self._account_ok)

    def last_error(self):
        return (1, "Success") if self._connect_ok else (2, "Connect failed")

    def account_info(self):
        return {"server": "MetaQuotes-Demo"}

    def shutdown(self) -> None:
        return None


def test_runtime_preflight_fails_clean_when_mt5_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import iris_bot.mt5_research_runtime as runtime

    settings = _scoped_settings(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime, "load_settings", lambda: settings)
    monkeypatch.setattr(runtime, "_runtime_probe", lambda: runtime.RuntimeProbe(
        platform_system="Windows",
        platform_release="11",
        python_executable="python.exe",
        python_version="3.11",
        project_root=str(settings.project_root),
        runs_dir=str(settings.data.runs_dir),
        raw_dir=str(settings.data.raw_dir),
        runtime_provenance={},
        metatrader5_importable=False,
        metatrader5_module_path=None,
        xgboost_importable=True,
        mt5_env_sources={},
    ))
    monkeypatch.setattr(runtime, "MT5Client", lambda cfg: _FakeClient(cfg, connect_ok=False))

    result = runtime._preflight("fetch-extended-history")

    assert result.ok is False
    assert "metatrader5_not_importable" in result.issues
    assert "mt5_connect_failed" in result.issues


def test_runtime_preflight_fails_clean_when_terminal_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import iris_bot.mt5_research_runtime as runtime

    settings = _scoped_settings(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime, "load_settings", lambda: settings)
    monkeypatch.setattr(runtime, "MT5Client", lambda cfg: _FakeClient(cfg, connect_ok=True, terminal_ok=False, account_ok=True))

    result = runtime._preflight("fetch-extended-history")

    assert result.ok is False
    assert "mt5_terminal_unavailable" in result.issues


def test_validate_runtime_only_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import iris_bot.mt5_research_runtime as runtime

    settings = _scoped_settings(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime, "load_settings", lambda: settings)
    monkeypatch.setattr(runtime, "_runtime_probe", lambda: runtime.RuntimeProbe(
        platform_system="Windows",
        platform_release="11",
        python_executable="python.exe",
        python_version="3.11",
        project_root=str(settings.project_root),
        runs_dir=str(settings.data.runs_dir),
        raw_dir=str(settings.data.raw_dir),
        runtime_provenance={"host_runtime": "test"},
        metatrader5_importable=True,
        metatrader5_module_path="C:/site-packages/MetaTrader5/__init__.py",
        xgboost_importable=True,
        mt5_env_sources={},
    ))
    monkeypatch.setattr(runtime, "MT5Client", lambda cfg: _FakeClient(cfg, connect_ok=True, terminal_ok=True, account_ok=True))

    assert runtime.validate_runtime_only("20260406T000000Z") == 0
    report_path = settings.data.runs_dir / "20260406T000000Z_mt5_research_execution" / "mt5_research_preflight_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "mt5_research_preflight_report"


def test_shell_wrapper_delegates_to_windows_runtime_without_quant_duplication() -> None:
    path = Path("scripts/run_mt5_research_windows.sh")
    content = path.read_text(encoding="utf-8")

    assert "python -m iris_bot.mt5_research_runtime" in content
    assert "cp -r \"$PROJECT_ROOT/src\"" in content
    assert "shutil.copytree" in content
    assert "fetch-extended-history" in content
