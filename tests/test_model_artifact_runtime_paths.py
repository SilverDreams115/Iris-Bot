from __future__ import annotations

from pathlib import Path

from iris_bot.config import load_settings
from iris_bot.model_artifacts import _resolve_runtime_project_path


def test_resolve_runtime_project_path_remaps_wsl_absolute_project_path(tmp_path: Path) -> None:
    settings = load_settings()
    object.__setattr__(settings, "project_root", tmp_path / "IRIS-Bot")
    runtime_path = settings.project_root / "data" / "runtime" / "demo_execution_models" / "EURUSD" / "model.json"
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text("{}", encoding="utf-8")

    resolved = _resolve_runtime_project_path(
        settings,
        "/home/silver/projects/IRIS-Bot/data/runtime/demo_execution_models/EURUSD/model.json",
    )

    assert resolved == runtime_path
