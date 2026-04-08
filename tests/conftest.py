from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _mock_official_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent _check_official_suite from spawning a real subprocess pytest run in every test.

    Tests that specifically exercise _check_official_suite behavior mock it themselves via
    their own monkeypatch call, which takes precedence over this autouse fixture.
    """
    monkeypatch.setattr(
        "iris_bot.demo_readiness._check_official_suite",
        lambda _settings: {
            "ok": True,
            "reason": "ok",
            "official_source": "test_fixture_autouse",
            "commands": {},
            "failed_commands": [],
        },
    )
