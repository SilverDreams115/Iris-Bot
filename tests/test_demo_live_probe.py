from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from iris_bot.config import load_settings
from iris_bot.demo_live_probe import run_demo_live_probe
from iris_bot.mt5 import BrokerSnapshot, MT5Client, OrderResult
from iris_bot.resilient import build_runtime_state_path, restore_runtime_state


class _FakeTick:
    ask = 1.1002
    bid = 1.1000


class _FakeMT5Module:
    def symbol_info_tick(self, symbol: str) -> _FakeTick | None:
        return _FakeTick() if symbol == "EURUSD" else None

    def shutdown(self) -> None:
        return None


class FakeProbeClient(MT5Client):
    def __init__(self, *, demo: bool = True, connect_ok: bool = True) -> None:
        super().__init__(replace(load_settings().mt5, enabled=True), mt5_module=_FakeMT5Module())
        self._connected = connect_ok
        self._demo = demo
        self._positions: list[dict[str, object]] = []

    def connect(self) -> bool:
        return self._connected

    def account_info(self) -> dict[str, object] | None:
        return {
            "server": "MetaQuotes-Demo" if self._demo else "RealServer",
            "company": "MetaQuotes",
            "name": "Demo Account" if self._demo else "Primary Account",
        }

    def send_market_order(self, order):  # type: ignore[override]
        self._positions = [{"ticket": 123456, "symbol": order.symbol, "volume": order.volume, "type": 0}]
        return OrderResult(True, 10009, "done", 123456, order.volume, order.price, {"symbol": order.symbol})

    def broker_state_snapshot(self, symbols: tuple[str, ...]) -> BrokerSnapshot:
        positions = [item for item in self._positions if item["symbol"] in symbols]
        return BrokerSnapshot(True, self.account_info() or {}, positions, [], [], {})

    def close_position(self, ticket: int, symbol: str, volume: float, side: str) -> OrderResult:
        self._positions = []
        return OrderResult(True, 10009, "done", ticket, volume, 1.1000, {"symbol": symbol, "side": side})


def _settings(tmp_path: Path):
    settings = load_settings()
    return replace(
        settings,
        mt5=replace(settings.mt5, enabled=True),
        data=replace(settings.data, runs_dir=tmp_path / "runs", runtime_dir=tmp_path / "runtime"),
        trading=replace(settings.trading, symbols=("EURUSD",)),
    )


def test_demo_live_probe_succeeds_and_writes_report(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    code, run_dir, report = run_demo_live_probe(settings, client_factory=lambda: FakeProbeClient())

    assert code == 0
    assert report.ok is True
    assert report.status == "completed"
    assert report.opened_position is not None
    assert report.close_order_result is not None
    assert (run_dir / "demo_live_probe_report.json").exists()
    assert (run_dir / "demo_live_probe_runtime_evidence.json").exists()
    restored, restore_report = restore_runtime_state(build_runtime_state_path(settings), require_clean=False)
    assert restore_report.ok is True
    assert restored is not None
    assert len(restored.closed_positions) == 1
    assert restored.closed_positions[0].symbol == "EURUSD"
    assert restored.closed_positions[0].exit_reason == "demo_live_probe_close"
    assert restored.current_session_status.mode == "demo_live_probe"


def test_demo_live_probe_blocks_non_demo_accounts(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    code, run_dir, report = run_demo_live_probe(settings, client_factory=lambda: FakeProbeClient(demo=False))

    assert code == 3
    assert report.ok is False
    assert report.reason == "account_not_demo"
    assert (run_dir / "demo_live_probe_report.json").exists()
