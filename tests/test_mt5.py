from collections import namedtuple
from dataclasses import replace

from iris_bot.config import MT5Config
from iris_bot.mt5 import MT5Client, OrderRequest, resolve_mt5_timeframe


FakeRate = namedtuple("FakeRate", ["time", "open", "high", "low", "close", "tick_volume"])
FakeTick = namedtuple("FakeTick", ["ask", "bid"])
FakeSymbolInfo = namedtuple(
    "FakeSymbolInfo",
    ["visible", "trade_mode", "volume_min", "volume_step", "volume_max", "filling_mode"],
)


class FakeMT5:
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_H1 = 60
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_FOK = 2
    ORDER_FILLING_RETURN = 4
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 10
    ORDER_TIME_GTC = 20

    def __init__(self, *, filling_mode: int = 1, volume_step: float = 0.01) -> None:
        self.initialized = False
        self.shutdown_called = False
        self.filling_mode = filling_mode
        self.volume_step = volume_step
        self.initialize_calls: list[dict[str, object]] = []
        self.login_calls = 0
        self.account_available = True
        self.terminal_available = True

    def initialize(self, **_: object) -> bool:
        self.initialize_calls.append(_)
        self.initialized = True
        return True

    def login(self, *_: object, **__: object) -> bool:
        self.login_calls += 1
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def terminal_info(self) -> dict[str, object]:
        if not self.terminal_available:
            return None  # type: ignore[return-value]
        return {"connected": True}

    def account_info(self):
        if not self.account_available:
            return None
        return namedtuple("AccountInfo", ["balance", "equity", "server"])(1000.0, 1000.0, "MetaQuotes-Demo")

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int) -> list[FakeRate]:
        assert symbol == "EURUSD"
        assert timeframe == self.TIMEFRAME_M5
        assert start_pos == 0
        assert count == 2
        return [
            FakeRate(1704067200, 1.1, 1.2, 1.0, 1.15, 100),
            FakeRate(1704067500, 1.15, 1.25, 1.1, 1.2, 110),
        ]

    def symbol_info(self, symbol: str) -> FakeSymbolInfo | None:
        if symbol == "INVALID":
            return None
        return FakeSymbolInfo(True, 1, 0.01, self.volume_step, 1.0, self.filling_mode)

    def symbol_select(self, symbol: str, enabled: bool) -> bool:
        return enabled and symbol != "HIDDEN"

    def symbol_info_tick(self, symbol: str) -> FakeTick | None:
        return None if symbol == "NO_TICK" else FakeTick(1.1002, 1.1000)


def test_resolve_mt5_timeframe() -> None:
    assert resolve_mt5_timeframe(FakeMT5(), "M5") == 5


def test_fetch_historical_bars_uses_mt5_module() -> None:
    client = MT5Client(replace(MT5Config(), enabled=True), mt5_module=FakeMT5())
    client.connect()

    bars = client.fetch_historical_bars("EURUSD", "M5", 2)

    assert len(bars) == 2
    assert bars[0].symbol == "EURUSD"
    assert bars[0].timeframe == "M5"
    assert bars[0].close == 1.15


def test_mt5_check_reports_symbol_validation() -> None:
    client = MT5Client(replace(MT5Config(), enabled=True), mt5_module=FakeMT5())
    client.connect()

    report = client.check(("EURUSD", "INVALID"))

    assert report.connected is True
    assert report.ok is False
    assert any(issue.code == "symbol_missing" for issue in report.issues)


def test_connect_prefers_authenticated_initialize_without_separate_login() -> None:
    config = replace(
        MT5Config(),
        enabled=True,
        login=123456,
        password="secret",
        server="MetaQuotes-Demo",
        path="C:/Program Files/MetaTrader 5/terminal64.exe",
    )
    mt5 = FakeMT5()

    client = MT5Client(config, mt5_module=mt5)

    assert client.connect() is True
    assert mt5.initialize_calls[0]["login"] == 123456
    assert mt5.initialize_calls[0]["password"] == "secret"
    assert mt5.initialize_calls[0]["server"] == "MetaQuotes-Demo"
    assert mt5.login_calls == 0


def test_connect_requires_usable_session_health() -> None:
    mt5 = FakeMT5()
    mt5.account_available = False
    client = MT5Client(
        replace(MT5Config(), enabled=True, login=123456, password="secret", server="MetaQuotes-Demo"),
        mt5_module=mt5,
    )

    assert client.connect() is False
    assert client.session_health() is not None
    assert client.session_health().account_accessible is False


def test_ensure_connection_recovers_from_terminal_drop() -> None:
    mt5 = FakeMT5()
    client = MT5Client(replace(MT5Config(), enabled=True), mt5_module=mt5)
    assert client.connect() is True

    mt5.terminal_available = False
    assert client.health_check().ok is False

    mt5.terminal_available = True
    assert client.ensure_connection(force_reconnect=True) is True


def test_dry_run_builds_request_when_valid() -> None:
    client = MT5Client(replace(MT5Config(), enabled=True), mt5_module=FakeMT5())
    client.connect()

    result = client.dry_run_market_order(
        OrderRequest(
            symbol="EURUSD",
            side="buy",
            volume=0.10,
            stop_loss=1.0950,
            take_profit=1.1100,
        )
    )

    assert result.accepted is True
    assert result.normalized_volume == 0.10
    assert result.request is not None
    assert result.request["type_filling"] == FakeMT5.ORDER_FILLING_IOC


def test_dry_run_rejects_volume_not_aligned_with_step() -> None:
    client = MT5Client(
        replace(MT5Config(), enabled=True),
        mt5_module=FakeMT5(volume_step=0.05),
    )
    client.connect()

    result = client.dry_run_market_order(
        OrderRequest(
            symbol="EURUSD",
            side="buy",
            volume=0.12,
            stop_loss=1.0950,
            take_profit=1.1100,
        )
    )

    assert result.accepted is False
    assert result.reason == "volume_invalid"


def test_dry_run_rejects_unsupported_filling_mode() -> None:
    client = MT5Client(
        replace(MT5Config(), enabled=True),
        mt5_module=FakeMT5(filling_mode=8),
    )
    client.connect()

    result = client.dry_run_market_order(
        OrderRequest(
            symbol="EURUSD",
            side="sell",
            volume=0.10,
            stop_loss=1.1050,
            take_profit=1.0900,
        )
    )

    assert result.accepted is False
    assert result.reason == "symbol_validation_failed"
    assert any(item.code == "filling_mode_invalid" for item in result.validations)
