from collections import namedtuple
from dataclasses import replace

from iris_bot.config import MT5Config
from iris_bot.mt5 import MT5Client, resolve_mt5_timeframe


FakeRate = namedtuple("FakeRate", ["time", "open", "high", "low", "close", "tick_volume"])


class FakeMT5:
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_H1 = 60

    def __init__(self) -> None:
        self.initialized = False
        self.shutdown_called = False

    def initialize(self, **_: object) -> bool:
        self.initialized = True
        return True

    def login(self, *_: object, **__: object) -> bool:
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int) -> list[FakeRate]:
        assert symbol == "EURUSD"
        assert timeframe == self.TIMEFRAME_M5
        assert start_pos == 0
        assert count == 2
        return [
            FakeRate(1704067200, 1.1, 1.2, 1.0, 1.15, 100),
            FakeRate(1704067500, 1.15, 1.25, 1.1, 1.2, 110),
        ]


def test_resolve_mt5_timeframe() -> None:
    assert resolve_mt5_timeframe(FakeMT5(), "M5") == 5


def test_fetch_historical_bars_uses_mt5_module() -> None:
    client = MT5Client(replace(MT5Config(), enabled=True))
    client._mt5 = FakeMT5()

    bars = client.fetch_historical_bars("EURUSD", "M5", 2)

    assert len(bars) == 2
    assert bars[0].symbol == "EURUSD"
    assert bars[0].timeframe == "M5"
    assert bars[0].close == 1.15
