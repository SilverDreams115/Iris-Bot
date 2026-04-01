from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from iris_bot.config import MT5Config
from iris_bot.data import Bar


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str
    volume: float
    stop_loss: float
    take_profit: float


class MT5Client:
    """Adaptador minimo para demo/live. Requiere el paquete MetaTrader5 instalado."""

    def __init__(self, config: MT5Config) -> None:
        self.config = config
        self._mt5: Any | None = None

    def connect(self) -> bool:
        try:
            import MetaTrader5 as mt5  # type: ignore
        except ImportError:
            return False

        self._mt5 = mt5
        kwargs: dict[str, Any] = {}
        if self.config.path:
            kwargs["path"] = self.config.path
        if not mt5.initialize(**kwargs):
            return False
        if self.config.login and self.config.password and self.config.server:
            return mt5.login(self.config.login, password=self.config.password, server=self.config.server)
        return True

    def shutdown(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()

    def send_market_order(self, order: OrderRequest) -> dict[str, Any]:
        if self._mt5 is None:
            raise RuntimeError("MT5 no esta conectado")

        tick = self._mt5.symbol_info_tick(order.symbol)
        if tick is None:
            raise RuntimeError(f"No hay tick disponible para {order.symbol}")

        order_type = self._mt5.ORDER_TYPE_BUY if order.side == "buy" else self._mt5.ORDER_TYPE_SELL
        price = tick.ask if order.side == "buy" else tick.bid
        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": order.volume,
            "type": order_type,
            "price": price,
            "sl": order.stop_loss,
            "tp": order.take_profit,
            "deviation": 20,
            "magic": 20260401,
            "comment": "IRIS-Bot",
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }
        result = self._mt5.order_send(request)
        return result._asdict() if hasattr(result, "_asdict") else {"result": result}

    def fetch_historical_bars(self, symbol: str, timeframe: str, count: int) -> list[Bar]:
        if self._mt5 is None:
            raise RuntimeError("MT5 no esta conectado")

        timeframe_code = resolve_mt5_timeframe(self._mt5, timeframe)
        rates = self._mt5.copy_rates_from_pos(symbol, timeframe_code, 0, count)
        if rates is None:
            raise RuntimeError(f"No se pudieron descargar barras para {symbol} {timeframe}")

        bars: list[Bar] = []
        for rate in rates:
            rate_data = rate._asdict() if hasattr(rate, "_asdict") else dict(rate)
            bars.append(
                Bar(
                    timestamp=datetime.fromtimestamp(rate_data["time"], tz=UTC).replace(tzinfo=None),
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(rate_data["open"]),
                    high=float(rate_data["high"]),
                    low=float(rate_data["low"]),
                    close=float(rate_data["close"]),
                    volume=float(rate_data.get("tick_volume", rate_data.get("real_volume", 0.0))),
                )
            )
        return bars


def resolve_mt5_timeframe(mt5_module: Any, timeframe: str) -> int:
    mapping = {
        "M1": "TIMEFRAME_M1",
        "M5": "TIMEFRAME_M5",
        "M15": "TIMEFRAME_M15",
        "M30": "TIMEFRAME_M30",
        "H1": "TIMEFRAME_H1",
        "H4": "TIMEFRAME_H4",
        "D1": "TIMEFRAME_D1",
    }
    attr_name = mapping.get(timeframe)
    if attr_name is None or not hasattr(mt5_module, attr_name):
        raise ValueError(f"Timeframe no soportado para MT5: {timeframe}")
    return getattr(mt5_module, attr_name)
