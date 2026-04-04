from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
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
    price: float | None = None


@dataclass(frozen=True)
class MT5ValidationIssue:
    scope: str
    code: str
    message: str
    details: dict[str, Any]


@dataclass(frozen=True)
class MT5ValidationReport:
    ok: bool
    connected: bool
    terminal_initialized: bool
    issues: list[MT5ValidationIssue]
    symbols: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "connected": self.connected,
            "terminal_initialized": self.terminal_initialized,
            "issues": [asdict(issue) for issue in self.issues],
            "symbols": self.symbols,
        }


@dataclass(frozen=True)
class DryRunOrderResult:
    accepted: bool
    reason: str
    normalized_volume: float | None
    request: dict[str, Any] | None
    validations: list[MT5ValidationIssue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "normalized_volume": self.normalized_volume,
            "request": self.request,
            "validations": [asdict(issue) for issue in self.validations],
        }


@dataclass(frozen=True)
class BrokerSnapshot:
    connected: bool
    account_info: dict[str, Any]
    positions: list[dict[str, Any]]
    pending_orders: list[dict[str, Any]]
    closed_trades: list[dict[str, Any]]
    scope_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MT5Client:
    """Client for historical data plus validated demo-order dry runs."""

    def __init__(self, config: MT5Config, mt5_module: Any | None = None) -> None:
        self.config = config
        self._mt5: Any | None = mt5_module
        self._connected = mt5_module is not None

    def _initialize_kwargs(self, *, authenticated: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.config.path:
            kwargs["path"] = self.config.path
        if authenticated and self.config.login and self.config.password and self.config.server:
            kwargs["login"] = self.config.login
            kwargs["password"] = self.config.password
            kwargs["server"] = self.config.server
        return kwargs

    def connect(self) -> bool:
        if self._mt5 is None:
            try:
                import MetaTrader5 as mt5  # type: ignore
            except ImportError:
                return False
            self._mt5 = mt5

        if self.config.login and self.config.password and self.config.server:
            auth_kwargs = self._initialize_kwargs(authenticated=True)
            if self._mt5.initialize(**auth_kwargs):
                self._connected = True
                return True
            if hasattr(self._mt5, "shutdown"):
                self._mt5.shutdown()
            base_kwargs = self._initialize_kwargs(authenticated=False)
            if not self._mt5.initialize(**base_kwargs):
                self._connected = False
                return False
            logged_in = bool(
                self._mt5.login(
                    self.config.login,
                    password=self.config.password,
                    server=self.config.server,
                )
            )
            self._connected = logged_in
            if not logged_in:
                self._mt5.shutdown()
            return logged_in
        kwargs = self._initialize_kwargs(authenticated=False)
        if not self._mt5.initialize(**kwargs):
            self._connected = False
            return False
        self._connected = True
        return True

    def shutdown(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
        self._connected = False

    def last_error(self) -> object:
        if self._mt5 is None or not hasattr(self._mt5, "last_error"):
            return None
        return self._mt5.last_error()

    def is_connected(self) -> bool:
        if self._mt5 is None or not self._connected:
            return False
        info = self._mt5.terminal_info() if hasattr(self._mt5, "terminal_info") else None
        return bool(info)

    def account_info(self) -> dict[str, Any] | None:
        if self._mt5 is None or not hasattr(self._mt5, "account_info"):
            return None
        info = self._mt5.account_info()
        if info is None:
            return None
        return info._asdict() if hasattr(info, "_asdict") else dict(info)

    def broker_state_snapshot(self, symbols: tuple[str, ...]) -> BrokerSnapshot:
        if self._mt5 is None or not self._connected:
            return BrokerSnapshot(False, {}, [], [], [], {"ownership_filter_active": False, "ignored_positions": 0, "ignored_orders": 0, "ignored_closed_trades": 0})
        account_info = self.account_info() or {}
        raw_positions = self._mt5.positions_get() if hasattr(self._mt5, "positions_get") else []
        positions = []
        ignored_positions = 0
        for item in raw_positions or []:
            data = item._asdict() if hasattr(item, "_asdict") else dict(item)
            if self._is_owned_record(data, symbols):
                positions.append(data)
            else:
                ignored_positions += 1
        raw_orders = self._mt5.orders_get() if hasattr(self._mt5, "orders_get") else []
        pending_orders = []
        ignored_orders = 0
        for item in raw_orders or []:
            data = item._asdict() if hasattr(item, "_asdict") else dict(item)
            if self._is_owned_record(data, symbols):
                pending_orders.append(data)
            else:
                ignored_orders += 1
        closed_trades: list[dict[str, Any]] = []
        ignored_closed_trades = 0
        if hasattr(self._mt5, "history_deals_get"):
            deals = self._mt5.history_deals_get(datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0), datetime.now(tz=UTC))
            for item in deals or []:
                data = item._asdict() if hasattr(item, "_asdict") else dict(item)
                if self._is_owned_record(data, symbols):
                    closed_trades.append(data)
                else:
                    ignored_closed_trades += 1
        return BrokerSnapshot(
            True,
            account_info,
            positions,
            pending_orders,
            closed_trades,
            {
                "ownership_filter_active": True,
                "magic_number": self.config.magic_number,
                "comment_tag": self.config.comment_tag,
                "reconcile_symbols_only": self.config.reconcile_symbols_only,
                "ignored_positions": ignored_positions,
                "ignored_orders": ignored_orders,
                "ignored_closed_trades": ignored_closed_trades,
            },
        )

    def broker_lifecycle_snapshot(self, symbols: tuple[str, ...], history_days: int) -> dict[str, Any]:
        if self._mt5 is None or not self._connected:
            return {
                "connected": False,
                "orders": [],
                "deals": [],
                "positions": [],
                "scope_report": {"ownership_filter_active": False},
            }
        start = datetime.now(tz=UTC) - timedelta(days=max(history_days, 1))
        end = datetime.now(tz=UTC)
        raw_positions = self._mt5.positions_get() if hasattr(self._mt5, "positions_get") else []
        raw_orders = self._mt5.history_orders_get(start, end) if hasattr(self._mt5, "history_orders_get") else []
        raw_deals = self._mt5.history_deals_get(start, end) if hasattr(self._mt5, "history_deals_get") else []
        positions: list[dict[str, Any]] = []
        orders: list[dict[str, Any]] = []
        deals: list[dict[str, Any]] = []
        ignored_positions = 0
        ignored_orders = 0
        ignored_deals = 0
        for item in raw_positions or []:
            data = item._asdict() if hasattr(item, "_asdict") else dict(item)
            if self._is_owned_record(data, symbols):
                positions.append(data)
            else:
                ignored_positions += 1
        for item in raw_orders or []:
            data = item._asdict() if hasattr(item, "_asdict") else dict(item)
            if self._is_owned_record(data, symbols):
                orders.append(data)
            else:
                ignored_orders += 1
        for item in raw_deals or []:
            data = item._asdict() if hasattr(item, "_asdict") else dict(item)
            if self._is_owned_record(data, symbols):
                deals.append(data)
            else:
                ignored_deals += 1
        return {
            "connected": True,
            "orders": orders,
            "deals": deals,
            "positions": positions,
            "scope_report": {
                "ownership_filter_active": True,
                "magic_number": self.config.magic_number,
                "comment_tag": self.config.comment_tag,
                "reconcile_symbols_only": self.config.reconcile_symbols_only,
                "ignored_positions": ignored_positions,
                "ignored_orders": ignored_orders,
                "ignored_deals": ignored_deals,
            },
        }

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
                    spread=float(rate_data.get("spread", 0.0)),
                )
            )
        return bars

    def check(self, symbols: tuple[str, ...]) -> MT5ValidationReport:
        issues: list[MT5ValidationIssue] = []
        symbol_reports: dict[str, dict[str, Any]] = {}
        connected = self._connected and self._mt5 is not None
        if not connected:
            issues.append(
                MT5ValidationIssue(
                    scope="connection",
                    code="not_connected",
                    message="MT5 no esta conectado",
                    details={},
                )
            )
            return MT5ValidationReport(False, False, False, issues, symbol_reports)

        terminal_initialized = True
        info = self._mt5.terminal_info() if hasattr(self._mt5, "terminal_info") else None
        if info is None:
            terminal_initialized = False
            issues.append(
                MT5ValidationIssue(
                    scope="terminal",
                    code="terminal_unavailable",
                    message="No se pudo leer terminal_info()",
                    details={},
                )
            )

        for symbol in symbols:
            symbol_result = self._validate_symbol(symbol)
            symbol_reports[symbol] = symbol_result
            for issue in symbol_result["issues"]:
                issues.append(
                    MT5ValidationIssue(
                        scope=f"symbol:{symbol}",
                        code=issue["code"],
                        message=issue["message"],
                        details=issue["details"],
                    )
                )

        return MT5ValidationReport(
            ok=connected and terminal_initialized and not issues,
            connected=connected,
            terminal_initialized=terminal_initialized,
            issues=issues,
            symbols=symbol_reports,
        )

    def dry_run_market_order(self, order: OrderRequest) -> DryRunOrderResult:
        if not self._connected or self._mt5 is None:
            issue = MT5ValidationIssue(
                scope="connection",
                code="not_connected",
                message="MT5 no esta conectado",
                details={},
            )
            return DryRunOrderResult(False, "not_connected", None, None, [issue])

        symbol_validation = self._validate_symbol(order.symbol)
        issues = [
            MT5ValidationIssue(
                scope=f"symbol:{order.symbol}",
                code=item["code"],
                message=item["message"],
                details=item["details"],
            )
            for item in symbol_validation["issues"]
        ]
        if issues:
            return DryRunOrderResult(False, "symbol_validation_failed", None, None, issues)

        info = symbol_validation["symbol_info"]
        normalized_volume = self._normalize_volume(
            requested=order.volume,
            min_lot=info["volume_min"],
            lot_step=info["volume_step"],
            max_lot=info["volume_max"],
        )
        if normalized_volume is None:
            issue = MT5ValidationIssue(
                scope=f"symbol:{order.symbol}",
                code="volume_invalid",
                message="El volumen no esta alineado con min/step/max",
                details={"requested_volume": order.volume, "symbol_info": info},
            )
            return DryRunOrderResult(False, "volume_invalid", None, None, issues + [issue])

        tick = self._mt5.symbol_info_tick(order.symbol)
        if tick is None:
            issue = MT5ValidationIssue(
                scope=f"symbol:{order.symbol}",
                code="tick_unavailable",
                message="No tick available",
                details={},
            )
            return DryRunOrderResult(False, "tick_unavailable", normalized_volume, None, issues + [issue])

        supported_filling = self._supported_fillings(info["filling_mode"])
        if not supported_filling:
            issue = MT5ValidationIssue(
                scope=f"symbol:{order.symbol}",
                code="filling_mode_unsupported",
                message="No hay filling mode soportado",
                details={"filling_mode": info["filling_mode"]},
            )
            return DryRunOrderResult(False, "filling_mode_unsupported", normalized_volume, None, issues + [issue])

        request = self._build_request(order, normalized_volume, tick, supported_filling[0])
        malformed = self._validate_request_payload(request)
        if malformed:
            issue = MT5ValidationIssue(
                scope=f"request:{order.symbol}",
                code="request_invalid",
                message="El request de orden no es valido",
                details={"request": request, "missing_fields": malformed},
            )
            return DryRunOrderResult(False, "request_invalid", normalized_volume, request, issues + [issue])

        return DryRunOrderResult(True, "dry_run_only", normalized_volume, request, issues)

    def _validate_symbol(self, symbol: str) -> dict[str, Any]:
        assert self._mt5 is not None
        issues: list[dict[str, Any]] = []
        raw_info = self._mt5.symbol_info(symbol)
        if raw_info is None:
            issues.append(
                {
                    "code": "symbol_missing",
                    "message": "El simbolo no existe en MT5",
                    "details": {"symbol": symbol},
                }
            )
            return {"ok": False, "issues": issues, "symbol_info": {}}

        info = raw_info._asdict() if hasattr(raw_info, "_asdict") else dict(raw_info)
        if not info.get("visible", False):
            selected = bool(self._mt5.symbol_select(symbol, True))
            info["visible"] = selected
            if not selected:
                issues.append(
                    {
                        "code": "symbol_not_visible",
                        "message": "El simbolo no es visible ni seleccionable",
                        "details": {"symbol": symbol},
                    }
                )
        trade_mode = int(info.get("trade_mode", 0))
        if trade_mode <= 0:
            issues.append(
                {
                    "code": "trading_not_allowed",
                    "message": "El simbolo no permite trading",
                    "details": {"trade_mode": trade_mode},
                }
            )
        volume_min = float(info.get("volume_min", 0.0))
        volume_step = float(info.get("volume_step", 0.0))
        volume_max = float(info.get("volume_max", 0.0))
        if volume_min <= 0.0:
            issues.append(
                {
                    "code": "min_lot_invalid",
                    "message": "volume_min invalido",
                    "details": {"volume_min": volume_min},
                }
            )
        if volume_step <= 0.0:
            issues.append(
                {
                    "code": "lot_step_invalid",
                    "message": "volume_step invalido",
                    "details": {"volume_step": volume_step},
                }
            )
        if volume_max <= 0.0 or volume_max < volume_min:
            issues.append(
                {
                    "code": "max_lot_invalid",
                    "message": "volume_max invalido",
                    "details": {"volume_max": volume_max, "volume_min": volume_min},
                }
            )
        if not self._supported_fillings(int(info.get("filling_mode", -1))):
            issues.append(
                {
                    "code": "filling_mode_invalid",
                    "message": "filling_mode no soportado",
                    "details": {"filling_mode": info.get("filling_mode")},
                }
            )
        return {"ok": not issues, "issues": issues, "symbol_info": info}

    def _supported_fillings(self, filling_mode: int) -> list[int]:
        assert self._mt5 is not None
        supported: list[int] = []
        for attr in ("ORDER_FILLING_IOC", "ORDER_FILLING_FOK", "ORDER_FILLING_RETURN"):
            if hasattr(self._mt5, attr):
                value = int(getattr(self._mt5, attr))
                if filling_mode == value or filling_mode & value == value:
                    supported.append(value)
        return supported

    def _normalize_volume(
        self,
        requested: float,
        min_lot: float,
        lot_step: float,
        max_lot: float,
    ) -> float | None:
        if requested < min_lot or requested > max_lot or lot_step <= 0.0:
            return None
        steps = round((requested - min_lot) / lot_step, 8)
        aligned = abs(steps - round(steps)) <= 1e-8
        if not aligned:
            return None
        return round(requested, 8)

    def _build_request(
        self,
        order: OrderRequest,
        normalized_volume: float,
        tick: Any,
        filling_mode: int,
    ) -> dict[str, Any]:
        assert self._mt5 is not None
        order_type = self._mt5.ORDER_TYPE_BUY if order.side == "buy" else self._mt5.ORDER_TYPE_SELL
        price = order.price
        if price is None:
            price = float(tick.ask if order.side == "buy" else tick.bid)
        return {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": order.symbol,
            "volume": normalized_volume,
            "type": order_type,
            "price": price,
            "sl": order.stop_loss,
            "tp": order.take_profit,
            "deviation": 20,
            "magic": self.config.magic_number,
            "comment": f"{self.config.comment_tag} demo dry-run",
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
        }

    def _is_owned_record(self, data: dict[str, Any], symbols: tuple[str, ...]) -> bool:
        if self.config.reconcile_symbols_only and symbols and data.get("symbol") not in symbols:
            return False
        magic = data.get("magic")
        if magic is not None:
            try:
                if int(magic) == self.config.magic_number:
                    return True
            except (TypeError, ValueError):
                pass
        comment = str(data.get("comment", ""))
        if self.config.comment_tag and self.config.comment_tag in comment:
            return True
        if self.config.reconcile_symbols_only and symbols and data.get("symbol") in symbols:
            return True
        return False

    def _validate_request_payload(self, request: dict[str, Any]) -> list[str]:
        required = [
            "action",
            "symbol",
            "volume",
            "type",
            "price",
            "sl",
            "tp",
            "deviation",
            "magic",
            "comment",
            "type_time",
            "type_filling",
        ]
        return [field for field in required if field not in request or request[field] in {None, ""}]


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
