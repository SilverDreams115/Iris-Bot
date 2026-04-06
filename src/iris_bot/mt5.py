from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, cast

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
    deviation: int | None = None


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
class OrderResult:
    accepted: bool
    retcode: int
    comment: str
    ticket: int | None     # MT5 order/position ticket on success
    volume: float | None   # actual filled volume
    price: float | None    # actual fill price
    request: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


@dataclass(frozen=True)
class MT5SessionHealth:
    ok: bool
    connected: bool
    terminal_available: bool
    account_accessible: bool
    last_error: object
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MT5Client:
    """Client for historical data plus validated demo-order dry runs."""

    def __init__(self, config: MT5Config, mt5_module: Any | None = None) -> None:
        self.config = config
        self._mt5: Any | None = mt5_module
        self._connected = mt5_module is not None
        self._last_health: MT5SessionHealth | None = None

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

        self._connected = False
        if self.config.login and self.config.password and self.config.server:
            auth_kwargs = self._initialize_kwargs(authenticated=True)
            if self._mt5.initialize(**auth_kwargs):
                self._connected = True
                if self.health_check().ok:
                    return True
                self.shutdown()
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
                return False
            if self.health_check().ok:
                return True
            self.shutdown()
            return False
        kwargs = self._initialize_kwargs(authenticated=False)
        if not self._mt5.initialize(**kwargs):
            self._connected = False
            return False
        self._connected = True
        if self.health_check(require_account=False).ok:
            return True
        self.shutdown()
        return False

    def shutdown(self) -> None:
        if self._mt5 is not None:
            self._mt5.shutdown()
        self._connected = False

    def last_error(self) -> object:
        if self._mt5 is None or not hasattr(self._mt5, "last_error"):
            return None
        return self._mt5.last_error()

    def is_connected(self) -> bool:
        return self.health_check().ok

    def health_check(self, *, require_account: bool = True) -> MT5SessionHealth:
        if self._mt5 is None:
            health = MT5SessionHealth(
                ok=self._connected,
                connected=self._connected,
                terminal_available=self._connected,
                account_accessible=self._connected,
                last_error=self.last_error(),
                details={"reason": "external_client_without_mt5_module"},
            )
            self._last_health = health
            return health
        if not self._connected:
            health = MT5SessionHealth(
                ok=False,
                connected=False,
                terminal_available=False,
                account_accessible=False,
                last_error=self.last_error(),
                details={"reason": "not_connected"},
            )
            self._last_health = health
            return health

        terminal_info = None
        if hasattr(self._mt5, "terminal_info"):
            terminal_info = self._mt5.terminal_info()
        terminal_available = terminal_info is not None

        account_info = None
        account_capability_available = hasattr(self._mt5, "account_info")
        if require_account and account_capability_available:
            account_info = self._mt5.account_info()
        account_accessible = (not require_account) or (not account_capability_available) or account_info is not None
        health = MT5SessionHealth(
            ok=terminal_available and account_accessible,
            connected=True,
            terminal_available=terminal_available,
            account_accessible=account_accessible,
            last_error=self.last_error(),
            details={
                "require_account": require_account,
                "terminal_info_present": terminal_available,
                "account_capability_available": account_capability_available,
                "account_info_present": account_info is not None if require_account and account_capability_available else None,
            },
        )
        self._last_health = health
        if not health.ok:
            self._connected = False
        return health

    def ensure_connection(self, *, require_account: bool = True, force_reconnect: bool = False) -> bool:
        if not force_reconnect:
            health = self.health_check(require_account=require_account)
            if health.ok:
                return True
        self.shutdown()
        if not self.connect():
            return False
        return self.health_check(require_account=require_account).ok

    def session_health(self) -> MT5SessionHealth | None:
        return self._last_health

    def account_info(self) -> dict[str, Any] | None:
        if not self.ensure_connection():
            return None
        if self._mt5 is None or not hasattr(self._mt5, "account_info"):
            return None
        info = self._mt5.account_info()
        if info is None:
            self._connected = False
            return None
        return info._asdict() if hasattr(info, "_asdict") else dict(info)

    def broker_state_snapshot(self, symbols: tuple[str, ...]) -> BrokerSnapshot:
        if not self.ensure_connection():
            return BrokerSnapshot(False, {}, [], [], [], {"ownership_filter_active": False, "ignored_positions": 0, "ignored_orders": 0, "ignored_closed_trades": 0})
        assert self._mt5 is not None
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
        if not self.ensure_connection():
            return {
                "connected": False,
                "orders": [],
                "deals": [],
                "positions": [],
                "scope_report": {"ownership_filter_active": False},
            }
        assert self._mt5 is not None
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
        if not self.ensure_connection(require_account=False):
            raise RuntimeError("MT5 is not connected")
        assert self._mt5 is not None

        timeframe_code = resolve_mt5_timeframe(self._mt5, timeframe)
        rates = self._mt5.copy_rates_from_pos(symbol, timeframe_code, 0, count)
        if rates is None:
            raise RuntimeError(f"No se pudieron descargar barras para {symbol} {timeframe}")

        bars: list[Bar] = []
        for rate in rates:
            # MT5 returns numpy structured array rows — convert via dtype field names.
            if hasattr(rate, "_asdict"):
                rate_data = rate._asdict()
            elif hasattr(rate, "dtype"):
                rate_data = {name: rate[name] for name in rate.dtype.names}
            else:
                rate_data = dict(rate)
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
        health = self.health_check()
        connected = health.connected and self._mt5 is not None
        if not connected:
            issues.append(
                MT5ValidationIssue(
                    scope="connection",
                    code="not_connected",
                    message="MT5 is not connected",
                    details={"health": health.to_dict()},
                )
            )
            return MT5ValidationReport(False, False, False, issues, symbol_reports)

        terminal_initialized = health.terminal_available
        if not terminal_initialized:
            issues.append(
                MT5ValidationIssue(
                    scope="terminal",
                    code="terminal_unavailable",
                    message="No se pudo leer terminal_info()",
                    details={"health": health.to_dict()},
                )
            )
        if not health.account_accessible:
            issues.append(
                MT5ValidationIssue(
                    scope="account",
                    code="account_unavailable",
                    message="No se pudo leer account_info()",
                    details={"health": health.to_dict()},
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
        if not self.ensure_connection():
            issue = MT5ValidationIssue(
                scope="connection",
                code="not_connected",
                message="MT5 is not connected",
                details={"health": self.session_health().to_dict() if self.session_health() else {}},
            )
            return DryRunOrderResult(False, "not_connected", None, None, [issue])
        assert self._mt5 is not None

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
                message="Volume is not aligned with min/step/max constraints",
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

        candidate_fillings = self._candidate_fillings(int(info.get("filling_mode", -1)))
        if not candidate_fillings:
            issue = MT5ValidationIssue(
                scope=f"symbol:{order.symbol}",
                code="filling_mode_unsupported",
                message="No supported filling mode available",
                details={"filling_mode": info["filling_mode"]},
            )
            return DryRunOrderResult(False, "filling_mode_unsupported", normalized_volume, None, issues + [issue])

        request = self._resolve_request_with_filling(order, normalized_volume, tick, candidate_fillings)
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

    def send_market_order(self, order: OrderRequest) -> OrderResult:
        """Send a real market order to MT5. Validates via dry_run first, then calls order_send."""
        dry = self.dry_run_market_order(order)
        if not dry.accepted:
            return OrderResult(
                accepted=False,
                retcode=-1,
                comment=dry.reason,
                ticket=None,
                volume=None,
                price=None,
                request=dry.request or {},
            )
        if self._mt5 is None:
            return OrderResult(False, -1, "not_connected", None, None, None, dry.request or {})
        result = self._mt5.order_send(dry.request)
        if result is None:
            return OrderResult(False, -1, "order_send_returned_none", None, None, None, dry.request or {})
        retcode = int(getattr(result, "retcode", -1))
        retcode_done = getattr(self._mt5, "TRADE_RETCODE_DONE", 10009)
        accepted = retcode == retcode_done
        return OrderResult(
            accepted=accepted,
            retcode=retcode,
            comment=str(getattr(result, "comment", "")),
            ticket=int(getattr(result, "order", 0)) or None,
            volume=float(getattr(result, "volume", 0.0)) or None,
            price=float(getattr(result, "price", 0.0)) or None,
            request=dry.request or {},
        )

    def close_position(self, ticket: int, symbol: str, volume: float, side: str) -> OrderResult:
        """Close an open MT5 position by ticket. side is the original position side ('buy'|'sell')."""
        if not self.ensure_connection():
            return OrderResult(False, -1, "not_connected", None, None, None, {})
        assert self._mt5 is not None
        close_side = "sell" if side == "buy" else "buy"
        order_type = self._mt5.ORDER_TYPE_SELL if close_side == "sell" else self._mt5.ORDER_TYPE_BUY
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(False, -1, "tick_unavailable", None, None, None, {})
        price = float(tick.bid if close_side == "sell" else tick.ask)
        symbol_info = self._validate_symbol(symbol)["symbol_info"]
        fillings = self._candidate_fillings(int(symbol_info.get("filling_mode", -1)))
        if not fillings:
            return OrderResult(False, -1, "filling_mode_unsupported", None, None, None, {})
        request: dict[str, Any] = self._resolve_close_request_with_filling(
            ticket=ticket,
            symbol=symbol,
            volume=volume,
            order_type=order_type,
            price=price,
            fillings=fillings,
        )
        result = self._mt5.order_send(request)
        if result is None:
            return OrderResult(False, -1, "order_send_returned_none", None, None, None, request)
        retcode = int(getattr(result, "retcode", -1))
        retcode_done = getattr(self._mt5, "TRADE_RETCODE_DONE", 10009)
        accepted = retcode == retcode_done
        return OrderResult(
            accepted=accepted,
            retcode=retcode,
            comment=str(getattr(result, "comment", "")),
            ticket=int(getattr(result, "order", 0)) or None,
            volume=float(getattr(result, "volume", 0.0)) or None,
            price=float(getattr(result, "price", 0.0)) or None,
            request=request,
        )

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
                    "message": "volume_min is invalid",
                    "details": {"volume_min": volume_min},
                }
            )
        if volume_step <= 0.0:
            issues.append(
                {
                    "code": "lot_step_invalid",
                    "message": "volume_step is invalid",
                    "details": {"volume_step": volume_step},
                }
            )
        if volume_max <= 0.0 or volume_max < volume_min:
            issues.append(
                {
                    "code": "max_lot_invalid",
                    "message": "volume_max is invalid",
                    "details": {"volume_max": volume_max, "volume_min": volume_min},
                }
            )
        if not self._candidate_fillings(int(info.get("filling_mode", -1))):
            issues.append(
                {
                    "code": "filling_mode_invalid",
                    "message": "filling_mode not supported",
                    "details": {"filling_mode": info.get("filling_mode")},
                }
            )
        return {"ok": not issues, "issues": issues, "symbol_info": info}

    def _candidate_fillings(self, filling_mode: int) -> list[int]:
        assert self._mt5 is not None
        constants = self._known_fillings()
        preferred: list[int] = []
        for value in constants:
            if value == filling_mode and value not in preferred:
                preferred.append(value)
        for value in constants:
            if value != 0 and filling_mode >= 0 and filling_mode & value == value and value not in preferred:
                preferred.append(value)
        return preferred

    def _known_fillings(self) -> list[int]:
        assert self._mt5 is not None
        constants: list[int] = []
        for attr in ("ORDER_FILLING_FOK", "ORDER_FILLING_IOC", "ORDER_FILLING_RETURN"):
            if hasattr(self._mt5, attr):
                value = int(getattr(self._mt5, attr))
                if value not in constants:
                    constants.append(value)
        return constants

    def _resolve_request_with_filling(
        self,
        order: OrderRequest,
        normalized_volume: float,
        tick: Any,
        fillings: list[int],
    ) -> dict[str, Any]:
        expanded_fillings = list(fillings)
        if self._mt5 is not None and hasattr(self._mt5, "order_check"):
            for filling in self._known_fillings():
                if filling not in expanded_fillings:
                    expanded_fillings.append(filling)
        for filling in expanded_fillings:
            request = self._build_request(order, normalized_volume, tick, filling)
            if self._order_check_accepts(request):
                return request
        return self._build_request(order, normalized_volume, tick, fillings[0])

    def _resolve_close_request_with_filling(
        self,
        *,
        ticket: int,
        symbol: str,
        volume: float,
        order_type: int,
        price: float,
        fillings: list[int],
    ) -> dict[str, Any]:
        assert self._mt5 is not None
        expanded_fillings = list(fillings)
        if hasattr(self._mt5, "order_check"):
            for filling in self._known_fillings():
                if filling not in expanded_fillings:
                    expanded_fillings.append(filling)
        for filling in expanded_fillings:
            request = {
                "action": self._mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": self.config.magic_number,
                "comment": f"{self.config.comment_tag} close",
                "type_time": self._mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            if self._order_check_accepts(request):
                return request
        return {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.config.magic_number,
            "comment": f"{self.config.comment_tag} close",
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": fillings[0],
        }

    def _order_check_accepts(self, request: dict[str, Any]) -> bool:
        if self._mt5 is None or not hasattr(self._mt5, "order_check"):
            return False
        result = self._mt5.order_check(request)
        if result is None:
            return False
        return int(getattr(result, "retcode", -1)) == 0

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
            "deviation": int(order.deviation or 20),
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
        raise ValueError(f"Unsupported timeframe for MT5: {timeframe}")
    return cast(int, getattr(mt5_module, attr_name))
