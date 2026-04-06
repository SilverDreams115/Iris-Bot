from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from iris_bot.config import Settings
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.mt5 import MT5Client, OrderRequest, OrderResult


@dataclass(frozen=True)
class DemoLiveProbeReport:
    ok: bool
    status: str
    reason: str
    symbol: str
    requested_volume: float
    account_info: dict[str, Any]
    demo_guard: dict[str, Any]
    open_order_result: dict[str, Any] | None
    opened_position: dict[str, Any] | None
    close_order_result: dict[str, Any] | None
    remaining_positions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_demo_account(account_info: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    indicators = {
        "server": str(account_info.get("server", "") or ""),
        "company": str(account_info.get("company", "") or ""),
        "name": str(account_info.get("name", "") or ""),
    }
    lowered = " ".join(value.lower() for value in indicators.values())
    return ("demo" in lowered), indicators


def _pick_symbol(settings: Settings) -> str:
    if not settings.trading.symbols:
        raise ValueError("No trading symbols configured")
    return settings.trading.symbols[0]


def _build_probe_order(symbol: str, side: str, tick: dict[str, float]) -> OrderRequest:
    entry_price = tick["ask"] if side == "buy" else tick["bid"]
    stop_distance = max(entry_price * 0.0010, 0.0005)
    take_profit_distance = max(entry_price * 0.0015, 0.0008)
    if side == "buy":
        stop_loss = entry_price - stop_distance
        take_profit = entry_price + take_profit_distance
    else:
        stop_loss = entry_price + stop_distance
        take_profit = entry_price - take_profit_distance
    return OrderRequest(
        symbol=symbol,
        side=side,
        volume=0.01,
        stop_loss=round(stop_loss, 8),
        take_profit=round(take_profit, 8),
        price=round(entry_price, 8),
    )


def _extract_tick(client: MT5Client, symbol: str) -> dict[str, float] | None:
    mt5_module = client._mt5  # noqa: SLF001 - narrow operational wrapper around MT5 adapter
    if mt5_module is None or not hasattr(mt5_module, "symbol_info_tick"):
        return None
    tick = mt5_module.symbol_info_tick(symbol)
    if tick is None:
        return None
    return {
        "ask": float(getattr(tick, "ask", 0.0)),
        "bid": float(getattr(tick, "bid", 0.0)),
    }


def _find_owned_position(
    client: MT5Client,
    symbol: str,
    *,
    attempts: int = 5,
    sleep_seconds: float = 0.2,
) -> dict[str, Any] | None:
    for _ in range(attempts):
        snapshot = client.broker_state_snapshot((symbol,))
        for position in snapshot.positions:
            if str(position.get("symbol", "")) == symbol:
                return position
        time.sleep(sleep_seconds)
    return None


def _failure(
    *,
    status: str,
    reason: str,
    symbol: str,
    requested_volume: float,
    account_info: dict[str, Any],
    demo_guard: dict[str, Any],
    open_order_result: dict[str, Any] | None,
    opened_position: dict[str, Any] | None,
    close_order_result: dict[str, Any] | None,
    remaining_positions: list[dict[str, Any]],
) -> DemoLiveProbeReport:
    return DemoLiveProbeReport(
        ok=False,
        status=status,
        reason=reason,
        symbol=symbol,
        requested_volume=requested_volume,
        account_info=account_info,
        demo_guard=demo_guard,
        open_order_result=open_order_result,
        opened_position=opened_position,
        close_order_result=close_order_result,
        remaining_positions=remaining_positions,
    )


def run_demo_live_probe(
    settings: Settings,
    client_factory: Callable[[], MT5Client] | None = None,
) -> tuple[int, Path, DemoLiveProbeReport]:
    run_dir = build_run_directory(settings.data.runs_dir, "demo_live_probe")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    symbol = _pick_symbol(settings)
    requested_volume = 0.01
    client = client_factory() if client_factory is not None else MT5Client(settings.mt5)

    if not settings.mt5.enabled:
        report = _failure(
            status="blocked",
            reason="mt5_disabled",
            symbol=symbol,
            requested_volume=requested_volume,
            account_info={},
            demo_guard={"ok": False, "reason": "mt5_disabled"},
            open_order_result=None,
            opened_position=None,
            close_order_result=None,
            remaining_positions=[],
        )
        write_json_report(run_dir, "demo_live_probe_report.json", report.to_dict())
        logger.error("demo_live_probe blocked: mt5_disabled")
        return 1, run_dir, report

    if not client.connect():
        report = _failure(
            status="failed",
            reason="connect_failed",
            symbol=symbol,
            requested_volume=requested_volume,
            account_info={},
            demo_guard={"ok": False, "reason": "connect_failed"},
            open_order_result=None,
            opened_position=None,
            close_order_result=None,
            remaining_positions=[],
        )
        write_json_report(run_dir, "demo_live_probe_report.json", report.to_dict())
        logger.error("demo_live_probe failed: connect_failed")
        return 2, run_dir, report

    try:
        account_info = client.account_info() or {}
        is_demo, indicators = _is_demo_account(account_info)
        demo_guard = {"ok": is_demo, "reason": "demo_account_confirmed" if is_demo else "account_not_demo", "indicators": indicators}
        if not is_demo:
            report = _failure(
                status="blocked",
                reason="account_not_demo",
                symbol=symbol,
                requested_volume=requested_volume,
                account_info=account_info,
                demo_guard=demo_guard,
                open_order_result=None,
                opened_position=None,
                close_order_result=None,
                remaining_positions=[],
            )
            write_json_report(run_dir, "demo_live_probe_report.json", report.to_dict())
            logger.error("demo_live_probe blocked: account_not_demo")
            return 3, run_dir, report

        tick = _extract_tick(client, symbol)
        if tick is None:
            report = _failure(
                status="failed",
                reason="tick_unavailable",
                symbol=symbol,
                requested_volume=requested_volume,
                account_info=account_info,
                demo_guard=demo_guard,
                open_order_result=None,
                opened_position=None,
                close_order_result=None,
                remaining_positions=[],
            )
            write_json_report(run_dir, "demo_live_probe_report.json", report.to_dict())
            logger.error("demo_live_probe failed: tick_unavailable")
            return 4, run_dir, report

        order = _build_probe_order(symbol, "buy", tick)
        open_result = client.send_market_order(order)
        if not open_result.accepted:
            report = _failure(
                status="failed",
                reason="open_order_rejected",
                symbol=symbol,
                requested_volume=requested_volume,
                account_info=account_info,
                demo_guard=demo_guard,
                open_order_result=open_result.to_dict(),
                opened_position=None,
                close_order_result=None,
                remaining_positions=[],
            )
            write_json_report(run_dir, "demo_live_probe_report.json", report.to_dict())
            logger.error("demo_live_probe failed: open_order_rejected retcode=%s", open_result.retcode)
            return 5, run_dir, report

        position = _find_owned_position(client, symbol)
        if position is None:
            report = _failure(
                status="failed",
                reason="opened_position_not_found",
                symbol=symbol,
                requested_volume=requested_volume,
                account_info=account_info,
                demo_guard=demo_guard,
                open_order_result=open_result.to_dict(),
                opened_position=None,
                close_order_result=None,
                remaining_positions=[],
            )
            write_json_report(run_dir, "demo_live_probe_report.json", report.to_dict())
            logger.error("demo_live_probe failed: opened_position_not_found")
            return 6, run_dir, report

        close_result = client.close_position(
            ticket=int(position.get("ticket", 0)),
            symbol=symbol,
            volume=float(position.get("volume", requested_volume)),
            side="buy" if int(position.get("type", 0)) == 0 else "sell",
        )
        remaining_snapshot = client.broker_state_snapshot((symbol,))
        remaining_positions = remaining_snapshot.positions
        if not close_result.accepted:
            report = _failure(
                status="failed",
                reason="close_order_rejected",
                symbol=symbol,
                requested_volume=requested_volume,
                account_info=account_info,
                demo_guard=demo_guard,
                open_order_result=open_result.to_dict(),
                opened_position=position,
                close_order_result=close_result.to_dict(),
                remaining_positions=remaining_positions,
            )
            write_json_report(run_dir, "demo_live_probe_report.json", report.to_dict())
            logger.error("demo_live_probe failed: close_order_rejected retcode=%s", close_result.retcode)
            return 7, run_dir, report

        report = DemoLiveProbeReport(
            ok=True,
            status="completed",
            reason="open_and_close_succeeded",
            symbol=symbol,
            requested_volume=requested_volume,
            account_info=account_info,
            demo_guard=demo_guard,
            open_order_result=open_result.to_dict(),
            opened_position=position,
            close_order_result=close_result.to_dict(),
            remaining_positions=remaining_positions,
        )
        write_json_report(run_dir, "demo_live_probe_report.json", report.to_dict())
        logger.info(
            "demo_live_probe completed symbol=%s open_ticket=%s close_ticket=%s remaining_positions=%s",
            symbol,
            report.open_order_result.get("ticket") if report.open_order_result else None,
            report.close_order_result.get("ticket") if report.close_order_result else None,
            len(remaining_positions),
        )
        return 0, run_dir, report
    finally:
        client.shutdown()
