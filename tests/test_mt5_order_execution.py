"""Tests for MT5Client.send_market_order() and close_position().

Uses a fully-mocked MT5 module — no real MT5 connection needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from iris_bot.config_types import MT5Config
from iris_bot.mt5 import MT5Client, OrderRequest, OrderResult


# ---------------------------------------------------------------------------
# MT5 module mock factory
# ---------------------------------------------------------------------------

def _make_mt5_module(
    *,
    order_send_retcode: int = 10009,     # TRADE_RETCODE_DONE
    order_send_returns_none: bool = False,
    fill_mode: int = 1,
) -> Any:
    """Build a minimal mock of the MetaTrader5 module."""
    mt5 = MagicMock()

    # Constants
    mt5.ORDER_TYPE_BUY = 0
    mt5.ORDER_TYPE_SELL = 1
    mt5.TRADE_ACTION_DEAL = 1
    mt5.ORDER_TIME_GTC = 1
    mt5.ORDER_FILLING_IOC = 1
    mt5.ORDER_FILLING_FOK = 2
    mt5.ORDER_FILLING_RETURN = 4
    mt5.TRADE_RETCODE_DONE = 10009

    # symbol_info
    info = MagicMock()
    info.visible = True
    info.trade_mode = 4
    info.volume_min = 0.01
    info.volume_step = 0.01
    info.volume_max = 100.0
    info.filling_mode = fill_mode
    info._asdict.return_value = {
        "visible": True,
        "trade_mode": 4,
        "volume_min": 0.01,
        "volume_step": 0.01,
        "volume_max": 100.0,
        "filling_mode": fill_mode,
    }
    mt5.symbol_info.return_value = info
    mt5.symbol_select.return_value = True

    # tick
    tick = MagicMock()
    tick.ask = 1.1002
    tick.bid = 1.1000
    mt5.symbol_info_tick.return_value = tick

    # order_send
    if order_send_returns_none:
        mt5.order_send.return_value = None
    else:
        result = MagicMock()
        result.retcode = order_send_retcode
        result.comment = "done" if order_send_retcode == 10009 else "error"
        result.order = 123456
        result.volume = 0.01
        result.price = 1.1002
        mt5.order_send.return_value = result

    return mt5


def _client(mt5_module: Any) -> MT5Client:
    cfg = MT5Config(magic_number=99999, comment_tag="TEST")
    client = MT5Client(cfg, mt5_module=mt5_module)
    return client


# ---------------------------------------------------------------------------
# send_market_order tests
# ---------------------------------------------------------------------------

def test_send_market_order_success() -> None:
    mt5 = _make_mt5_module(order_send_retcode=10009)
    client = _client(mt5)

    order = OrderRequest(symbol="EURUSD", side="buy", volume=0.01, stop_loss=1.0990, take_profit=1.1020)
    result = client.send_market_order(order)

    assert result.accepted is True
    assert result.retcode == 10009
    assert result.ticket == 123456
    assert result.price == 1.1002
    mt5.order_send.assert_called_once()


def test_send_market_order_broker_rejects() -> None:
    mt5 = _make_mt5_module(order_send_retcode=10006)   # TRADE_RETCODE_REJECT
    client = _client(mt5)

    order = OrderRequest(symbol="EURUSD", side="buy", volume=0.01, stop_loss=1.0990, take_profit=1.1020)
    result = client.send_market_order(order)

    assert result.accepted is False
    assert result.retcode == 10006


def test_send_market_order_returns_none() -> None:
    mt5 = _make_mt5_module(order_send_returns_none=True)
    client = _client(mt5)

    order = OrderRequest(symbol="EURUSD", side="buy", volume=0.01, stop_loss=1.0990, take_profit=1.1020)
    result = client.send_market_order(order)

    assert result.accepted is False
    assert result.comment == "order_send_returned_none"


def test_send_market_order_not_connected() -> None:
    cfg = MT5Config()
    client = MT5Client(cfg)   # no mt5_module → not connected

    order = OrderRequest(symbol="EURUSD", side="buy", volume=0.01, stop_loss=1.0990, take_profit=1.1020)
    result = client.send_market_order(order)

    assert result.accepted is False


def test_send_market_order_short() -> None:
    mt5 = _make_mt5_module(order_send_retcode=10009)
    client = _client(mt5)

    order = OrderRequest(symbol="EURUSD", side="sell", volume=0.01, stop_loss=1.1020, take_profit=1.0980)
    result = client.send_market_order(order)

    assert result.accepted is True
    call_request = mt5.order_send.call_args[0][0]
    assert call_request["type"] == mt5.ORDER_TYPE_SELL


# ---------------------------------------------------------------------------
# close_position tests
# ---------------------------------------------------------------------------

def test_close_position_success() -> None:
    mt5 = _make_mt5_module(order_send_retcode=10009)
    client = _client(mt5)

    result = client.close_position(ticket=123456, symbol="EURUSD", volume=0.01, side="buy")

    assert result.accepted is True
    call_request = mt5.order_send.call_args[0][0]
    assert call_request["position"] == 123456
    assert call_request["type"] == mt5.ORDER_TYPE_SELL   # closing a buy = sell


def test_close_position_short() -> None:
    mt5 = _make_mt5_module(order_send_retcode=10009)
    client = _client(mt5)

    result = client.close_position(ticket=789, symbol="EURUSD", volume=0.01, side="sell")

    assert result.accepted is True
    call_request = mt5.order_send.call_args[0][0]
    assert call_request["type"] == mt5.ORDER_TYPE_BUY    # closing a sell = buy


def test_close_position_no_tick() -> None:
    mt5 = _make_mt5_module()
    mt5.symbol_info_tick.return_value = None
    client = _client(mt5)

    result = client.close_position(ticket=1, symbol="EURUSD", volume=0.01, side="buy")

    assert result.accepted is False
    assert result.comment == "tick_unavailable"


def test_close_position_not_connected() -> None:
    cfg = MT5Config()
    client = MT5Client(cfg)

    result = client.close_position(ticket=1, symbol="EURUSD", volume=0.01, side="buy")

    assert result.accepted is False
    assert result.comment == "not_connected"


# ---------------------------------------------------------------------------
# OrderResult serialization
# ---------------------------------------------------------------------------

def test_order_result_to_dict() -> None:
    r = OrderResult(
        accepted=True, retcode=10009, comment="done",
        ticket=1, volume=0.01, price=1.1002, request={"symbol": "EURUSD"},
    )
    d = r.to_dict()
    assert d["accepted"] is True
    assert d["retcode"] == 10009
    assert d["ticket"] == 1
