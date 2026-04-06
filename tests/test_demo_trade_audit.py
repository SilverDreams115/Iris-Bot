"""Tests for demo_trade_audit module.

All tests use pure in-memory data — no real MT5 connection required.
"""
from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_bot.demo_trade_audit import (
    DemoTradeAuditReport,
    FillQualityStats,
    PnlDivergenceStats,
    SlippageStats,
    _compute_fill_quality,
    _compute_pnl_divergence,
    _compute_slippage,
    _pip_value,
    match_trades,
    run_demo_trade_audit,
)
from iris_bot.config_types import (
    BacktestConfig,
    DataConfig,
    DemoAuditConfig,
    LoggingConfig,
    MT5Config,
    OperationalConfig,
    ReconciliationConfig,
    RecoveryConfig,
    RiskConfig,
    SessionConfig,
    Settings,
    TradingConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(tmp_path: Path) -> Settings:
    data_cfg = DataConfig(
        raw_dir=tmp_path / "data/raw",
        processed_dir=tmp_path / "data/processed",
        runs_dir=tmp_path / "runs",
        runtime_dir=tmp_path / "data/runtime",
    )
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    return Settings(
        data=data_cfg,
        demo_audit=DemoAuditConfig(history_days=7, timestamp_tolerance_seconds=300),
    )


def _write_closed_trades(run_dir: Path, trades: list[dict]) -> None:
    path = run_dir / "closed_trades.csv"
    if not trades:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=trades[0].keys())
        writer.writeheader()
        writer.writerows(trades)


def _broker_deal(
    symbol: str,
    side: str,
    price: float,
    profit: float,
    time_unix: float,
) -> dict:
    deal_type = 0 if side == "buy" else 1
    return {
        "symbol": symbol,
        "type": deal_type,
        "entry": 0,         # 0 = entry deal
        "price": price,
        "profit": profit,
        "time": time_unix,
        "comment": "",
        "reason": 0,
    }


def _local_trade(
    symbol: str,
    direction: int,
    entry_price: float,
    net_pnl_usd: float,
    entry_ts: str,
) -> dict:
    return {
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry_price,
        "net_pnl_usd": net_pnl_usd,
        "entry_timestamp": entry_ts,
    }


# ---------------------------------------------------------------------------
# Unit tests: pip value
# ---------------------------------------------------------------------------

def test_pip_value_standard() -> None:
    assert _pip_value("EURUSD") == 0.0001


def test_pip_value_jpy() -> None:
    assert _pip_value("USDJPY") == 0.01


# ---------------------------------------------------------------------------
# Unit tests: match_trades
# ---------------------------------------------------------------------------

def test_match_trades_exact_match() -> None:
    ts_unix = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp()
    ts_iso = "2026-01-01T12:00:00"

    local = [_local_trade("EURUSD", 1, 1.1000, 5.0, ts_iso)]
    broker = [_broker_deal("EURUSD", "buy", 1.1003, 4.8, ts_unix)]

    matched, unmatched_local, unmatched_broker = match_trades(local, broker, timestamp_tolerance_seconds=300)

    assert len(matched) == 1
    assert not unmatched_local
    assert not unmatched_broker


def test_match_trades_timestamp_outside_tolerance() -> None:
    ts_unix = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp()
    ts_iso = "2026-01-01T10:00:00"   # 2 hours apart — outside 300s tolerance

    local = [_local_trade("EURUSD", 1, 1.1000, 5.0, ts_iso)]
    broker = [_broker_deal("EURUSD", "buy", 1.1003, 4.8, ts_unix)]

    matched, unmatched_local, unmatched_broker = match_trades(local, broker)

    assert not matched
    assert len(unmatched_local) == 1
    assert len(unmatched_broker) == 1


def test_match_trades_side_mismatch_not_matched() -> None:
    ts_unix = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp()
    ts_iso = "2026-01-01T12:00:00"

    local = [_local_trade("EURUSD", 1, 1.1000, 5.0, ts_iso)]   # long (buy)
    broker = [_broker_deal("EURUSD", "sell", 1.1003, -4.8, ts_unix)]  # sell

    matched, unmatched_local, unmatched_broker = match_trades(local, broker)

    assert not matched
    assert len(unmatched_local) == 1
    assert len(unmatched_broker) == 1


def test_match_trades_symbol_mismatch() -> None:
    ts_unix = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp()
    ts_iso = "2026-01-01T12:00:00"

    local = [_local_trade("EURUSD", 1, 1.1000, 5.0, ts_iso)]
    broker = [_broker_deal("GBPUSD", "buy", 1.3000, 5.0, ts_unix)]

    matched, unmatched_local, unmatched_broker = match_trades(local, broker)

    assert not matched
    assert len(unmatched_local) == 1
    assert len(unmatched_broker) == 1


def test_match_trades_multiple_greedy() -> None:
    """Two local trades with nearby broker deals — greedy match by proximity."""
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp()
    local = [
        _local_trade("EURUSD", 1, 1.1000, 5.0, "2026-01-01T12:00:00"),
        _local_trade("EURUSD", 1, 1.1020, 3.0, "2026-01-01T13:00:00"),
    ]
    broker = [
        _broker_deal("EURUSD", "buy", 1.1002, 4.9, base + 10),
        _broker_deal("EURUSD", "buy", 1.1022, 2.9, base + 3600 + 20),
    ]

    matched, unmatched_local, unmatched_broker = match_trades(local, broker)

    assert len(matched) == 2
    assert not unmatched_local
    assert not unmatched_broker


# ---------------------------------------------------------------------------
# Unit tests: slippage
# ---------------------------------------------------------------------------

def test_compute_slippage_zero_when_no_trades() -> None:
    stats = _compute_slippage([], tolerance_pips=2.0)
    assert stats.trade_count == 0
    assert stats.within_tolerance is True


def test_compute_slippage_within_tolerance() -> None:
    # EURUSD: 1 pip = 0.0001; slippage = 0.0002 = 2 pips
    local = _local_trade("EURUSD", 1, 1.1000, 5.0, "2026-01-01T12:00:00")
    broker = _broker_deal("EURUSD", "buy", 1.1002, 4.9, 0)
    stats = _compute_slippage([(local, broker)], tolerance_pips=2.0)
    assert abs(stats.mean_pips - 2.0) < 0.01
    assert stats.within_tolerance is True


def test_compute_slippage_exceeds_tolerance() -> None:
    # slippage = 5 pips, tolerance = 2
    local = _local_trade("EURUSD", 1, 1.1000, 5.0, "2026-01-01T12:00:00")
    broker = _broker_deal("EURUSD", "buy", 1.1005, 4.5, 0)
    stats = _compute_slippage([(local, broker)], tolerance_pips=2.0)
    assert stats.mean_pips > 2.0
    assert stats.within_tolerance is False


# ---------------------------------------------------------------------------
# Unit tests: P&L divergence
# ---------------------------------------------------------------------------

def test_pnl_divergence_no_divergence() -> None:
    local = _local_trade("EURUSD", 1, 1.1000, 5.00, "2026-01-01T12:00:00")
    broker = _broker_deal("EURUSD", "buy", 1.1003, 5.00, 0)
    stats = _compute_pnl_divergence([(local, broker)], tolerance_pct=5.0)
    assert stats.divergence_pct == 0.0
    assert stats.within_tolerance is True


def test_pnl_divergence_exceeds_tolerance() -> None:
    local = _local_trade("EURUSD", 1, 1.1000, 10.00, "2026-01-01T12:00:00")
    broker = _broker_deal("EURUSD", "buy", 1.1003, 8.00, 0)   # 20% divergence
    stats = _compute_pnl_divergence([(local, broker)], tolerance_pct=5.0)
    assert stats.divergence_pct > 5.0
    assert stats.within_tolerance is False


# ---------------------------------------------------------------------------
# Unit tests: fill quality
# ---------------------------------------------------------------------------

def test_fill_quality_clean() -> None:
    deals = [
        {"price": 1.1002, "comment": "", "reason": 0},
        {"price": 1.3000, "comment": "", "reason": 0},
    ]
    stats = _compute_fill_quality(deals)
    assert stats.partial_fills == 0
    assert stats.requotes == 0
    assert stats.rejected_orders == 0
    assert stats.total_deals == 2


def test_fill_quality_partial_and_requote() -> None:
    deals = [
        {"price": 1.1002, "comment": "partial fill", "reason": 0},
        {"price": 1.3000, "comment": "requote", "reason": 0},
    ]
    stats = _compute_fill_quality(deals)
    assert stats.partial_fills == 1
    assert stats.requotes == 1


# ---------------------------------------------------------------------------
# Integration test: run_demo_trade_audit with mock MT5
# ---------------------------------------------------------------------------

class _MockBrokerLifecycleClient:
    """Minimal MT5Client mock that returns pre-loaded snapshot data."""

    def __init__(self, deals: list[dict], connected: bool = True) -> None:
        self._deals = deals
        self._connected = connected

    def connect(self) -> bool:
        return self._connected

    def broker_lifecycle_snapshot(self, symbols: tuple, history_days: int) -> dict:
        return {"deals": self._deals, "connected": self._connected}

    def shutdown(self) -> None:
        pass


def test_run_demo_trade_audit_no_demo_run(tmp_path: Path) -> None:
    """Audit skips gracefully when no demo_dry_resilient run exists."""
    settings = _make_settings(tmp_path)
    code, run_dir, report = run_demo_trade_audit(settings)
    assert code == 0
    assert report.ok is True
    assert (run_dir / "demo_trade_audit.json").exists()


def test_run_demo_trade_audit_matching_trades(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    # Create fake demo_dry_resilient run dir
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    demo_dir = settings.data.runs_dir / "20260101T120000Z_demo_dry_resilient"
    demo_dir.mkdir(parents=True)
    trades = [_local_trade("EURUSD", 1, 1.1000, 5.00, ts.isoformat())]
    _write_closed_trades(demo_dir, trades)

    broker_deals = [_broker_deal("EURUSD", "buy", 1.1002, 5.00, ts.timestamp() + 30)]
    client = _MockBrokerLifecycleClient(broker_deals)

    code, run_dir, report = run_demo_trade_audit(settings, client_factory=lambda: client)

    assert code == 0
    assert report.ok is True
    assert report.fills_compared == 1
    assert report.pnl_divergence.divergence_pct == 0.0


def test_run_demo_trade_audit_unmatched_broker_deal_fails(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    demo_dir = settings.data.runs_dir / "20260101T120000Z_demo_dry_resilient"
    demo_dir.mkdir(parents=True)
    _write_closed_trades(demo_dir, [])   # no local trades

    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    broker_deals = [_broker_deal("EURUSD", "buy", 1.1002, 5.00, ts.timestamp())]
    client = _MockBrokerLifecycleClient(broker_deals)

    code, run_dir, report = run_demo_trade_audit(settings, client_factory=lambda: client)

    assert code == 1       # audit failed
    assert report.ok is False
    assert len(report.unmatched_broker_deals) == 1


def test_run_demo_trade_audit_mt5_disconnected(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    demo_dir = settings.data.runs_dir / "20260101T120000Z_demo_dry_resilient"
    demo_dir.mkdir(parents=True)
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    trades = [_local_trade("EURUSD", 1, 1.1000, 5.00, ts.isoformat())]
    _write_closed_trades(demo_dir, trades)

    client = _MockBrokerLifecycleClient([], connected=False)

    code, run_dir, report = run_demo_trade_audit(settings, client_factory=lambda: client)

    assert code == 2       # connection failed
    assert (run_dir / "demo_trade_audit.json").exists()


def test_run_demo_trade_audit_pnl_divergence_blocks(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    demo_dir = settings.data.runs_dir / "20260101T120000Z_demo_dry_resilient"
    demo_dir.mkdir(parents=True)
    ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    # Local says +$50, broker says +$40 → 20% divergence > 5% tolerance
    trades = [_local_trade("EURUSD", 1, 1.1000, 50.00, ts.isoformat())]
    _write_closed_trades(demo_dir, trades)

    broker_deals = [_broker_deal("EURUSD", "buy", 1.1002, 40.00, ts.timestamp() + 10)]
    client = _MockBrokerLifecycleClient(broker_deals)

    code, run_dir, report = run_demo_trade_audit(settings, client_factory=lambda: client)

    assert code == 1
    assert report.ok is False
    assert report.pnl_divergence.within_tolerance is False
