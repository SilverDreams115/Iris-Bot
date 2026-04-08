"""
Demo trade audit module.

Compares local closed_trades (from the latest demo_dry_resilient run) against
the actual deal history pulled from MT5. Detects:
  - Fill slippage: local intended entry price vs actual MT5 fill price
  - P&L divergence: local net_pnl_usd vs MT5 deal profit
  - Unmatched trades: positions that exist in one side but not the other
  - Fill quality issues: partial fills, requotes, rejected orders

Audit result is written to demo_trade_audit.json in the run directory.
The governance validation layer reads this file before approving a profile.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from iris_bot.artifacts import wrap_artifact
from iris_bot.config import Settings
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.mt5 import MT5Client


# ---------------------------------------------------------------------------
# Pip value helpers
# ---------------------------------------------------------------------------

def _pip_value(symbol: str) -> float:
    """Return the pip size for a symbol. JPY pairs use 0.01; all others 0.0001."""
    return 0.01 if "JPY" in symbol.upper() else 0.0001


def _price_to_pips(price_diff: float, symbol: str) -> float:
    pip = _pip_value(symbol)
    return abs(price_diff) / pip if pip > 0 else 0.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SlippageStats:
    mean_pips: float
    max_pips: float
    within_tolerance: bool
    trade_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PnlDivergenceStats:
    local_pnl_usd: float
    broker_pnl_usd: float
    divergence_pct: float     # |local - broker| / max(|broker|, 1) * 100
    within_tolerance: bool
    matched_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FillQualityStats:
    partial_fills: int
    requotes: int
    rejected_orders: int
    total_deals: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DemoTradeAuditReport:
    run_timestamp: str
    fills_compared: int
    slippage: SlippageStats
    pnl_divergence: PnlDivergenceStats
    fill_quality: FillQualityStats
    unmatched_broker_deals: list[dict[str, Any]]
    unmatched_local_trades: list[dict[str, Any]]
    history_days: int
    source_run: str

    @property
    def ok(self) -> bool:
        return (
            self.pnl_divergence.within_tolerance
            and self.slippage.within_tolerance
            and len(self.unmatched_broker_deals) == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_timestamp": self.run_timestamp,
            "ok": self.ok,
            "fills_compared": self.fills_compared,
            "slippage": self.slippage.to_dict(),
            "pnl_divergence": self.pnl_divergence.to_dict(),
            "fill_quality": self.fill_quality.to_dict(),
            "unmatched_broker_deals": self.unmatched_broker_deals,
            "unmatched_local_trades": self.unmatched_local_trades,
            "history_days": self.history_days,
            "source_run": self.source_run,
        }


# ---------------------------------------------------------------------------
# Trade loading helpers
# ---------------------------------------------------------------------------

def _load_local_closed_trades(run_dir: Path) -> list[dict[str, Any]]:
    """Read closed_trades.csv from a run directory."""
    path = run_dir / "closed_trades.csv"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(dict(row))
    return rows


def _latest_demo_run(settings: Settings) -> Path | None:
    candidates = sorted(settings.data.runs_dir.glob("*_demo_dry_resilient"))
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Trade matching
# ---------------------------------------------------------------------------

def _entry_timestamp_unix(trade: dict[str, Any]) -> float | None:
    ts_str = trade.get("entry_timestamp", "")
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(str(ts_str))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _deal_timestamp_unix(deal: dict[str, Any]) -> float | None:
    raw = deal.get("time")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return None


def _deal_side(deal: dict[str, Any]) -> str:
    """MT5 deal entry type: 0 = buy, 1 = sell."""
    try:
        deal_type = int(deal.get("type", 0))
        entry = int(deal.get("entry", 0))   # 0 = entry, 1 = exit
    except (TypeError, ValueError):
        return "unknown"
    if entry != 0:
        return "close"
    return "buy" if deal_type == 0 else "sell"


def _local_side(trade: dict[str, Any]) -> str:
    try:
        direction = int(trade.get("direction", 1))
    except (TypeError, ValueError):
        return "unknown"
    return "buy" if direction == 1 else "sell"


def match_trades(
    local_trades: list[dict[str, Any]],
    broker_deals: list[dict[str, Any]],
    timestamp_tolerance_seconds: int = 300,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedy matching of local trades vs broker deals by symbol + side + timestamp.

    Returns:
        matched: list of (local_trade, broker_deal) pairs
        unmatched_local: local trades with no broker counterpart
        unmatched_broker: broker deals with no local counterpart
    """
    # Only match entry deals (entry == 0)
    entry_deals = [d for d in broker_deals if _deal_side(d) not in ("close", "unknown")]

    used_broker: set[int] = set()
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    unmatched_local: list[dict[str, Any]] = []

    for local in local_trades:
        local_symbol = str(local.get("symbol", ""))
        local_side = _local_side(local)
        local_ts = _entry_timestamp_unix(local)
        if local_ts is None:
            unmatched_local.append(local)
            continue

        best_idx: int | None = None
        best_dist = float("inf")
        for i, deal in enumerate(entry_deals):
            if i in used_broker:
                continue
            if str(deal.get("symbol", "")) != local_symbol:
                continue
            if _deal_side(deal) != local_side:
                continue
            deal_ts = _deal_timestamp_unix(deal)
            if deal_ts is None:
                continue
            dist = abs(deal_ts - local_ts)
            if dist <= timestamp_tolerance_seconds and dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None:
            used_broker.add(best_idx)
            matched.append((local, entry_deals[best_idx]))
        else:
            unmatched_local.append(local)

    unmatched_broker = [deal for i, deal in enumerate(entry_deals) if i not in used_broker]
    return matched, unmatched_local, unmatched_broker


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_slippage(
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
    tolerance_pips: float,
) -> SlippageStats:
    if not matched:
        return SlippageStats(0.0, 0.0, True, 0)
    slippages: list[float] = []
    for local, deal in matched:
        try:
            local_price = float(local.get("entry_price", 0.0))
            deal_price = float(deal.get("price", 0.0))
            symbol = str(local.get("symbol", ""))
            if local_price > 0 and deal_price > 0:
                slippages.append(_price_to_pips(deal_price - local_price, symbol))
        except (TypeError, ValueError):
            continue
    if not slippages:
        return SlippageStats(0.0, 0.0, True, 0)
    mean_pips = sum(slippages) / len(slippages)
    max_pips = max(slippages)
    return SlippageStats(
        mean_pips=round(mean_pips, 4),
        max_pips=round(max_pips, 4),
        within_tolerance=mean_pips <= tolerance_pips,
        trade_count=len(slippages),
    )


def _compute_pnl_divergence(
    matched: list[tuple[dict[str, Any], dict[str, Any]]],
    tolerance_pct: float,
) -> PnlDivergenceStats:
    if not matched:
        return PnlDivergenceStats(0.0, 0.0, 0.0, True, 0)
    local_total = 0.0
    broker_total = 0.0
    count = 0
    for local, deal in matched:
        try:
            local_pnl = float(local.get("net_pnl_usd", 0.0))
            broker_pnl = float(deal.get("profit", 0.0))
            local_total += local_pnl
            broker_total += broker_pnl
            count += 1
        except (TypeError, ValueError):
            continue
    denom = max(abs(broker_total), 1.0)
    divergence_pct = abs(local_total - broker_total) / denom * 100.0
    return PnlDivergenceStats(
        local_pnl_usd=round(local_total, 4),
        broker_pnl_usd=round(broker_total, 4),
        divergence_pct=round(divergence_pct, 2),
        within_tolerance=divergence_pct <= tolerance_pct,
        matched_count=count,
    )


def _compute_fill_quality(broker_deals: list[dict[str, Any]]) -> FillQualityStats:
    partial = 0
    requotes = 0
    rejected = 0
    for deal in broker_deals:
        comment = str(deal.get("comment", "")).lower()
        retcode = int(deal.get("reason", 0))
        if "partial" in comment:
            partial += 1
        if "requote" in comment or retcode == 3:
            requotes += 1
        if not deal.get("price") or float(deal.get("price", 0.0)) == 0.0:
            rejected += 1
    return FillQualityStats(
        partial_fills=partial,
        requotes=requotes,
        rejected_orders=rejected,
        total_deals=len(broker_deals),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_demo_trade_audit(
    settings: Settings,
    client_factory: Callable[[], MT5Client] | None = None,
) -> tuple[int, Path, DemoTradeAuditReport]:
    """Run the demo trade audit and write demo_trade_audit.json to a new run directory.

    Returns:
        (exit_code, run_dir, report)
        exit_code: 0=ok, 1=audit_failed, 2=connection_failed
    """
    run_dir = build_run_directory(settings.data.runs_dir, "demo_trade_audit")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    audit_cfg = settings.demo_audit
    now = datetime.now(tz=UTC).isoformat()

    source_run = _latest_demo_run(settings)
    if source_run is None:
        report = DemoTradeAuditReport(
            run_timestamp=now,
            fills_compared=0,
            slippage=SlippageStats(0.0, 0.0, True, 0),
            pnl_divergence=PnlDivergenceStats(0.0, 0.0, 0.0, True, 0),
            fill_quality=FillQualityStats(0, 0, 0, 0),
            unmatched_broker_deals=[],
            unmatched_local_trades=[],
            history_days=audit_cfg.history_days,
            source_run="none",
        )
        write_json_report(run_dir, "demo_trade_audit.json", wrap_artifact("demo_trade_audit", report.to_dict()))
        logger.warning("No demo_dry_resilient run found; audit skipped")
        return 0, run_dir, report

    local_trades = _load_local_closed_trades(source_run)

    client = client_factory() if client_factory is not None else MT5Client(settings.mt5)
    connected = client.connect()
    if not connected:
        report = DemoTradeAuditReport(
            run_timestamp=now,
            fills_compared=0,
            slippage=SlippageStats(0.0, 0.0, True, 0),
            pnl_divergence=PnlDivergenceStats(0.0, 0.0, 0.0, True, 0),
            fill_quality=FillQualityStats(0, 0, 0, 0),
            unmatched_broker_deals=[],
            unmatched_local_trades=[{"reason": "mt5_not_connected"} for _ in local_trades],
            history_days=audit_cfg.history_days,
            source_run=source_run.name,
        )
        write_json_report(run_dir, "demo_trade_audit.json", wrap_artifact("demo_trade_audit", report.to_dict()))
        logger.error("MT5 connection failed; cannot audit broker deal history")
        client.shutdown()
        return 2, run_dir, report

    broker_snapshot = client.broker_lifecycle_snapshot(settings.trading.symbols, audit_cfg.history_days)
    broker_deals: list[dict[str, Any]] = broker_snapshot.get("deals", [])

    matched, unmatched_local, unmatched_broker = match_trades(
        local_trades,
        broker_deals,
        audit_cfg.timestamp_tolerance_seconds,
    )

    slippage = _compute_slippage(matched, audit_cfg.slippage_tolerance_pips)
    pnl_div = _compute_pnl_divergence(matched, audit_cfg.pnl_divergence_tolerance_pct)
    fill_quality = _compute_fill_quality(broker_deals)

    report = DemoTradeAuditReport(
        run_timestamp=now,
        fills_compared=len(matched),
        slippage=slippage,
        pnl_divergence=pnl_div,
        fill_quality=fill_quality,
        unmatched_broker_deals=unmatched_broker,
        unmatched_local_trades=unmatched_local,
        history_days=audit_cfg.history_days,
        source_run=source_run.name,
    )

    write_json_report(run_dir, "demo_trade_audit.json", wrap_artifact("demo_trade_audit", report.to_dict()))
    client.shutdown()

    status = "ok" if report.ok else "issues_found"
    logger.info(
        "demo_trade_audit %s matched=%d slippage_mean=%.2f pips pnl_div=%.1f%% unmatched_broker=%d",
        status,
        len(matched),
        slippage.mean_pips,
        pnl_div.divergence_pct,
        len(unmatched_broker),
    )
    return (0 if report.ok else 1), run_dir, report


def load_latest_demo_audit(settings: Settings) -> dict[str, Any] | None:
    """Load the most recent demo_trade_audit.json report payload. Returns None if not found."""
    candidates = sorted(settings.data.runs_dir.glob("*_demo_trade_audit"))
    if not candidates:
        return None
    path = candidates[-1] / "demo_trade_audit.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        payload = raw.get("payload", raw)
        return payload if isinstance(payload, dict) else None
    except (json.JSONDecodeError, OSError):
        return None
