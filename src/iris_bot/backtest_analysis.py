from __future__ import annotations

from typing import Protocol, Sequence

from iris_bot.config import BacktestConfig
from iris_bot.risk import ForexInstrument, realized_pnl_usd
from iris_bot.processed_dataset import ProcessedRow
from iris_bot.xgb_model import XGBoostMultiClassModel

from iris_bot.backtest_pricing import commission_usd, exit_execution_price


class MarkToMarketPosition(Protocol):
    @property
    def direction(self) -> int: ...

    @property
    def entry_price(self) -> float: ...

    @property
    def volume_lots(self) -> float: ...

    @property
    def commission_entry_usd(self) -> float: ...


class TradeMetricRecord(Protocol):
    @property
    def net_pnl_usd(self) -> float: ...


class EquityLikePoint(Protocol):
    @property
    def equity(self) -> float: ...

    @property
    def open_positions(self) -> int: ...


def compute_signal_probabilities(
    model: XGBoostMultiClassModel,
    rows: list[ProcessedRow],
    feature_names: list[str],
) -> list[dict[int, float]]:
    matrix = [[row.features[name] for name in feature_names] for row in rows]
    return model.predict_probabilities(matrix)


def mark_to_market(
    position: MarkToMarketPosition,
    bar: ProcessedRow,
    instrument: ForexInstrument,
    config: BacktestConfig,
    aux_rates: dict[str, float] | None = None,
) -> float:
    exit_price = exit_execution_price(bar.close, position.direction, instrument, config)
    gross_pnl = realized_pnl_usd(instrument, position.entry_price, exit_price, position.direction, position.volume_lots, aux_rates)
    commission_exit = commission_usd(position.volume_lots, config)
    return gross_pnl - position.commission_entry_usd - commission_exit


def trade_metrics(
    trades: Sequence[TradeMetricRecord],
    starting_balance: float,
    ending_balance: float,
    equity_curve: Sequence[EquityLikePoint],
    total_steps: int,
) -> dict[str, object]:
    total_trades = len(trades)
    wins = [t for t in trades if t.net_pnl_usd > 0.0]
    losses = [t for t in trades if t.net_pnl_usd < 0.0]
    gross_profit = sum(t.net_pnl_usd for t in wins)
    gross_loss = -sum(t.net_pnl_usd for t in losses)
    net_pnl = ending_balance - starting_balance
    average_pnl = 0.0 if total_trades == 0 else net_pnl / total_trades
    profit_factor = gross_profit / gross_loss if gross_loss > 0.0 else (999.0 if gross_profit > 0.0 else 0.0)
    win_rate = 0.0 if total_trades == 0 else len(wins) / total_trades

    peak = starting_balance
    max_drawdown = 0.0
    for point in equity_curve:
        peak = max(peak, point.equity)
        drawdown = peak - point.equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    exposure = 0.0 if total_steps == 0 else sum(1 for pt in equity_curve if pt.open_positions > 0) / total_steps
    return {
        "starting_balance_usd": starting_balance,
        "ending_balance_usd": ending_balance,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "net_pnl_usd": net_pnl,
        "average_pnl_usd": average_pnl,
        "expectancy_usd": average_pnl,
        "max_drawdown_usd": max_drawdown,
        "profit_factor": profit_factor,
        "exposure": exposure,
        "return_pct": 0.0 if starting_balance == 0.0 else net_pnl / starting_balance,
    }
