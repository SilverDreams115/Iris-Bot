from __future__ import annotations

from dataclasses import dataclass

from iris_bot.config import RiskConfig, TradingConfig
from iris_bot.features import FeatureVector
from iris_bot.risk import calculate_position_size
from iris_bot.strategy import build_signal, train_model


@dataclass(frozen=True)
class BacktestTrade:
    symbol: str
    timeframe: str
    direction: int
    confidence: float
    pnl: float


@dataclass(frozen=True)
class BacktestResult:
    starting_balance: float
    ending_balance: float
    trades: list[BacktestTrade]

    @property
    def total_return(self) -> float:
        return 0.0 if self.starting_balance == 0.0 else (self.ending_balance - self.starting_balance) / self.starting_balance

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for trade in self.trades if trade.pnl > 0.0)
        return wins / len(self.trades)


def run_backtest(
    rows: list[FeatureVector],
    starting_balance: float,
    risk: RiskConfig,
    trading: TradingConfig,
) -> BacktestResult:
    if len(rows) <= trading.training_window:
        return BacktestResult(starting_balance=starting_balance, ending_balance=starting_balance, trades=[])

    balance = starting_balance
    trades: list[BacktestTrade] = []
    open_symbols: set[str] = set()
    model = train_model(rows[: trading.training_window])

    for row in rows[trading.training_window :]:
        signal = build_signal(row, model, risk, trading)
        if signal is None:
            continue
        if trading.one_position_per_symbol and signal.symbol in open_symbols:
            continue
        if len(open_symbols) >= risk.max_open_positions:
            continue

        stop_distance = max(signal.atr_3 * signal.close * risk.atr_stop_loss_multiplier, signal.close * 0.001)
        target_distance = max(signal.atr_3 * signal.close * risk.atr_take_profit_multiplier, signal.close * 0.002)
        stop_price = signal.close - stop_distance if signal.direction == 1 else signal.close + stop_distance
        size = calculate_position_size(
            balance=balance,
            risk_per_trade=risk.risk_per_trade,
            entry_price=signal.close,
            stop_loss_price=stop_price,
        )
        if size == 0.0:
            continue

        open_symbols.add(signal.symbol)
        next_move = row.next_return * signal.close
        pnl = size * (next_move if signal.direction == 1 else -next_move)
        pnl = max(min(pnl, target_distance * size), -stop_distance * size)
        balance += pnl
        trades.append(
            BacktestTrade(
                symbol=signal.symbol,
                timeframe=signal.timeframe,
                direction=signal.direction,
                confidence=signal.confidence,
                pnl=pnl,
            )
        )
        open_symbols.remove(signal.symbol)

    return BacktestResult(
        starting_balance=starting_balance,
        ending_balance=balance,
        trades=trades,
    )
