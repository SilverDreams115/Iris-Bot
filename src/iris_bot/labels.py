from __future__ import annotations

from dataclasses import dataclass

from iris_bot.config import LabelingConfig
from iris_bot.data import Bar


@dataclass(frozen=True)
class LabelOutcome:
    label: int
    label_reason: str
    horizon_end_timestamp: str


def next_bar_direction_label(series: list[Bar], index: int, config: LabelingConfig) -> LabelOutcome | None:
    if index + 1 >= len(series):
        return None

    current = series[index]
    next_bar = series[index + 1]
    move = 0.0 if current.close == 0.0 else (next_bar.close - current.close) / current.close

    if abs(move) < config.min_abs_return and config.allow_no_trade:
        label = 0
        reason = "next_bar_small_move"
    else:
        label = 1 if move > 0.0 else -1
        reason = "next_bar_direction"

    return LabelOutcome(label=label, label_reason=reason, horizon_end_timestamp=next_bar.timestamp.isoformat())


def triple_barrier_label(series: list[Bar], index: int, config: LabelingConfig) -> LabelOutcome | None:
    current = series[index]
    if index + config.horizon_bars >= len(series):
        return None

    upper_barrier = current.close * (1.0 + config.take_profit_pct)
    lower_barrier = current.close * (1.0 - config.stop_loss_pct)
    horizon_slice = series[index + 1 : index + config.horizon_bars + 1]

    for bar in horizon_slice:
        long_hit = bar.high >= upper_barrier
        short_hit = bar.low <= lower_barrier
        if long_hit and short_hit:
            label = 0 if config.allow_no_trade else 1
            reason = "triple_barrier_ambiguous"
            return LabelOutcome(label=label, label_reason=reason, horizon_end_timestamp=bar.timestamp.isoformat())
        if long_hit:
            return LabelOutcome(label=1, label_reason="triple_barrier_take_profit", horizon_end_timestamp=bar.timestamp.isoformat())
        if short_hit:
            return LabelOutcome(label=-1, label_reason="triple_barrier_stop_loss", horizon_end_timestamp=bar.timestamp.isoformat())

    last_bar = horizon_slice[-1]
    terminal_move = 0.0 if current.close == 0.0 else (last_bar.close - current.close) / current.close
    timeout_direction_floor = max(
        config.min_abs_return,
        min(config.take_profit_pct, config.stop_loss_pct) * config.timeout_direction_min_barrier_fraction,
    )
    if (
        config.allow_no_trade
        and config.timeout_handling_mode == "neutral_by_barrier_fraction"
        and abs(terminal_move) < timeout_direction_floor
    ):
        return LabelOutcome(
            label=0,
            label_reason="triple_barrier_timeout_filtered_small_move",
            horizon_end_timestamp=last_bar.timestamp.isoformat(),
        )
    if abs(terminal_move) < config.min_abs_return and config.allow_no_trade:
        label = 0
        reason = "triple_barrier_timeout_small_move"
    else:
        label = 1 if terminal_move > 0.0 else -1
        reason = "triple_barrier_timeout_direction"
    return LabelOutcome(label=label, label_reason=reason, horizon_end_timestamp=last_bar.timestamp.isoformat())


def build_label(series: list[Bar], index: int, config: LabelingConfig) -> LabelOutcome | None:
    if config.mode == "next_bar_direction":
        return next_bar_direction_label(series, index, config)
    if config.mode == "triple_barrier":
        return triple_barrier_label(series, index, config)
    raise ValueError(f"Unsupported labeling mode: {config.mode}")
