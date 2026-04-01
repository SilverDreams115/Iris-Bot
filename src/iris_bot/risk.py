from __future__ import annotations


def calculate_position_size(
    balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float,
    pip_value: float = 1.0,
    min_lot: float = 0.01,
    lot_step: float = 0.01,
) -> float:
    risk_amount = max(balance, 0.0) * max(risk_per_trade, 0.0)
    stop_distance = abs(entry_price - stop_loss_price)
    if risk_amount == 0.0 or stop_distance == 0.0 or pip_value <= 0.0:
        return 0.0

    raw_size = risk_amount / (stop_distance * pip_value)
    stepped = int(raw_size / lot_step) * lot_step
    return max(min_lot, round(stepped, 2)) if stepped > 0.0 else 0.0
