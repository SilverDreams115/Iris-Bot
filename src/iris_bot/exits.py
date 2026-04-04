from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from iris_bot.config import BacktestConfig, DynamicExitConfig, RiskConfig
from iris_bot.processed_dataset import ProcessedRow

# ATR feature names used for dynamic exit calculations.
# Update here if the feature schema changes.
ATR_FEATURE_LONG = "atr_10"
ATR_FEATURE_SHORT = "atr_5"


@dataclass(frozen=True)
class ExitLevel:
    price: float
    distance: float
    details: dict[str, Any]


@dataclass(frozen=True)
class SymbolExitProfile:
    stop_policy: str = "static"
    target_policy: str = "static"
    stop_atr_multiplier: float = 1.5
    target_atr_multiplier: float = 3.0
    stop_min_pct: float | None = None
    stop_max_pct: float | None = None
    target_min_pct: float | None = None
    target_max_pct: float | None = None


class StopPolicy(ABC):
    name = "static"

    @abstractmethod
    def stop_loss_price(
        self,
        row: ProcessedRow,
        entry_price: float,
        direction: int,
        backtest: BacktestConfig,
        risk: RiskConfig,
        dynamic_config: DynamicExitConfig,
        symbol_profile: SymbolExitProfile | None = None,
    ) -> ExitLevel: ...


class TargetPolicy(ABC):
    name = "static"

    @abstractmethod
    def take_profit_price(
        self,
        row: ProcessedRow,
        entry_price: float,
        direction: int,
        backtest: BacktestConfig,
        risk: RiskConfig,
        dynamic_config: DynamicExitConfig,
        symbol_profile: SymbolExitProfile | None = None,
    ) -> ExitLevel: ...


def _atr_fraction(row: ProcessedRow) -> float:
    return max(
        float(row.features.get(ATR_FEATURE_LONG, 0.0)),
        float(row.features.get(ATR_FEATURE_SHORT, 0.0)),
        0.0,
    )


def _volatility_fraction(row: ProcessedRow) -> float:
    return max(
        float(row.features.get("rolling_volatility_10", 0.0)),
        float(row.features.get("rolling_volatility_5", 0.0)),
        0.0,
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _resolve_bounds(
    minimum: float | None,
    maximum: float | None,
    default_minimum: float,
    default_maximum: float,
) -> tuple[float, float]:
    return (
        default_minimum if minimum is None else minimum,
        default_maximum if maximum is None else maximum,
    )


class StaticStopPolicy(StopPolicy):
    def stop_loss_price(
        self,
        row: ProcessedRow,
        entry_price: float,
        direction: int,
        backtest: BacktestConfig,
        risk: RiskConfig,
        dynamic_config: DynamicExitConfig,
        symbol_profile: SymbolExitProfile | None = None,
    ) -> ExitLevel:
        del dynamic_config, symbol_profile
        if backtest.use_atr_stops:
            atr_distance = _atr_fraction(row) * entry_price * risk.atr_stop_loss_multiplier
            floor_distance = entry_price * backtest.fixed_stop_loss_pct
            distance = max(atr_distance, floor_distance)
        else:
            atr_distance = 0.0
            floor_distance = entry_price * backtest.fixed_stop_loss_pct
            distance = floor_distance
        price = entry_price - distance if direction == 1 else entry_price + distance
        return ExitLevel(
            price=price,
            distance=distance,
            details={
                "policy": self.name,
                "entry_price": entry_price,
                "atr_fraction": _atr_fraction(row),
                "atr_distance": atr_distance,
                "floor_distance": floor_distance,
                "final_distance": distance,
                "direction": direction,
            },
        )


class StaticTargetPolicy(TargetPolicy):
    def take_profit_price(
        self,
        row: ProcessedRow,
        entry_price: float,
        direction: int,
        backtest: BacktestConfig,
        risk: RiskConfig,
        dynamic_config: DynamicExitConfig,
        symbol_profile: SymbolExitProfile | None = None,
    ) -> ExitLevel:
        del dynamic_config, symbol_profile
        if backtest.use_atr_stops:
            atr_distance = _atr_fraction(row) * entry_price * risk.atr_take_profit_multiplier
            floor_distance = entry_price * backtest.fixed_take_profit_pct
            distance = max(atr_distance, floor_distance)
        else:
            atr_distance = 0.0
            floor_distance = entry_price * backtest.fixed_take_profit_pct
            distance = floor_distance
        price = entry_price + distance if direction == 1 else entry_price - distance
        return ExitLevel(
            price=price,
            distance=distance,
            details={
                "policy": self.name,
                "entry_price": entry_price,
                "atr_fraction": _atr_fraction(row),
                "atr_distance": atr_distance,
                "floor_distance": floor_distance,
                "final_distance": distance,
                "direction": direction,
            },
        )


class ATRDynamicStopPolicy(StopPolicy):
    name = "atr_dynamic"

    def stop_loss_price(
        self,
        row: ProcessedRow,
        entry_price: float,
        direction: int,
        backtest: BacktestConfig,
        risk: RiskConfig,
        dynamic_config: DynamicExitConfig,
        symbol_profile: SymbolExitProfile | None = None,
    ) -> ExitLevel:
        profile = symbol_profile or SymbolExitProfile()
        atr_fraction = _atr_fraction(row)
        volatility_fraction = _volatility_fraction(row)
        volatility_adjustment = 1.0 + min(0.75, volatility_fraction * dynamic_config.volatility_adjustment_scale)
        multiplier = profile.stop_atr_multiplier
        raw_distance = entry_price * atr_fraction * multiplier * volatility_adjustment
        floor_distance = entry_price * backtest.fixed_stop_loss_pct
        min_pct, max_pct = _resolve_bounds(
            profile.stop_min_pct,
            profile.stop_max_pct,
            dynamic_config.min_stop_loss_pct,
            dynamic_config.max_stop_loss_pct,
        )
        min_distance = entry_price * min_pct
        max_distance = entry_price * max_pct
        distance = _clamp(max(raw_distance, floor_distance), min_distance, max_distance)
        price = entry_price - distance if direction == 1 else entry_price + distance
        return ExitLevel(
            price=price,
            distance=distance,
            details={
                "policy": self.name,
                "entry_price": entry_price,
                "direction": direction,
                "atr_fraction": atr_fraction,
                "volatility_fraction": volatility_fraction,
                "volatility_adjustment": volatility_adjustment,
                "atr_multiplier": multiplier,
                "raw_distance": raw_distance,
                "floor_distance": floor_distance,
                "min_distance": min_distance,
                "max_distance": max_distance,
                "final_distance": distance,
            },
        )


class ATRDynamicTargetPolicy(TargetPolicy):
    name = "atr_dynamic"

    def take_profit_price(
        self,
        row: ProcessedRow,
        entry_price: float,
        direction: int,
        backtest: BacktestConfig,
        risk: RiskConfig,
        dynamic_config: DynamicExitConfig,
        symbol_profile: SymbolExitProfile | None = None,
    ) -> ExitLevel:
        profile = symbol_profile or SymbolExitProfile()
        atr_fraction = _atr_fraction(row)
        volatility_fraction = _volatility_fraction(row)
        volatility_adjustment = 1.0 + min(0.75, volatility_fraction * dynamic_config.volatility_adjustment_scale)
        multiplier = profile.target_atr_multiplier
        raw_distance = entry_price * atr_fraction * multiplier * volatility_adjustment
        floor_distance = entry_price * backtest.fixed_take_profit_pct
        min_pct, max_pct = _resolve_bounds(
            profile.target_min_pct,
            profile.target_max_pct,
            dynamic_config.min_take_profit_pct,
            dynamic_config.max_take_profit_pct,
        )
        min_distance = entry_price * min_pct
        max_distance = entry_price * max_pct
        distance = _clamp(max(raw_distance, floor_distance), min_distance, max_distance)
        price = entry_price + distance if direction == 1 else entry_price - distance
        return ExitLevel(
            price=price,
            distance=distance,
            details={
                "policy": self.name,
                "entry_price": entry_price,
                "direction": direction,
                "atr_fraction": atr_fraction,
                "volatility_fraction": volatility_fraction,
                "volatility_adjustment": volatility_adjustment,
                "atr_multiplier": multiplier,
                "raw_distance": raw_distance,
                "floor_distance": floor_distance,
                "min_distance": min_distance,
                "max_distance": max_distance,
                "final_distance": distance,
            },
        )


def build_exit_policies(
    stop_policy_name: str,
    target_policy_name: str,
) -> tuple[StopPolicy, TargetPolicy]:
    stop_policy_map: dict[str, StopPolicy] = {
        "static": StaticStopPolicy(),
        "atr_dynamic": ATRDynamicStopPolicy(),
    }
    target_policy_map: dict[str, TargetPolicy] = {
        "static": StaticTargetPolicy(),
        "atr_dynamic": ATRDynamicTargetPolicy(),
    }
    if stop_policy_name not in stop_policy_map:
        raise ValueError(f"Unsupported stop policy: {stop_policy_name}")
    if target_policy_name not in target_policy_map:
        raise ValueError(f"Unsupported target policy: {target_policy_name}")
    return stop_policy_map[stop_policy_name], target_policy_map[target_policy_name]
