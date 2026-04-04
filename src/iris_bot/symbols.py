from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from iris_bot.artifacts import read_artifact_payload, wrap_artifact
from iris_bot.config import Settings
from iris_bot.exits import SymbolExitProfile
from iris_bot.operational import atomic_write_json
from iris_bot.sessions import canonical_session_name


@dataclass(frozen=True)
class SymbolStrategyProfile:
    symbol: str
    enabled_state: str
    enabled: bool
    allowed_timeframes: tuple[str, ...]
    allowed_sessions: tuple[str, ...]
    threshold: float
    allow_long: bool
    allow_short: bool
    risk_multiplier: float
    max_open_positions: int
    stop_policy: str
    target_policy: str
    stop_atr_multiplier: float
    target_atr_multiplier: float
    stop_min_pct: float | None
    stop_max_pct: float | None
    target_min_pct: float | None
    target_max_pct: float | None
    no_trade_min_expectancy_usd: float
    regime_filter: str = "off"
    notes: str = ""
    profile_id: str = ""
    model_variant: str = "global_model"
    source_run_id: str = ""
    promotion_state: str = "candidate"
    promotion_reason: str = ""
    rollback_target: str | None = None

    @property
    def exit_profile(self) -> SymbolExitProfile:
        return SymbolExitProfile(
            stop_policy=self.stop_policy,
            target_policy=self.target_policy,
            stop_atr_multiplier=self.stop_atr_multiplier,
            target_atr_multiplier=self.target_atr_multiplier,
            stop_min_pct=self.stop_min_pct,
            stop_max_pct=self.stop_max_pct,
            target_min_pct=self.target_min_pct,
            target_max_pct=self.target_max_pct,
        )


def strategy_profiles_path(settings: Settings) -> Path:
    return settings.data.runtime_dir / settings.strategy.profiles_filename


def default_symbol_strategy_profile(settings: Settings, symbol: str) -> SymbolStrategyProfile:
    return SymbolStrategyProfile(
        symbol=symbol,
        enabled_state="enabled",
        enabled=True,
        allowed_timeframes=(settings.trading.primary_timeframe,),
        allowed_sessions=("asia", "london", "new_york"),
        threshold=max(settings.risk.min_confidence, min(settings.threshold.grid)),
        allow_long=settings.trading.allow_long,
        allow_short=settings.trading.allow_short,
        risk_multiplier=1.0,
        max_open_positions=settings.risk.max_open_positions,
        stop_policy=settings.exit_policy.stop_policy,
        target_policy=settings.exit_policy.target_policy,
        stop_atr_multiplier=settings.risk.atr_stop_loss_multiplier,
        target_atr_multiplier=settings.risk.atr_take_profit_multiplier,
        stop_min_pct=settings.dynamic_exits.min_stop_loss_pct,
        stop_max_pct=settings.dynamic_exits.max_stop_loss_pct,
        target_min_pct=settings.dynamic_exits.min_take_profit_pct,
        target_max_pct=settings.dynamic_exits.max_take_profit_pct,
        no_trade_min_expectancy_usd=settings.strategy.min_expectancy_usd,
        notes="default_common_profile",
        profile_id="",
        model_variant="global_model",
        source_run_id="",
        promotion_state="candidate",
        promotion_reason="default_profile",
        rollback_target=None,
    )


def _merge_profile(default_profile: SymbolStrategyProfile, payload: dict[str, Any]) -> SymbolStrategyProfile:
    merged = asdict(default_profile)
    merged.update(payload)
    merged["allowed_timeframes"] = tuple(merged["allowed_timeframes"])
    merged["allowed_sessions"] = tuple(merged["allowed_sessions"])
    return SymbolStrategyProfile(**merged)


def load_symbol_strategy_profiles(settings: Settings) -> dict[str, SymbolStrategyProfile]:
    defaults = {
        symbol: default_symbol_strategy_profile(settings, symbol)
        for symbol in settings.trading.symbols
    }
    path = strategy_profiles_path(settings)
    if not path.exists():
        return defaults
    payload = read_artifact_payload(path, expected_type="strategy_profiles")
    common_overrides = payload.get("common", {})
    symbol_overrides = payload.get("symbols", {})
    resolved: dict[str, SymbolStrategyProfile] = {}
    for symbol, default_profile in defaults.items():
        merged = _merge_profile(default_profile, common_overrides)
        if symbol in symbol_overrides:
            merged = _merge_profile(merged, symbol_overrides[symbol])
        resolved[symbol] = merged
    return resolved


def write_symbol_strategy_profiles(
    settings: Settings,
    common_profile: dict[str, Any],
    symbol_profiles: dict[str, dict[str, Any]],
) -> Path:
    path = strategy_profiles_path(settings)
    atomic_write_json(
        path,
        wrap_artifact(
            "strategy_profiles",
            {
                "common": common_profile,
                "symbols": symbol_profiles,
            },
            compatibility={"loader": "load_symbol_strategy_profiles"},
        ),
    )
    return path


def session_name(timestamp: datetime) -> str:
    return canonical_session_name(timestamp)


def row_allowed_by_profile(profile: SymbolStrategyProfile | None, timestamp: datetime, timeframe: str) -> bool:
    if profile is None:
        return True
    if timeframe not in profile.allowed_timeframes:
        return False
    if not profile.allowed_sessions:
        return True
    return session_name(timestamp) in set(profile.allowed_sessions)


def profile_trace(profile: SymbolStrategyProfile | None) -> dict[str, Any]:
    if profile is None:
        return {
            "active_profile_id": "",
            "model_variant": "global_model",
            "profile_source_run_id": "",
            "enablement_state": "unknown",
            "promotion_state": "unknown",
            "promotion_reason": "",
        }
    return {
        "active_profile_id": profile.profile_id,
        "model_variant": profile.model_variant,
        "profile_source_run_id": profile.source_run_id,
        "enablement_state": profile.enabled_state,
        "promotion_state": profile.promotion_state,
        "promotion_reason": profile.promotion_reason,
    }
