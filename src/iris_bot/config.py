from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.01
    max_open_positions: int = 4
    min_balance_usd: float = 25.0
    min_confidence: float = 0.55
    atr_stop_loss_multiplier: float = 1.5
    atr_take_profit_multiplier: float = 3.0


@dataclass(frozen=True)
class TradingConfig:
    symbols: tuple[str, ...] = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")
    timeframes: tuple[str, ...] = ("M5", "M15", "H1")
    allow_long: bool = True
    allow_short: bool = True
    one_position_per_symbol: bool = True
    training_window: int = 150


@dataclass(frozen=True)
class MT5Config:
    enabled: bool = False
    login: int | None = None
    password: str | None = None
    server: str | None = None
    path: str | None = None
    history_bars: int = 1500


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def load_settings() -> "Settings":
    project_root = Path(__file__).resolve().parents[2]
    return Settings(
        project_root=project_root,
        data_dir=project_root / "data",
        risk=RiskConfig(
            risk_per_trade=_env_float("IRIS_RISK_PER_TRADE", 0.01),
            max_open_positions=_env_int("IRIS_MAX_OPEN_POSITIONS", 4) or 4,
            min_balance_usd=_env_float("IRIS_MIN_BALANCE_USD", 25.0),
            min_confidence=_env_float("IRIS_MIN_CONFIDENCE", 0.55),
            atr_stop_loss_multiplier=_env_float("IRIS_ATR_SL_MULTIPLIER", 1.5),
            atr_take_profit_multiplier=_env_float("IRIS_ATR_TP_MULTIPLIER", 3.0),
        ),
        trading=TradingConfig(
            symbols=_env_tuple("IRIS_SYMBOLS", ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")),
            timeframes=_env_tuple("IRIS_TIMEFRAMES", ("M5", "M15", "H1")),
            allow_long=_env_bool("IRIS_ALLOW_LONG", True),
            allow_short=_env_bool("IRIS_ALLOW_SHORT", True),
            one_position_per_symbol=_env_bool("IRIS_ONE_POSITION_PER_SYMBOL", True),
            training_window=_env_int("IRIS_TRAINING_WINDOW", 150) or 150,
        ),
        mt5=MT5Config(
            enabled=_env_bool("IRIS_MT5_ENABLED", False),
            login=_env_int("IRIS_MT5_LOGIN", None),
            password=os.getenv("IRIS_MT5_PASSWORD"),
            server=os.getenv("IRIS_MT5_SERVER"),
            path=os.getenv("IRIS_MT5_PATH"),
            history_bars=_env_int("IRIS_MT5_HISTORY_BARS", 1500) or 1500,
        ),
    )


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    mt5: MT5Config = field(default_factory=MT5Config)


settings = load_settings()
