from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from iris_bot.config import BacktestConfig, DynamicExitConfig, RiskConfig
    from iris_bot.exits import SymbolExitProfile
    from iris_bot.operational import ExitPolicyConfig, PaperEngineState
    from iris_bot.processed_dataset import ProcessedRow


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    volume: float
    entry_price: float
    stop_loss: float
    take_profit: float
    signal_timestamp: str


@dataclass(frozen=True)
class ExecutionDecision:
    accepted: bool
    reason: str
    details: dict[str, Any]


ExecutionValidator = Callable[[OrderIntent], ExecutionDecision]


@dataclass(frozen=True)
class PaperSessionConfig:
    """All static configuration for a paper/demo engine session.

    Separating session config from per-call data (rows, probabilities,
    initial_state) keeps ``run_paper_engine`` signatures manageable and makes
    the config independently testable and serializable.
    """

    # --- Identity ---
    mode: str  # "paper" | "demo_dry"

    # --- Trading gates ---
    threshold: float
    trading_symbols: tuple[str, ...]
    one_position_per_symbol: bool
    allow_long: bool
    allow_short: bool

    # --- Execution config ---
    backtest: BacktestConfig
    risk: RiskConfig

    # --- Exit config ---
    intrabar_policy: str = "conservative"
    aux_rates: dict[str, float] | None = None
    exit_policy_config: ExitPolicyConfig | None = None
    dynamic_exit_config: DynamicExitConfig | None = None

    # --- Per-symbol profile overrides ---
    symbol_exit_profiles: dict[str, SymbolExitProfile] | None = None
    symbol_strategy_profiles: dict[str, Any] | None = None
    threshold_by_symbol: dict[str, float] | None = None

    # --- Hooks ---
    execution_validator: ExecutionValidator | None = None
    should_process_row: Callable[[PaperEngineState, ProcessedRow], tuple[bool, str]] | None = None
