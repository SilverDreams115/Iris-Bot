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
    max_daily_loss_usd: float = 50.0
    cooldown_bars_after_loss: int = 0


@dataclass(frozen=True)
class TradingConfig:
    symbols: tuple[str, ...] = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")
    timeframes: tuple[str, ...] = ("M5", "M15", "H1")
    allow_long: bool = True
    allow_short: bool = True
    one_position_per_symbol: bool = True
    training_window: int = 150
    primary_timeframe: str = "M15"


@dataclass(frozen=True)
class MT5Config:
    enabled: bool = False
    login: int | None = None
    password: str | None = None
    server: str | None = None
    path: str | None = None
    history_bars: int = 1500
    magic_number: int = 20260401
    comment_tag: str = "IRIS-Bot"
    reconcile_symbols_only: bool = True


@dataclass(frozen=True)
class ReconciliationConfig:
    policy: str = "hard_fail"
    price_tolerance: float = 0.0005
    volume_tolerance: float = 0.000001


@dataclass(frozen=True)
class RecoveryConfig:
    reconnect_retries: int = 3
    reconnect_backoff_seconds: float = 0.0
    require_state_restore_clean: bool = True


@dataclass(frozen=True)
class SessionConfig:
    enabled: bool = True
    allowed_weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    allowed_start_hour_utc: int = 0
    allowed_end_hour_utc: int = 23


@dataclass(frozen=True)
class OperationalConfig:
    repeated_rejection_alert_threshold: int = 3
    persistence_state_filename: str = "runtime_state.json"


@dataclass(frozen=True)
class SoakConfig:
    cycles: int = 3
    pause_seconds: float = 0.0
    restore_between_cycles: bool = True


@dataclass(frozen=True)
class ChaosConfig:
    enabled: bool = False
    scenarios: tuple[str, ...] = ()
    every_n_cycles: int = 0


@dataclass(frozen=True)
class DynamicExitConfig:
    volatility_adjustment_scale: float = 8.0
    min_stop_loss_pct: float = 0.0010
    max_stop_loss_pct: float = 0.0100
    min_take_profit_pct: float = 0.0015
    max_take_profit_pct: float = 0.0200


@dataclass(frozen=True)
class StrategyConfig:
    profiles_filename: str = "strategy_profiles.json"
    min_symbol_rows: int = 180
    min_validation_trades: int = 5
    min_expectancy_usd: float = 0.0
    max_drawdown_usd: float = 125.0
    min_profit_factor: float = 1.05
    min_positive_walkforward_ratio: float = 0.50
    caution_no_trade_ratio: float = 0.85
    feature_dominance_threshold: float = 0.98
    feature_min_variance: float = 1e-10
    threshold_metric: str = "economic_expectancy"


@dataclass(frozen=True)
class EnduranceConfig:
    symbols: tuple[str, ...] = ()
    target_symbol: str = ""
    mode: str = "paper"
    enabled_only: bool = True
    min_cycles_for_stability: int = 3
    max_expectancy_degradation_pct: float = 0.35
    max_profit_factor_degradation_pct: float = 0.35


@dataclass(frozen=True)
class GovernanceConfig:
    registry_filename: str = "strategy_profile_registry.json"
    target_symbol: str = ""
    target_profile_id: str = ""
    promotion_target_state: str = "approved_demo"
    require_active_profile: bool = True
    allow_validated_fallback: bool = False
    audit_only: bool = True


@dataclass(frozen=True)
class LifecycleConfig:
    history_days: int = 3
    max_critical_mismatches: int = 0
    audit_only: bool = True


@dataclass(frozen=True)
class ApprovedDemoGateConfig:
    """
    Hard floors for approved_demo promotion gate.

    All thresholds are configurable via environment variables.
    Defaults are deliberately conservative — ante la duda, bloquear.

    No threshold here can be relaxed to "make a symbol pass".
    To change a threshold, update the env var and document the reason.
    """
    # Trade evidence
    min_trade_count: int = 10             # Total trades across all endurance cycles
    # Signal efficiency (computed from endurance symbol data)
    max_no_trade_ratio: float = 0.80      # Max fraction of total signals that were no-trade
    max_blocked_trades_ratio: float = 0.30  # Max fraction of (trades+blocked) that were blocked
    # Economic floors (per-cycle average)
    min_profit_factor: float = 1.1        # Minimum acceptable profit factor
    min_expectancy_usd: float = 0.50      # Minimum acceptable expectancy per trade (USD)
    # Lifecycle evidence requirements
    lifecycle_max_critical: int = 0       # Zero critical mismatches tolerated
    require_lifecycle_audit_ok: bool = True  # audit_ok is now required, not just caution-eligible
    lifecycle_max_age_hours: float = 72.0   # Evidence must be recent (3 days)
    # Endurance requirements (stricter than EnduranceConfig)
    endurance_min_cycles: int = 3         # Minimum endurance cycles completed
    endurance_min_trades: int = 10        # Minimum total trades in endurance
    # Degradation limits (tighter than EnduranceConfig defaults of 0.35)
    max_expectancy_degradation_pct: float = 0.25
    max_profit_factor_degradation_pct: float = 0.25


@dataclass(frozen=True)
class ExitPolicyRuntimeConfig:
    stop_policy: str = "static"
    target_policy: str = "static"


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    processed_dir: Path
    runs_dir: Path
    runtime_dir: Path
    raw_dataset_name: str = "market.csv"

    @property
    def raw_dataset_path(self) -> Path:
        return self.raw_dir / self.raw_dataset_name

    @property
    def raw_metadata_path(self) -> Path:
        return self.raw_dir / f"{self.raw_dataset_name}.metadata.json"


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(frozen=True)
class LabelingConfig:
    mode: str = "triple_barrier"
    min_abs_return: float = 0.0002
    horizon_bars: int = 8
    take_profit_pct: float = 0.0020
    stop_loss_pct: float = 0.0020
    allow_no_trade: bool = True


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    test_ratio: float = 0.2


@dataclass(frozen=True)
class WalkForwardConfig:
    enabled: bool = True
    train_window: int = 240
    validation_window: int = 80
    test_window: int = 80
    step: int = 80


@dataclass(frozen=True)
class ThresholdConfig:
    grid: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
    objective_metric: str = "macro_f1"


@dataclass(frozen=True)
class XGBoostConfig:
    enabled: bool = True
    num_boost_round: int = 80
    early_stopping_rounds: int = 10
    eta: float = 0.08
    max_depth: int = 4
    min_child_weight: float = 2.0
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    use_primary_timeframe_only: bool = True
    benchmark_name: str = "momentum_sign"
    save_predictions_csv: bool = True
    processed_dataset_name: str = "processed_market.csv"

    @property
    def processed_dataset_path(self) -> Path:
        return self._processed_dir / self.processed_dataset_name

    @property
    def processed_manifest_path(self) -> Path:
        return self._processed_dir / f"{self.processed_dataset_name}.manifest.json"

    @property
    def processed_schema_path(self) -> Path:
        return self._processed_dir / f"{self.processed_dataset_name}.schema.json"

    _processed_dir: Path = Path("data/processed")


@dataclass(frozen=True)
class BacktestConfig:
    starting_balance_usd: float = 1000.0
    spread_pips: float = 1.2
    slippage_pips: float = 0.2
    commission_per_lot_per_side_usd: float = 3.5
    contract_size: float = 100000.0
    min_lot: float = 0.01
    lot_step: float = 0.01
    max_lot: float = 100.0
    use_atr_stops: bool = True
    fixed_stop_loss_pct: float = 0.0020
    fixed_take_profit_pct: float = 0.0030
    max_holding_bars: int = 8
    experiment_run_dir: str | None = None
    allow_long: bool = True
    allow_short: bool = True
    intrabar_policy: str = "conservative"
    # Política para cuando SL y TP se tocan en la misma vela:
    #   "conservative" → gana el stop loss  (peor resultado para el trader)
    #   "optimistic"   → gana el take profit (mejor resultado para el trader)


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


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def load_settings() -> "Settings":
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"
    return Settings(
        project_root=project_root,
        data_dir=data_root,
        risk=RiskConfig(
            risk_per_trade=_env_float("IRIS_RISK_PER_TRADE", 0.01),
            max_open_positions=_env_int("IRIS_MAX_OPEN_POSITIONS", 4) or 4,
            min_balance_usd=_env_float("IRIS_MIN_BALANCE_USD", 25.0),
            min_confidence=_env_float("IRIS_MIN_CONFIDENCE", 0.55),
            atr_stop_loss_multiplier=_env_float("IRIS_ATR_SL_MULTIPLIER", 1.5),
            atr_take_profit_multiplier=_env_float("IRIS_ATR_TP_MULTIPLIER", 3.0),
            max_daily_loss_usd=_env_float("IRIS_MAX_DAILY_LOSS_USD", 50.0),
            cooldown_bars_after_loss=_env_int("IRIS_COOLDOWN_BARS_AFTER_LOSS", 0) or 0,
        ),
        trading=TradingConfig(
            symbols=_env_tuple("IRIS_SYMBOLS", ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")),
            timeframes=_env_tuple("IRIS_TIMEFRAMES", ("M5", "M15", "H1")),
            allow_long=_env_bool("IRIS_ALLOW_LONG", True),
            allow_short=_env_bool("IRIS_ALLOW_SHORT", True),
            one_position_per_symbol=_env_bool("IRIS_ONE_POSITION_PER_SYMBOL", True),
            training_window=_env_int("IRIS_TRAINING_WINDOW", 150) or 150,
            primary_timeframe=_env_str("IRIS_PRIMARY_TIMEFRAME", "M15"),
        ),
        mt5=MT5Config(
            enabled=_env_bool("IRIS_MT5_ENABLED", False),
            login=_env_int("IRIS_MT5_LOGIN", None),
            password=os.getenv("IRIS_MT5_PASSWORD"),
            server=os.getenv("IRIS_MT5_SERVER"),
            path=os.getenv("IRIS_MT5_PATH"),
            history_bars=_env_int("IRIS_MT5_HISTORY_BARS", 1500) or 1500,
            magic_number=_env_int("IRIS_MT5_MAGIC_NUMBER", 20260401) or 20260401,
            comment_tag=_env_str("IRIS_MT5_COMMENT_TAG", "IRIS-Bot"),
            reconcile_symbols_only=_env_bool("IRIS_MT5_RECONCILE_SYMBOLS_ONLY", True),
        ),
        exit_policy=ExitPolicyRuntimeConfig(
            stop_policy=_env_str("IRIS_STOP_POLICY", "static"),
            target_policy=_env_str("IRIS_TARGET_POLICY", "static"),
        ),
        data=DataConfig(
            raw_dir=data_root / "raw",
            processed_dir=data_root / "processed",
            runs_dir=project_root / "runs",
            runtime_dir=data_root / "runtime",
            raw_dataset_name=_env_str("IRIS_RAW_DATASET_NAME", "market.csv"),
        ),
        reconciliation=ReconciliationConfig(
            policy=_env_str("IRIS_RECONCILIATION_POLICY", "hard_fail"),
            price_tolerance=_env_float("IRIS_RECONCILIATION_PRICE_TOLERANCE", 0.0005),
            volume_tolerance=_env_float("IRIS_RECONCILIATION_VOLUME_TOLERANCE", 0.000001),
        ),
        recovery=RecoveryConfig(
            reconnect_retries=_env_int("IRIS_RECONNECT_RETRIES", 3) or 3,
            reconnect_backoff_seconds=_env_float("IRIS_RECONNECT_BACKOFF_SECONDS", 0.0),
            require_state_restore_clean=_env_bool("IRIS_REQUIRE_STATE_RESTORE_CLEAN", True),
        ),
        session=SessionConfig(
            enabled=_env_bool("IRIS_SESSION_ENABLED", True),
            allowed_weekdays=tuple(int(item) for item in _env_tuple("IRIS_ALLOWED_WEEKDAYS", ("0", "1", "2", "3", "4"))),
            allowed_start_hour_utc=_env_int("IRIS_ALLOWED_START_HOUR_UTC", 0) or 0,
            allowed_end_hour_utc=_env_int("IRIS_ALLOWED_END_HOUR_UTC", 23) or 23,
        ),
        operational=OperationalConfig(
            repeated_rejection_alert_threshold=_env_int("IRIS_REPEATED_REJECTION_ALERT_THRESHOLD", 3) or 3,
            persistence_state_filename=_env_str("IRIS_PERSISTENCE_STATE_FILENAME", "runtime_state.json"),
        ),
        soak=SoakConfig(
            cycles=_env_int("IRIS_SOAK_CYCLES", 3) or 3,
            pause_seconds=_env_float("IRIS_SOAK_PAUSE_SECONDS", 0.0),
            restore_between_cycles=_env_bool("IRIS_SOAK_RESTORE_BETWEEN_CYCLES", True),
        ),
        chaos=ChaosConfig(
            enabled=_env_bool("IRIS_CHAOS_ENABLED", False),
            scenarios=_env_tuple("IRIS_CHAOS_SCENARIOS", ()),
            every_n_cycles=_env_int("IRIS_CHAOS_EVERY_N_CYCLES", 0) or 0,
        ),
        dynamic_exits=DynamicExitConfig(
            volatility_adjustment_scale=_env_float("IRIS_DYNAMIC_EXIT_VOL_ADJUST_SCALE", 8.0),
            min_stop_loss_pct=_env_float("IRIS_DYNAMIC_EXIT_MIN_STOP_PCT", 0.0010),
            max_stop_loss_pct=_env_float("IRIS_DYNAMIC_EXIT_MAX_STOP_PCT", 0.0100),
            min_take_profit_pct=_env_float("IRIS_DYNAMIC_EXIT_MIN_TAKE_PROFIT_PCT", 0.0015),
            max_take_profit_pct=_env_float("IRIS_DYNAMIC_EXIT_MAX_TAKE_PROFIT_PCT", 0.0200),
        ),
        strategy=StrategyConfig(
            profiles_filename=_env_str("IRIS_STRATEGY_PROFILES_FILENAME", "strategy_profiles.json"),
            min_symbol_rows=_env_int("IRIS_STRATEGY_MIN_SYMBOL_ROWS", 180) or 180,
            min_validation_trades=_env_int("IRIS_STRATEGY_MIN_VALIDATION_TRADES", 5) or 5,
            min_expectancy_usd=_env_float("IRIS_STRATEGY_MIN_EXPECTANCY_USD", 0.0),
            max_drawdown_usd=_env_float("IRIS_STRATEGY_MAX_DRAWDOWN_USD", 125.0),
            min_profit_factor=_env_float("IRIS_STRATEGY_MIN_PROFIT_FACTOR", 1.05),
            min_positive_walkforward_ratio=_env_float("IRIS_STRATEGY_MIN_POSITIVE_WF_RATIO", 0.50),
            caution_no_trade_ratio=_env_float("IRIS_STRATEGY_CAUTION_NO_TRADE_RATIO", 0.85),
            feature_dominance_threshold=_env_float("IRIS_STRATEGY_FEATURE_DOMINANCE_THRESHOLD", 0.98),
            feature_min_variance=_env_float("IRIS_STRATEGY_FEATURE_MIN_VARIANCE", 1e-10),
            threshold_metric=_env_str("IRIS_STRATEGY_THRESHOLD_METRIC", "economic_expectancy"),
        ),
        endurance=EnduranceConfig(
            symbols=_env_tuple("IRIS_ENDURANCE_SYMBOLS", ()),
            target_symbol=_env_str("IRIS_ENDURANCE_TARGET_SYMBOL", ""),
            mode=_env_str("IRIS_ENDURANCE_MODE", "paper"),
            enabled_only=_env_bool("IRIS_ENDURANCE_ENABLED_ONLY", True),
            min_cycles_for_stability=_env_int("IRIS_ENDURANCE_MIN_CYCLES", 3) or 3,
            max_expectancy_degradation_pct=_env_float("IRIS_ENDURANCE_MAX_EXPECTANCY_DEGRADATION_PCT", 0.35),
            max_profit_factor_degradation_pct=_env_float("IRIS_ENDURANCE_MAX_PROFIT_FACTOR_DEGRADATION_PCT", 0.35),
        ),
        governance=GovernanceConfig(
            registry_filename=_env_str("IRIS_GOVERNANCE_REGISTRY_FILENAME", "strategy_profile_registry.json"),
            target_symbol=_env_str("IRIS_GOVERNANCE_TARGET_SYMBOL", ""),
            target_profile_id=_env_str("IRIS_GOVERNANCE_TARGET_PROFILE_ID", ""),
            promotion_target_state=_env_str("IRIS_GOVERNANCE_PROMOTION_TARGET_STATE", "approved_demo"),
            require_active_profile=_env_bool("IRIS_GOVERNANCE_REQUIRE_ACTIVE_PROFILE", True),
            allow_validated_fallback=_env_bool("IRIS_GOVERNANCE_ALLOW_VALIDATED_FALLBACK", False),
            audit_only=_env_bool("IRIS_GOVERNANCE_AUDIT_ONLY", True),
        ),
        lifecycle=LifecycleConfig(
            history_days=_env_int("IRIS_LIFECYCLE_HISTORY_DAYS", 3) or 3,
            max_critical_mismatches=_env_int("IRIS_LIFECYCLE_MAX_CRITICAL_MISMATCHES", 0) or 0,
            audit_only=_env_bool("IRIS_LIFECYCLE_AUDIT_ONLY", True),
        ),
        approved_demo_gate=ApprovedDemoGateConfig(
            min_trade_count=_env_int("IRIS_GATE_MIN_TRADE_COUNT", 10) or 10,
            max_no_trade_ratio=_env_float("IRIS_GATE_MAX_NO_TRADE_RATIO", 0.80),
            max_blocked_trades_ratio=_env_float("IRIS_GATE_MAX_BLOCKED_TRADES_RATIO", 0.30),
            min_profit_factor=_env_float("IRIS_GATE_MIN_PROFIT_FACTOR", 1.1),
            min_expectancy_usd=_env_float("IRIS_GATE_MIN_EXPECTANCY_USD", 0.50),
            lifecycle_max_critical=_env_int("IRIS_GATE_LIFECYCLE_MAX_CRITICAL", 0) or 0,
            require_lifecycle_audit_ok=_env_bool("IRIS_GATE_REQUIRE_LIFECYCLE_AUDIT_OK", True),
            lifecycle_max_age_hours=_env_float("IRIS_GATE_LIFECYCLE_MAX_AGE_HOURS", 72.0),
            endurance_min_cycles=_env_int("IRIS_GATE_ENDURANCE_MIN_CYCLES", 3) or 3,
            endurance_min_trades=_env_int("IRIS_GATE_ENDURANCE_MIN_TRADES", 10) or 10,
            max_expectancy_degradation_pct=_env_float("IRIS_GATE_MAX_EXPECTANCY_DEGRADATION_PCT", 0.25),
            max_profit_factor_degradation_pct=_env_float("IRIS_GATE_MAX_PROFIT_FACTOR_DEGRADATION_PCT", 0.25),
        ),
        logging=LoggingConfig(level=_env_str("IRIS_LOG_LEVEL", "INFO")),
        labeling=LabelingConfig(
            mode=_env_str("IRIS_LABEL_MODE", "triple_barrier"),
            min_abs_return=_env_float("IRIS_LABEL_MIN_ABS_RETURN", 0.0002),
            horizon_bars=_env_int("IRIS_LABEL_HORIZON_BARS", 8) or 8,
            take_profit_pct=_env_float("IRIS_LABEL_TAKE_PROFIT_PCT", 0.0020),
            stop_loss_pct=_env_float("IRIS_LABEL_STOP_LOSS_PCT", 0.0020),
            allow_no_trade=_env_bool("IRIS_LABEL_ALLOW_NO_TRADE", True),
        ),
        split=SplitConfig(
            train_ratio=_env_float("IRIS_TRAIN_RATIO", 0.6),
            validation_ratio=_env_float("IRIS_VALIDATION_RATIO", 0.2),
            test_ratio=_env_float("IRIS_TEST_RATIO", 0.2),
        ),
        walk_forward=WalkForwardConfig(
            enabled=_env_bool("IRIS_WALK_FORWARD_ENABLED", True),
            train_window=_env_int("IRIS_WF_TRAIN_WINDOW", 240) or 240,
            validation_window=_env_int("IRIS_WF_VALIDATION_WINDOW", 80) or 80,
            test_window=_env_int("IRIS_WF_TEST_WINDOW", 80) or 80,
            step=_env_int("IRIS_WF_STEP", 80) or 80,
        ),
        threshold=ThresholdConfig(
            grid=tuple(float(item) for item in _env_tuple("IRIS_THRESHOLD_GRID", ("0.45", "0.50", "0.55", "0.60", "0.65", "0.70"))),
            objective_metric=_env_str("IRIS_THRESHOLD_METRIC", "macro_f1"),
        ),
        xgboost=XGBoostConfig(
            enabled=_env_bool("IRIS_XGB_ENABLED", True),
            num_boost_round=_env_int("IRIS_XGB_NUM_BOOST_ROUND", 80) or 80,
            early_stopping_rounds=_env_int("IRIS_XGB_EARLY_STOPPING_ROUNDS", 10) or 10,
            eta=_env_float("IRIS_XGB_ETA", 0.08),
            max_depth=_env_int("IRIS_XGB_MAX_DEPTH", 4) or 4,
            min_child_weight=_env_float("IRIS_XGB_MIN_CHILD_WEIGHT", 2.0),
            subsample=_env_float("IRIS_XGB_SUBSAMPLE", 0.9),
            colsample_bytree=_env_float("IRIS_XGB_COLSAMPLE_BYTREE", 0.9),
            reg_lambda=_env_float("IRIS_XGB_REG_LAMBDA", 1.0),
            reg_alpha=_env_float("IRIS_XGB_REG_ALPHA", 0.0),
            seed=_env_int("IRIS_XGB_SEED", 42) or 42,
        ),
        experiment=ExperimentConfig(
            use_primary_timeframe_only=_env_bool("IRIS_USE_PRIMARY_TIMEFRAME_ONLY", True),
            benchmark_name=_env_str("IRIS_BENCHMARK_NAME", "momentum_sign"),
            save_predictions_csv=_env_bool("IRIS_SAVE_PREDICTIONS_CSV", True),
            processed_dataset_name=_env_str("IRIS_PROCESSED_DATASET_NAME", "processed_market.csv"),
            _processed_dir=data_root / "processed",
        ),
        backtest=BacktestConfig(
            starting_balance_usd=_env_float("IRIS_BACKTEST_STARTING_BALANCE_USD", 1000.0),
            spread_pips=_env_float("IRIS_BACKTEST_SPREAD_PIPS", 1.2),
            slippage_pips=_env_float("IRIS_BACKTEST_SLIPPAGE_PIPS", 0.2),
            commission_per_lot_per_side_usd=_env_float("IRIS_BACKTEST_COMMISSION_PER_LOT_PER_SIDE_USD", 3.5),
            contract_size=_env_float("IRIS_BACKTEST_CONTRACT_SIZE", 100000.0),
            min_lot=_env_float("IRIS_BACKTEST_MIN_LOT", 0.01),
            lot_step=_env_float("IRIS_BACKTEST_LOT_STEP", 0.01),
            max_lot=_env_float("IRIS_BACKTEST_MAX_LOT", 100.0),
            use_atr_stops=_env_bool("IRIS_BACKTEST_USE_ATR_STOPS", True),
            fixed_stop_loss_pct=_env_float("IRIS_BACKTEST_FIXED_STOP_LOSS_PCT", 0.0020),
            fixed_take_profit_pct=_env_float("IRIS_BACKTEST_FIXED_TAKE_PROFIT_PCT", 0.0030),
            max_holding_bars=_env_int("IRIS_BACKTEST_MAX_HOLDING_BARS", 8) or 8,
            experiment_run_dir=os.getenv("IRIS_BACKTEST_EXPERIMENT_RUN_DIR"),
            allow_long=_env_bool("IRIS_BACKTEST_ALLOW_LONG", True),
            allow_short=_env_bool("IRIS_BACKTEST_ALLOW_SHORT", True),
            intrabar_policy=_env_str("IRIS_BACKTEST_INTRABAR_POLICY", "conservative"),
        ),
    )


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    exit_policy: ExitPolicyRuntimeConfig = field(default_factory=ExitPolicyRuntimeConfig)
    data: DataConfig = field(default_factory=lambda: DataConfig(Path("data/raw"), Path("data/processed"), Path("runs"), Path("data/runtime")))
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    operational: OperationalConfig = field(default_factory=OperationalConfig)
    soak: SoakConfig = field(default_factory=SoakConfig)
    chaos: ChaosConfig = field(default_factory=ChaosConfig)
    dynamic_exits: DynamicExitConfig = field(default_factory=DynamicExitConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    endurance: EnduranceConfig = field(default_factory=EnduranceConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    approved_demo_gate: ApprovedDemoGateConfig = field(default_factory=ApprovedDemoGateConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


settings = load_settings()
