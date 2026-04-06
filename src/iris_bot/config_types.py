from __future__ import annotations

from dataclasses import dataclass, field
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
    evidence_max_age_days: float | None = None  # None = unbounded; set to prune old entries on each ingest


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
class DemoAuditConfig:
    history_days: int = 7
    timestamp_tolerance_seconds: int = 300     # ±5 min for trade matching
    pnl_divergence_tolerance_pct: float = 5.0  # max % divergence local vs broker PnL
    slippage_tolerance_pips: float = 2.0        # max acceptable mean slippage
    enabled: bool = True


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
    Defaults are deliberately conservative; ante la duda, bloquear.
    """

    min_trade_count: int = 10
    max_no_trade_ratio: float = 0.80
    max_blocked_trades_ratio: float = 0.30
    min_profit_factor: float = 1.1
    min_expectancy_usd: float = 0.50
    lifecycle_max_critical: int = 0
    require_lifecycle_audit_ok: bool = True
    lifecycle_max_age_hours: float = 72.0
    endurance_min_cycles: int = 3
    endurance_min_trades: int = 10
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
    format: str = "text"  # "text" | "json" — json writes run.jsonl alongside run.log


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
    refinement_steps: int = 10  # second fine-grained pass; 0 = disabled


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
    use_class_weights: bool = True
    class_weight_max_multiplier: float = 5.0
    use_probability_calibration: bool = True
    probability_calibration_method: str = "global_temperature"
    calibration_min_temperature: float = 0.8
    calibration_max_temperature: float = 2.0
    calibration_temperature_step: float = 0.1


@dataclass(frozen=True)
class SignificanceConfig:
    enabled: bool = False
    trials: int = 100
    seed: int = 42
    metric_name: str = "total_net_pnl_usd"
    higher_is_better: bool = True
    minimum_valid_folds: int = 1


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


@dataclass(frozen=True)
class DemoExecutionConfig:
    enabled: bool = False
    target_symbol: str = ""
    max_orders_per_run: int = 1
    auto_close_after_entry: bool = True
    require_explicit_activation: bool = True
    deviation_points: int = 20
    registry_filename: str = "demo_execution_registry.json"


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"
    risk: RiskConfig = field(default_factory=RiskConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    exit_policy: ExitPolicyRuntimeConfig = field(default_factory=ExitPolicyRuntimeConfig)
    data: DataConfig = field(
        default_factory=lambda: DataConfig(Path("data/raw"), Path("data/processed"), Path("runs"), Path("data/runtime"))
    )
    reconciliation: ReconciliationConfig = field(default_factory=ReconciliationConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    operational: OperationalConfig = field(default_factory=OperationalConfig)
    demo_audit: DemoAuditConfig = field(default_factory=DemoAuditConfig)
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
    significance: SignificanceConfig = field(default_factory=SignificanceConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    demo_execution: DemoExecutionConfig = field(default_factory=DemoExecutionConfig)
