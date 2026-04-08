from __future__ import annotations

import os
from pathlib import Path

from iris_bot.config_types import (
    ApprovedDemoGateConfig,
    BacktestConfig,
    ChaosConfig,
    DataConfig,
    DemoExecutionConfig,
    DynamicExitConfig,
    EnduranceConfig,
    ExitPolicyRuntimeConfig,
    ExperimentConfig,
    GovernanceConfig,
    LabelingConfig,
    LifecycleConfig,
    LoggingConfig,
    MT5Config,
    OperationalConfig,
    ReconciliationConfig,
    RecoveryConfig,
    RiskConfig,
    SessionConfig,
    SignificanceConfig,
    Settings,
    SoakConfig,
    SplitConfig,
    StrategyConfig,
    ThresholdConfig,
    TradingConfig,
    WalkForwardConfig,
    XGBoostConfig,
)

_ENV_SOURCES: dict[str, str] = {}


def _load_local_env_file() -> None:
    """Load simple KEY=VALUE pairs from project-root .env if present.

    Existing environment variables win over .env values.
    This keeps shell overrides authoritative while allowing local secrets
    for developer workflows without adding an external dependency.
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if key in os.environ:
            _ENV_SOURCES.setdefault(key, "process_env")
            continue
        parsed = value.strip()
        if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {"'", '"'}:
            parsed = parsed[1:-1]
        os.environ[key] = parsed
        _ENV_SOURCES[key] = ".env"


def env_source(name: str) -> str:
    if name in os.environ:
        return _ENV_SOURCES.get(name, "process_env")
    return "default"


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


def _load_risk_config() -> RiskConfig:
    return RiskConfig(
        risk_per_trade=_env_float("IRIS_RISK_PER_TRADE", 0.01),
        max_open_positions=_env_int("IRIS_MAX_OPEN_POSITIONS", 4) or 4,
        min_balance_usd=_env_float("IRIS_MIN_BALANCE_USD", 25.0),
        min_confidence=_env_float("IRIS_MIN_CONFIDENCE", 0.55),
        atr_stop_loss_multiplier=_env_float("IRIS_ATR_SL_MULTIPLIER", 1.5),
        atr_take_profit_multiplier=_env_float("IRIS_ATR_TP_MULTIPLIER", 3.0),
        max_daily_loss_usd=_env_float("IRIS_MAX_DAILY_LOSS_USD", 50.0),
        cooldown_bars_after_loss=_env_int("IRIS_COOLDOWN_BARS_AFTER_LOSS", 0) or 0,
    )


def _load_trading_config() -> TradingConfig:
    return TradingConfig(
        symbols=_env_tuple("IRIS_SYMBOLS", ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD")),
        timeframes=_env_tuple("IRIS_TIMEFRAMES", ("M5", "M15", "H1")),
        allow_long=_env_bool("IRIS_ALLOW_LONG", True),
        allow_short=_env_bool("IRIS_ALLOW_SHORT", True),
        one_position_per_symbol=_env_bool("IRIS_ONE_POSITION_PER_SYMBOL", True),
        training_window=_env_int("IRIS_TRAINING_WINDOW", 150) or 150,
        primary_timeframe=_env_str("IRIS_PRIMARY_TIMEFRAME", "M15"),
    )


def _load_mt5_config() -> MT5Config:
    return MT5Config(
        enabled=_env_bool("IRIS_MT5_ENABLED", False),
        login=_env_int("IRIS_MT5_LOGIN", None),
        password=os.getenv("IRIS_MT5_PASSWORD"),
        server=os.getenv("IRIS_MT5_SERVER"),
        path=os.getenv("IRIS_MT5_PATH"),
        history_bars=_env_int("IRIS_MT5_HISTORY_BARS", 1500) or 1500,
        magic_number=_env_int("IRIS_MT5_MAGIC_NUMBER", 20260401) or 20260401,
        comment_tag=_env_str("IRIS_MT5_COMMENT_TAG", "IRIS-Bot"),
        ownership_mode=_env_str("IRIS_MT5_OWNERSHIP_MODE", "strict"),
        reconcile_symbols_only=_env_bool("IRIS_MT5_RECONCILE_SYMBOLS_ONLY", True),
    )


def _load_exit_policy_config() -> ExitPolicyRuntimeConfig:
    return ExitPolicyRuntimeConfig(
        stop_policy=_env_str("IRIS_STOP_POLICY", "static"),
        target_policy=_env_str("IRIS_TARGET_POLICY", "static"),
    )


def _load_data_config(project_root: Path, data_root: Path) -> DataConfig:
    return DataConfig(
        raw_dir=data_root / "raw",
        processed_dir=data_root / "processed",
        runs_dir=project_root / "runs",
        runtime_dir=data_root / "runtime",
        raw_dataset_name=_env_str("IRIS_RAW_DATASET_NAME", "market.csv"),
    )


def _load_reconciliation_config() -> ReconciliationConfig:
    return ReconciliationConfig(
        policy=_env_str("IRIS_RECONCILIATION_POLICY", "hard_fail"),
        price_tolerance=_env_float("IRIS_RECONCILIATION_PRICE_TOLERANCE", 0.0005),
        volume_tolerance=_env_float("IRIS_RECONCILIATION_VOLUME_TOLERANCE", 0.000001),
    )


def _load_recovery_config() -> RecoveryConfig:
    return RecoveryConfig(
        reconnect_retries=_env_int("IRIS_RECONNECT_RETRIES", 3) or 3,
        reconnect_backoff_seconds=_env_float("IRIS_RECONNECT_BACKOFF_SECONDS", 0.0),
        require_state_restore_clean=_env_bool("IRIS_REQUIRE_STATE_RESTORE_CLEAN", True),
    )


def _load_session_config() -> SessionConfig:
    return SessionConfig(
        enabled=_env_bool("IRIS_SESSION_ENABLED", True),
        allowed_weekdays=tuple(int(item) for item in _env_tuple("IRIS_ALLOWED_WEEKDAYS", ("0", "1", "2", "3", "4"))),
        allowed_start_hour_utc=_env_int("IRIS_ALLOWED_START_HOUR_UTC", 0) or 0,
        allowed_end_hour_utc=_env_int("IRIS_ALLOWED_END_HOUR_UTC", 23) or 23,
    )


def _load_operational_config() -> OperationalConfig:
    return OperationalConfig(
        repeated_rejection_alert_threshold=_env_int("IRIS_REPEATED_REJECTION_ALERT_THRESHOLD", 3) or 3,
        persistence_state_filename=_env_str("IRIS_PERSISTENCE_STATE_FILENAME", "runtime_state.json"),
    )


def _load_soak_config() -> SoakConfig:
    return SoakConfig(
        cycles=_env_int("IRIS_SOAK_CYCLES", 3) or 3,
        pause_seconds=_env_float("IRIS_SOAK_PAUSE_SECONDS", 0.0),
        restore_between_cycles=_env_bool("IRIS_SOAK_RESTORE_BETWEEN_CYCLES", True),
    )


def _load_chaos_config() -> ChaosConfig:
    return ChaosConfig(
        enabled=_env_bool("IRIS_CHAOS_ENABLED", False),
        scenarios=_env_tuple("IRIS_CHAOS_SCENARIOS", ()),
        every_n_cycles=_env_int("IRIS_CHAOS_EVERY_N_CYCLES", 0) or 0,
    )


def _load_dynamic_exits_config() -> DynamicExitConfig:
    return DynamicExitConfig(
        volatility_adjustment_scale=_env_float("IRIS_DYNAMIC_EXIT_VOL_ADJUST_SCALE", 8.0),
        min_stop_loss_pct=_env_float("IRIS_DYNAMIC_EXIT_MIN_STOP_PCT", 0.0010),
        max_stop_loss_pct=_env_float("IRIS_DYNAMIC_EXIT_MAX_STOP_PCT", 0.0100),
        min_take_profit_pct=_env_float("IRIS_DYNAMIC_EXIT_MIN_TAKE_PROFIT_PCT", 0.0015),
        max_take_profit_pct=_env_float("IRIS_DYNAMIC_EXIT_MAX_TAKE_PROFIT_PCT", 0.0200),
    )


def _load_strategy_config() -> StrategyConfig:
    return StrategyConfig(
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
    )


def _load_endurance_config() -> EnduranceConfig:
    return EnduranceConfig(
        symbols=_env_tuple("IRIS_ENDURANCE_SYMBOLS", ()),
        target_symbol=_env_str("IRIS_ENDURANCE_TARGET_SYMBOL", ""),
        mode=_env_str("IRIS_ENDURANCE_MODE", "paper"),
        enabled_only=_env_bool("IRIS_ENDURANCE_ENABLED_ONLY", True),
        min_cycles_for_stability=_env_int("IRIS_ENDURANCE_MIN_CYCLES", 3) or 3,
        max_expectancy_degradation_pct=_env_float("IRIS_ENDURANCE_MAX_EXPECTANCY_DEGRADATION_PCT", 0.35),
        max_profit_factor_degradation_pct=_env_float("IRIS_ENDURANCE_MAX_PROFIT_FACTOR_DEGRADATION_PCT", 0.35),
    )


def _load_governance_config() -> GovernanceConfig:
    return GovernanceConfig(
        registry_filename=_env_str("IRIS_GOVERNANCE_REGISTRY_FILENAME", "strategy_profile_registry.json"),
        policy_filename=_env_str("IRIS_GOVERNANCE_POLICY_FILENAME", "governance_policy.json"),
        target_symbol=_env_str("IRIS_GOVERNANCE_TARGET_SYMBOL", ""),
        target_profile_id=_env_str("IRIS_GOVERNANCE_TARGET_PROFILE_ID", ""),
        promotion_target_state=_env_str("IRIS_GOVERNANCE_PROMOTION_TARGET_STATE", "approved_demo"),
        require_active_profile=_env_bool("IRIS_GOVERNANCE_REQUIRE_ACTIVE_PROFILE", True),
        allow_validated_fallback=_env_bool("IRIS_GOVERNANCE_ALLOW_VALIDATED_FALLBACK", False),
        audit_only=_env_bool("IRIS_GOVERNANCE_AUDIT_ONLY", True),
    )


def _load_lifecycle_config() -> LifecycleConfig:
    return LifecycleConfig(
        history_days=_env_int("IRIS_LIFECYCLE_HISTORY_DAYS", 3) or 3,
        max_critical_mismatches=_env_int("IRIS_LIFECYCLE_MAX_CRITICAL_MISMATCHES", 0) or 0,
        audit_only=_env_bool("IRIS_LIFECYCLE_AUDIT_ONLY", True),
    )


def _load_approved_demo_gate_config() -> ApprovedDemoGateConfig:
    return ApprovedDemoGateConfig(
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
    )


def _load_labeling_config() -> LabelingConfig:
    return LabelingConfig(
        mode=_env_str("IRIS_LABEL_MODE", "triple_barrier"),
        min_abs_return=_env_float("IRIS_LABEL_MIN_ABS_RETURN", 0.0002),
        horizon_bars=_env_int("IRIS_LABEL_HORIZON_BARS", 8) or 8,
        take_profit_pct=_env_float("IRIS_LABEL_TAKE_PROFIT_PCT", 0.0020),
        stop_loss_pct=_env_float("IRIS_LABEL_STOP_LOSS_PCT", 0.0020),
        allow_no_trade=_env_bool("IRIS_LABEL_ALLOW_NO_TRADE", True),
    )


def _load_split_config() -> SplitConfig:
    return SplitConfig(
        train_ratio=_env_float("IRIS_TRAIN_RATIO", 0.6),
        validation_ratio=_env_float("IRIS_VALIDATION_RATIO", 0.2),
        test_ratio=_env_float("IRIS_TEST_RATIO", 0.2),
    )


def _load_walk_forward_config() -> WalkForwardConfig:
    return WalkForwardConfig(
        enabled=_env_bool("IRIS_WALK_FORWARD_ENABLED", True),
        train_window=_env_int("IRIS_WF_TRAIN_WINDOW", 240) or 240,
        validation_window=_env_int("IRIS_WF_VALIDATION_WINDOW", 80) or 80,
        test_window=_env_int("IRIS_WF_TEST_WINDOW", 80) or 80,
        step=_env_int("IRIS_WF_STEP", 80) or 80,
    )


def _load_threshold_config() -> ThresholdConfig:
    return ThresholdConfig(
        grid=tuple(float(item) for item in _env_tuple("IRIS_THRESHOLD_GRID", ("0.45", "0.50", "0.55", "0.60", "0.65", "0.70"))),
        objective_metric=_env_str("IRIS_THRESHOLD_METRIC", "macro_f1"),
    )


def _load_xgboost_config() -> XGBoostConfig:
    return XGBoostConfig(
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
        use_class_weights=_env_bool("IRIS_XGB_USE_CLASS_WEIGHTS", True),
        class_weight_max_multiplier=_env_float("IRIS_XGB_CLASS_WEIGHT_MAX_MULTIPLIER", 5.0),
        use_probability_calibration=_env_bool("IRIS_XGB_USE_PROBABILITY_CALIBRATION", True),
        probability_calibration_method=_env_str("IRIS_XGB_PROBABILITY_CALIBRATION_METHOD", "global_temperature"),
        calibration_min_temperature=_env_float("IRIS_XGB_CALIBRATION_MIN_TEMPERATURE", 0.8),
        calibration_max_temperature=_env_float("IRIS_XGB_CALIBRATION_MAX_TEMPERATURE", 2.0),
        calibration_temperature_step=_env_float("IRIS_XGB_CALIBRATION_TEMPERATURE_STEP", 0.1),
    )


def _load_experiment_config(data_root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        use_primary_timeframe_only=_env_bool("IRIS_USE_PRIMARY_TIMEFRAME_ONLY", True),
        benchmark_name=_env_str("IRIS_BENCHMARK_NAME", "momentum_sign"),
        save_predictions_csv=_env_bool("IRIS_SAVE_PREDICTIONS_CSV", True),
        processed_dataset_name=_env_str("IRIS_PROCESSED_DATASET_NAME", "processed_market.csv"),
        _processed_dir=data_root / "processed",
    )


def _load_significance_config() -> SignificanceConfig:
    return SignificanceConfig(
        enabled=_env_bool("IRIS_SIGNIFICANCE_ENABLED", False),
        trials=_env_int("IRIS_SIGNIFICANCE_TRIALS", 100) or 100,
        seed=_env_int("IRIS_SIGNIFICANCE_SEED", 42) or 42,
        metric_name=_env_str("IRIS_SIGNIFICANCE_METRIC", "total_net_pnl_usd"),
        higher_is_better=_env_bool("IRIS_SIGNIFICANCE_HIGHER_IS_BETTER", True),
        minimum_valid_folds=_env_int("IRIS_SIGNIFICANCE_MIN_VALID_FOLDS", 1) or 1,
    )


def _load_backtest_config() -> BacktestConfig:
    return BacktestConfig(
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
    )


def _load_demo_execution_config() -> DemoExecutionConfig:
    return DemoExecutionConfig(
        enabled=_env_bool("IRIS_DEMO_EXECUTION_ENABLED", False),
        target_symbol=_env_str("IRIS_DEMO_EXECUTION_TARGET_SYMBOL", ""),
        max_orders_per_run=_env_int("IRIS_DEMO_EXECUTION_MAX_ORDERS_PER_RUN", 1) or 1,
        auto_close_after_entry=_env_bool("IRIS_DEMO_EXECUTION_AUTO_CLOSE_AFTER_ENTRY", False),
        require_explicit_activation=_env_bool("IRIS_DEMO_EXECUTION_REQUIRE_EXPLICIT_ACTIVATION", True),
        deviation_points=_env_int("IRIS_DEMO_EXECUTION_DEVIATION_POINTS", 20) or 20,
        registry_filename=_env_str("IRIS_DEMO_EXECUTION_REGISTRY_FILENAME", "demo_execution_registry.json"),
    )


def load_settings() -> Settings:
    _load_local_env_file()
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"
    return Settings(
        project_root=project_root,
        data_dir=data_root,
        risk=_load_risk_config(),
        trading=_load_trading_config(),
        mt5=_load_mt5_config(),
        exit_policy=_load_exit_policy_config(),
        data=_load_data_config(project_root, data_root),
        reconciliation=_load_reconciliation_config(),
        recovery=_load_recovery_config(),
        session=_load_session_config(),
        operational=_load_operational_config(),
        soak=_load_soak_config(),
        chaos=_load_chaos_config(),
        dynamic_exits=_load_dynamic_exits_config(),
        strategy=_load_strategy_config(),
        endurance=_load_endurance_config(),
        governance=_load_governance_config(),
        lifecycle=_load_lifecycle_config(),
        approved_demo_gate=_load_approved_demo_gate_config(),
        logging=LoggingConfig(level=_env_str("IRIS_LOG_LEVEL", "INFO")),
        labeling=_load_labeling_config(),
        split=_load_split_config(),
        walk_forward=_load_walk_forward_config(),
        threshold=_load_threshold_config(),
        xgboost=_load_xgboost_config(),
        significance=_load_significance_config(),
        experiment=_load_experiment_config(data_root),
        backtest=_load_backtest_config(),
        demo_execution=_load_demo_execution_config(),
    )


def validate_config(s: Settings) -> list[str]:
    errors: list[str] = []

    if s.mt5.enabled:
        if not s.mt5.login:
            errors.append("mt5.enabled=True but IRIS_MT5_LOGIN is not set")
        if not s.mt5.password:
            errors.append("mt5.enabled=True but IRIS_MT5_PASSWORD is not set")
        if not s.mt5.server:
            errors.append("mt5.enabled=True but IRIS_MT5_SERVER is not set")
    if s.mt5.ownership_mode not in {"strict", "compatibility", "audit_only"}:
        errors.append(
            "mt5.ownership_mode="
            f"{s.mt5.ownership_mode!r} must be one of "
            "{'strict', 'compatibility', 'audit_only'}"
        )

    if not (0 < s.risk.risk_per_trade <= 0.5):
        errors.append(f"risk_per_trade={s.risk.risk_per_trade} is outside (0, 0.5]")
    if s.risk.max_open_positions < 1:
        errors.append(f"max_open_positions={s.risk.max_open_positions} must be >= 1")

    if s.trading.primary_timeframe not in s.trading.timeframes:
        errors.append(
            f"primary_timeframe={s.trading.primary_timeframe!r} not in timeframes={s.trading.timeframes}"
        )

    ratio_sum = s.split.train_ratio + s.split.validation_ratio + s.split.test_ratio
    if abs(ratio_sum - 1.0) > 0.01:
        errors.append(f"split ratios sum to {ratio_sum:.3f}, expected 1.0")

    if s.approved_demo_gate.min_profit_factor < 1.0:
        errors.append(
            f"approved_demo_gate.min_profit_factor={s.approved_demo_gate.min_profit_factor} should be >= 1.0"
        )
    if s.significance.trials < 1:
        errors.append(f"significance.trials={s.significance.trials} must be >= 1")
    if s.significance.minimum_valid_folds < 1:
        errors.append(
            f"significance.minimum_valid_folds={s.significance.minimum_valid_folds} must be >= 1"
        )

    return errors
