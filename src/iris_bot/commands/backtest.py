from __future__ import annotations

import dataclasses

from iris_bot.backtest import run_backtest
from iris_bot.config import Settings
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.paper import run_paper_session
from iris_bot.processed_dataset import load_processed_dataset
from iris_bot.symbols import load_symbol_strategy_profiles, row_allowed_by_profile
from iris_bot.wf_backtest import run_walkforward_economic_backtest


def run_walkforward_backtest_command(settings: Settings, intrabar_policy_override: str | None = None) -> int:
    effective_settings = settings
    if intrabar_policy_override:
        effective_settings = dataclasses.replace(
            effective_settings,
            backtest=dataclasses.replace(effective_settings.backtest, intrabar_policy=intrabar_policy_override),
        )
    run_dir = build_run_directory(effective_settings.data.runs_dir, "wf_backtest")
    logger = configure_logging(run_dir, effective_settings.logging.level, effective_settings.logging.format)
    try:
        dataset = load_processed_dataset(
            effective_settings.experiment.processed_dataset_path,
            effective_settings.experiment.processed_schema_path,
            effective_settings.experiment.processed_manifest_path,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    rows = [row for row in dataset.rows if row.timeframe == effective_settings.trading.primary_timeframe]
    symbol_profiles = load_symbol_strategy_profiles(effective_settings)
    rows = [
        row
        for row in rows
        if symbol_profiles.get(row.symbol) is None
        or (
            symbol_profiles[row.symbol].enabled_state in {"enabled", "caution"}
            and row_allowed_by_profile(symbol_profiles[row.symbol], row.timestamp, row.timeframe)
        )
    ]
    rows.sort(key=lambda r: (r.timestamp, r.symbol))
    min_required = effective_settings.walk_forward.train_window + effective_settings.walk_forward.validation_window + effective_settings.walk_forward.test_window
    if len(rows) < min_required:
        logger.error("Insuficientes filas para walk-forward economico: tiene %s, necesita al menos %s", len(rows), min_required)
        return 2
    result = run_walkforward_economic_backtest(rows=rows, feature_names=dataset.feature_names, settings=effective_settings, run_dir=run_dir, logger=logger)
    write_json_report(run_dir, "wf_backtest_summary.json", result)
    logger.info("wf_backtest complete  total_folds=%s  valid=%s  skipped=%s", result["total_folds"], result["valid_folds"], result["skipped_folds"])
    return 0


def run_paper_command(settings: Settings) -> int:
    exit_code, _ = run_paper_session(settings, mode="paper")
    return exit_code


def run_backtest_command(settings: Settings, intrabar_policy_override: str | None = None) -> int:
    return run_backtest(settings, intrabar_policy_override)
