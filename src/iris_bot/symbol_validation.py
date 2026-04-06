from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from iris_bot.artifacts import read_artifact_payload
from iris_bot.config import Settings
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.processed_dataset import ProcessedRow
from iris_bot.symbol_validation_reporting import (
    _build_common_profile,
    _build_symbol_profile_override,
    _initialize_aggregate_portfolio,
    _initialize_validation_reports,
    _record_short_symbol_decision,
    _record_symbol_validation_outputs,
    _update_aggregate_portfolio,
    _write_validation_reports,
)
from iris_bot.symbol_validation_support import (
    _block_reason_matrix,
    _build_leakage_safe_split,
    _choose_best_model,
    _enablement_decision,
    _evaluate_configuration,
    _exit_policy_diagnostics,
    _load_dataset,
    _primary_rows,
    _prune_feature_names,
    _recommendation_for_symbol,
    _rows_to_matrix,
    _session_timeframe_diagnostics,
    _symbol_walkforward,
)
from iris_bot.symbols import write_symbol_strategy_profiles
from iris_bot.xgb_model import XGBoostMultiClassModel


def run_strategy_validation(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "strategy_validation")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    dataset = _load_dataset(settings)
    all_rows = _primary_rows(dataset, settings)
    grouped: dict[str, list[ProcessedRow]] = defaultdict(list)
    for row in all_rows:
        grouped[row.symbol].append(row)
    if not all_rows:
        logger.error("No processed rows in primary timeframe for per-symbol validation")
        return 1

    global_split = _build_leakage_safe_split(all_rows, settings)
    global_model = XGBoostMultiClassModel(settings.xgboost)
    global_matrix_train, global_labels_train = _rows_to_matrix(global_split.fit_train, dataset.feature_names)
    global_matrix_val, global_labels_val = _rows_to_matrix(global_split.fit_validation, dataset.feature_names)
    try:
        global_model.fit(global_matrix_train, global_labels_train, global_matrix_val, global_labels_val)
    except RuntimeError as exc:
        logger.error(str(exc))
        return 2

    aggregate_portfolio = _initialize_aggregate_portfolio()
    reports = _initialize_validation_reports()
    common_profile = _build_common_profile(settings)
    symbol_profile_overrides: dict[str, dict[str, Any]] = {}

    for symbol, symbol_rows in sorted(grouped.items()):
        aggregate_portfolio["symbols_evaluated"] += 1
        if len(symbol_rows) < 30:
            decision = {"symbol": symbol, "state": "disabled", "enabled": False, "reasons": ["insufficient_symbol_rows"], "chosen_model": "global_model", "chosen_exit_policy": "static"}
            _record_short_symbol_decision(reports, symbol, decision)
            aggregate_portfolio["disabled_symbols"] += 1
            continue

        symbol_split = _build_leakage_safe_split(symbol_rows, settings)
        global_selection_rows = symbol_split.selection
        global_test_rows = symbol_split.final_test
        global_payload = _evaluate_configuration(
            "global_model",
            global_model,
            global_split.fit_train,
            global_split.fit_validation,
            global_selection_rows,
            global_test_rows,
            dataset.feature_names,
            settings,
            symbol,
        )

        symbol_feature_names, pruning_report = _prune_feature_names(symbol_split.fit_train, dataset.feature_names, settings)
        symbol_model = XGBoostMultiClassModel(settings.xgboost)
        symbol_payload = _evaluate_configuration(
            "symbol_model",
            symbol_model,
            symbol_split.fit_train,
            symbol_split.fit_validation,
            symbol_split.selection,
            symbol_split.final_test,
            symbol_feature_names,
            settings,
            symbol,
        )
        symbol_payload["feature_pruning"] = pruning_report

        comparison = {
            "global_model": global_payload,
            "symbol_model": symbol_payload,
        }
        chosen_model, chosen_payload = _choose_best_model(comparison)
        walkforward = _symbol_walkforward(
            symbol_rows=symbol_rows,
            all_rows=all_rows,
            chosen_model_name=chosen_model,
            chosen_feature_names=chosen_payload["feature_names"],
            chosen_exit_policy=chosen_payload["preferred_exit_policy"],
            settings=settings,
            symbol=symbol,
        )
        decision = _enablement_decision(symbol, len(symbol_rows), chosen_model, chosen_payload, walkforward, settings)
        block_matrix = _block_reason_matrix(symbol, len(symbol_rows), chosen_payload, walkforward, settings)
        session_timeframe_diagnostics = _session_timeframe_diagnostics(symbol, chosen_payload, settings)
        exit_policy_diagnostics = _exit_policy_diagnostics(symbol, chosen_payload)
        recommendation = _recommendation_for_symbol(decision, block_matrix)

        _record_symbol_validation_outputs(
            reports,
            symbol=symbol,
            chosen_model=chosen_model,
            chosen_payload=chosen_payload,
            comparison=comparison,
            symbol_split_reports=symbol_split.reports,
            walkforward=walkforward,
            decision=decision,
            block_matrix=block_matrix,
            session_timeframe_diagnostics=session_timeframe_diagnostics,
            exit_policy_diagnostics=exit_policy_diagnostics,
            recommendation=recommendation,
        )
        symbol_profile_overrides[symbol] = _build_symbol_profile_override(
            settings,
            symbol=symbol,
            decision=decision,
            chosen_model=chosen_model,
            chosen_payload=chosen_payload,
            run_name=run_dir.name,
        )
        _update_aggregate_portfolio(aggregate_portfolio, decision, chosen_payload)

    write_symbol_strategy_profiles(settings, common_profile, symbol_profile_overrides)
    _write_validation_reports(
        run_dir,
        global_split_reports=global_split.reports,
        symbol_enablement_report=reports["symbol_enablement_report"],
        aggregate_portfolio=aggregate_portfolio,
        reports=reports,
    )
    logger.info("strategy_validation symbols=%s enabled=%s caution=%s disabled=%s run_dir=%s", aggregate_portfolio["symbols_evaluated"], aggregate_portfolio["enabled_symbols"], aggregate_portfolio["caution_symbols"], aggregate_portfolio["disabled_symbols"], run_dir)
    return 0


def _latest_validation_run(settings: Settings) -> Path | None:
    candidates = sorted(settings.data.runs_dir.glob("*_strategy_validation"))
    return candidates[-1] if candidates else None


def compare_symbol_models(settings: Settings) -> int:
    return run_strategy_validation(settings)


def evaluate_dynamic_exits(settings: Settings) -> int:
    return run_strategy_validation(settings)


def build_symbol_profiles(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "build_symbol_profiles")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    path = settings.data.runtime_dir / settings.strategy.profiles_filename
    payload = read_artifact_payload(path, expected_type="strategy_profiles") if path.exists() else {"common": {}, "symbols": {}}
    write_json_report(run_dir, "strategy_profiles.json", payload)
    logger.info("strategy_profiles path=%s run_dir=%s", path, run_dir)
    return 0


def symbol_go_no_go(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "symbol_go_no_go")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    latest = _latest_validation_run(settings)
    if latest is None:
        logger.info("No prior validation found; running full validation")
        return run_strategy_validation(settings)
    report_path = latest / "symbol_enablement_report.json"
    if not report_path.exists():
        logger.error("symbol_enablement_report.json not found in %s", latest)
        return 1
    payload = read_artifact_payload(report_path, expected_type="symbol_enablement")
    write_json_report(run_dir, "symbol_enablement_report.json", payload)
    logger.info("symbol_go_no_go loaded_from=%s run_dir=%s", latest, run_dir)
    return 0


def audit_strategy_block_causes(settings: Settings) -> int:
    return run_strategy_validation(settings)
