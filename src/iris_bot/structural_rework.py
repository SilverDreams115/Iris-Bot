from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from statistics import pstdev
from typing import Any

from iris_bot.artifacts import wrap_artifact
from iris_bot.backtest import run_backtest_engine
from iris_bot.config import Settings
from iris_bot.demo_execution_registry import default_demo_execution_registry, save_demo_execution_registry
from iris_bot.governance import _latest_endurance_evidence, _latest_lifecycle_evidence, _lifecycle_evidence_age_hours
from iris_bot.governance_active import resolve_active_profile_entry
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.model_artifacts import (
    build_model_artifact_manifest,
    model_artifact_dir,
    write_model_artifact_manifest,
)
from iris_bot.portfolio import build_portfolio_separation
from iris_bot.processed_dataset import ProcessedRow, build_processed_dataset
from iris_bot.profile_registry import load_strategy_profile_registry
from iris_bot.quant_experiments import run_experiment_matrix
from iris_bot.splits import temporal_train_validation_test_split
from iris_bot.thresholds import apply_probability_threshold, select_threshold_from_probabilities
from iris_bot.walk_forward import generate_walk_forward_windows
from iris_bot.xgb_model import XGBoostMultiClassModel
from iris_bot.data import load_bars


_MIN_TRAIN_ROWS = 30
_MIN_VALIDATION_ROWS = 10
_MIN_TEST_ROWS = 5


def _safe_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _latest_experiment_matrix_run(settings: Settings) -> Path | None:
    candidates = sorted(settings.data.runs_dir.glob("*_experiment_matrix"))
    return candidates[-1] if candidates else None


def _load_or_build_experiment_matrix(settings: Settings) -> Path:
    latest = _latest_experiment_matrix_run(settings)
    if latest is None:
        exit_code = run_experiment_matrix(settings)
        if exit_code != 0:
            raise RuntimeError(f"run_experiment_matrix failed with exit_code={exit_code}")
        latest = _latest_experiment_matrix_run(settings)
    if latest is None:
        raise FileNotFoundError("No experiment_matrix run available")
    return latest


def _load_matrix_result(run_dir: Path, experiment_id: str) -> dict[str, Any]:
    path = run_dir / experiment_id / "experiment_result.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"experiment_result invalid in {path}")
    return payload


def _selected_structural_result(run_dir: Path) -> dict[str, Any]:
    matrix_report = json.loads((run_dir / "experiment_matrix_report.json").read_text(encoding="utf-8"))
    selected = matrix_report.get("selected_symbol_context_variant") or "exp4_symbol_specific"
    return _load_matrix_result(run_dir, str(selected))


def _effective_settings_from_result(settings: Settings, result: dict[str, Any]) -> Settings:
    effective = result["effective_config"]
    return replace(
        settings,
        labeling=replace(settings.labeling, **effective["labeling"]),
        xgboost=replace(settings.xgboost, **effective["xgboost"]),
        split=replace(settings.split, **effective["split"]),
        walk_forward=replace(settings.walk_forward, **effective["walk_forward"]),
        threshold=replace(settings.threshold, **effective["threshold"]),
        backtest=replace(settings.backtest, **effective["backtest"]),
    )


def _prepare_rows(settings: Settings) -> list[ProcessedRow]:
    bars = load_bars(settings.data.raw_dataset_path)
    if not bars:
        raise FileNotFoundError(f"Raw dataset not found: {settings.data.raw_dataset_path}")
    dataset = build_processed_dataset(bars, settings.labeling)
    rows = [row for row in dataset.rows if row.timeframe == settings.trading.primary_timeframe]
    rows.sort(key=lambda row: row.timestamp)
    return rows


def _economic_weights(rows: list[ProcessedRow], cap: float = 3.0) -> list[float]:
    atrs = [row.features.get("atr_5", 0.0) for row in rows]
    sorted_atrs = sorted(atrs)
    median_atr = sorted_atrs[len(sorted_atrs) // 2] if sorted_atrs else 0.0
    if median_atr <= 0.0:
        return [1.0] * len(rows)
    return [min(atr / median_atr, cap) for atr in atrs]


def _train_symbol_model(
    settings: Settings,
    train_rows: list[ProcessedRow],
    validation_rows: list[ProcessedRow],
    feature_names: list[str],
) -> tuple[XGBoostMultiClassModel, float, dict[str, Any]]:
    train_matrix = [[row.features[name] for name in feature_names] for row in train_rows]
    validation_matrix = [[row.features[name] for name in feature_names] for row in validation_rows]
    train_labels = [row.label for row in train_rows]
    validation_labels = [row.label for row in validation_rows]
    model = XGBoostMultiClassModel(settings.xgboost)
    model.fit(
        train_matrix,
        train_labels,
        validation_matrix,
        validation_labels,
        feature_names=feature_names,
        sample_weights=_economic_weights(train_rows),
    )
    validation_probabilities = model.predict_probabilities(validation_matrix)
    threshold_result = select_threshold_from_probabilities(
        probabilities=validation_probabilities,
        labels=validation_labels,
        grid=settings.threshold.grid,
        metric_name=settings.threshold.objective_metric,
        refinement_steps=settings.threshold.refinement_steps,
    )
    return model, threshold_result.threshold, asdict(threshold_result)


def _evaluate_rows(
    settings: Settings,
    rows: list[ProcessedRow],
    probabilities: list[dict[int, float]],
    threshold: float,
) -> dict[str, Any]:
    predictions = apply_probability_threshold(probabilities, threshold)
    metrics, _, _ = run_backtest_engine(
        rows=rows,
        probabilities=probabilities,
        threshold=threshold,
        backtest=settings.backtest,
        risk=settings.risk,
        intrabar_policy=settings.backtest.intrabar_policy,
        exit_policy_config=settings.exit_policy,
        dynamic_exit_config=settings.dynamic_exits,
    )
    no_trade_ratio = Counter(predictions).get(0, 0) / len(predictions) if predictions else 0.0
    return {
        "row_count": len(rows),
        "trade_count": metrics["total_trades"],
        "net_pnl_usd": metrics["net_pnl_usd"],
        "expectancy_usd": metrics["expectancy_usd"],
        "profit_factor": metrics["profit_factor"],
        "max_drawdown_usd": metrics["max_drawdown_usd"],
        "no_trade_ratio": no_trade_ratio,
        "threshold": threshold,
    }


def _walk_forward_symbol(
    settings: Settings,
    rows: list[ProcessedRow],
    feature_names: list[str],
) -> dict[str, Any]:
    windows = generate_walk_forward_windows(
        total_rows=len(rows),
        train_window=settings.walk_forward.train_window,
        validation_window=settings.walk_forward.validation_window,
        test_window=settings.walk_forward.test_window,
        step=settings.walk_forward.step,
    )
    summaries: list[dict[str, Any]] = []
    for window in windows:
        train_rows = rows[window.train_start : window.train_end]
        validation_rows = rows[window.validation_start : window.validation_end]
        test_rows = rows[window.test_start : window.test_end]
        if len(train_rows) < _MIN_TRAIN_ROWS or len(validation_rows) < _MIN_VALIDATION_ROWS or len(test_rows) < _MIN_TEST_ROWS:
            summaries.append({"fold_index": window.fold_index, "skipped": True, "reason": "insufficient_rows"})
            continue
        model, threshold, threshold_report = _train_symbol_model(settings, train_rows, validation_rows, feature_names)
        test_matrix = [[row.features[name] for name in feature_names] for row in test_rows]
        probabilities = model.predict_probabilities(test_matrix)
        metrics = _evaluate_rows(settings, test_rows, probabilities, threshold)
        summaries.append(
            {
                "fold_index": window.fold_index,
                "skipped": False,
                "best_iteration": model.best_iteration,
                "best_score": model.best_score,
                "threshold_report": threshold_report,
                **metrics,
            }
        )
    valid = [item for item in summaries if not item.get("skipped")]
    net_pnls = [float(item["net_pnl_usd"]) for item in valid]
    pfs = [float(item["profit_factor"]) for item in valid]
    expectancies = [float(item["expectancy_usd"]) for item in valid]
    no_trades = [float(item["no_trade_ratio"]) for item in valid]
    drawdowns = [float(item["max_drawdown_usd"]) for item in valid]
    return {
        "total_folds": len(summaries),
        "valid_folds": len(valid),
        "positive_folds": sum(1 for item in valid if float(item["net_pnl_usd"]) > 0.0),
        "fold_summaries": summaries,
        "aggregate": {
            "total_net_pnl_usd": sum(net_pnls),
            "mean_profit_factor": sum(pfs) / len(pfs) if pfs else 0.0,
            "mean_expectancy_usd": sum(expectancies) / len(expectancies) if expectancies else 0.0,
            "worst_fold_drawdown_usd": max(drawdowns) if drawdowns else 0.0,
            "mean_no_trade_ratio": sum(no_trades) / len(no_trades) if no_trades else 0.0,
            "net_pnl_stddev": pstdev(net_pnls) if len(net_pnls) > 1 else 0.0,
        },
    }


def _endurance_lifecycle_status(settings: Settings, symbol: str) -> dict[str, Any]:
    endurance = _latest_endurance_evidence(settings, symbol)
    lifecycle = _latest_lifecycle_evidence(settings)
    lifecycle_payload = (lifecycle or {}).get("payload", {})
    lifecycle_symbol = (lifecycle_payload.get("symbols", {}) or {}).get(symbol, {})
    endurance_payload = (endurance or {}).get("payload", {})
    endurance_symbol = (endurance_payload.get("symbols", {}) or {}).get(symbol, {})
    return {
        "endurance": {
            "present": endurance is not None,
            "decision": endurance_symbol.get("decision", "missing"),
            "cycles_completed": int(endurance_symbol.get("cycles_completed", 0) or 0),
        },
        "lifecycle": {
            "present": lifecycle is not None,
            "critical_mismatch_count": int(lifecycle_symbol.get("critical_mismatch_count", 0) or 0),
            "age_hours": _lifecycle_evidence_age_hours(lifecycle),
        },
    }


def _decision_for_symbol(settings: Settings, symbol: str, test_metrics: dict[str, Any], walk_forward: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    active_status = resolve_active_profile_entry(settings, symbol)
    if not active_status["ok"] or active_status.get("promotion_state") != "approved_demo":
        reasons.append("active_profile_not_approved_demo")
    evidence = _endurance_lifecycle_status(settings, symbol)
    if evidence["endurance"]["decision"] != "go":
        reasons.append("endurance_not_go")
    if evidence["endurance"]["cycles_completed"] < settings.approved_demo_gate.endurance_min_cycles:
        reasons.append("endurance_cycles_insufficient")
    if evidence["lifecycle"]["critical_mismatch_count"] > settings.approved_demo_gate.lifecycle_max_critical:
        reasons.append("lifecycle_not_clean")
    age_hours = evidence["lifecycle"]["age_hours"]
    if age_hours is None or age_hours > settings.approved_demo_gate.lifecycle_max_age_hours:
        reasons.append("lifecycle_not_recent")
    if test_metrics["trade_count"] < settings.approved_demo_gate.min_trade_count:
        reasons.append("test_trade_count_below_floor")
    if test_metrics["expectancy_usd"] <= 0.0:
        reasons.append("test_expectancy_non_positive")
    if test_metrics["profit_factor"] < settings.approved_demo_gate.min_profit_factor:
        reasons.append("test_profit_factor_below_floor")
    if test_metrics["max_drawdown_usd"] > settings.strategy.max_drawdown_usd:
        reasons.append("test_drawdown_above_floor")
    if test_metrics["no_trade_ratio"] > settings.approved_demo_gate.max_no_trade_ratio:
        reasons.append("test_no_trade_ratio_above_floor")
    aggregate = walk_forward["aggregate"]
    valid_folds = int(walk_forward["valid_folds"])
    positive_ratio = (int(walk_forward["positive_folds"]) / valid_folds) if valid_folds else 0.0
    if valid_folds < 2:
        reasons.append("walk_forward_folds_insufficient")
    if aggregate["total_net_pnl_usd"] <= 0.0:
        reasons.append("walk_forward_total_net_pnl_non_positive")
    if aggregate["mean_expectancy_usd"] <= 0.0:
        reasons.append("walk_forward_expectancy_non_positive")
    if aggregate["mean_profit_factor"] < settings.approved_demo_gate.min_profit_factor:
        reasons.append("walk_forward_profit_factor_below_floor")
    if aggregate["mean_no_trade_ratio"] > settings.strategy.caution_no_trade_ratio:
        reasons.append("walk_forward_no_trade_ratio_above_floor")
    if positive_ratio < settings.strategy.min_positive_walkforward_ratio:
        reasons.append("walk_forward_positive_ratio_below_floor")
    if aggregate["worst_fold_drawdown_usd"] > settings.strategy.max_drawdown_usd:
        reasons.append("walk_forward_drawdown_above_floor")

    if not reasons:
        return "APPROVED_FOR_DEMO_EXECUTION", reasons
    soft_reasons = {
        "walk_forward_total_net_pnl_non_positive",
        "walk_forward_expectancy_non_positive",
        "walk_forward_profit_factor_below_floor",
        "walk_forward_positive_ratio_below_floor",
    }
    if all(reason in soft_reasons for reason in reasons) and test_metrics["expectancy_usd"] > 0.0 and test_metrics["profit_factor"] >= 1.0:
        return "CANDIDATE_FOR_DEMO_EXECUTION", reasons
    return "REJECT_FOR_DEMO_EXECUTION", reasons


def run_structural_rework_evaluation(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "structural_rework")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    matrix_run_dir = _load_or_build_experiment_matrix(settings)
    selected_result = _selected_structural_result(matrix_run_dir)
    effective_settings = _effective_settings_from_result(settings, selected_result)
    rows = _prepare_rows(effective_settings)
    feature_names = list(rows[0].features.keys()) if rows else []
    separation = build_portfolio_separation(effective_settings, load_strategy_profile_registry(effective_settings))

    per_symbol_rows = {
        symbol: [row for row in rows if row.symbol == symbol]
        for symbol in sorted({row.symbol for row in rows})
    }
    symbol_reports: dict[str, Any] = {}
    demo_registry = default_demo_execution_registry()
    approved_symbol: str | None = None
    for symbol, symbol_rows in per_symbol_rows.items():
        if len(symbol_rows) < (_MIN_TRAIN_ROWS + _MIN_VALIDATION_ROWS + _MIN_TEST_ROWS):
            symbol_reports[symbol] = {
                "decision": "REJECT_FOR_DEMO_EXECUTION",
                "reasons": ["insufficient_symbol_rows"],
                "row_count": len(symbol_rows),
            }
            continue
        split = temporal_train_validation_test_split(
            symbol_rows,
            effective_settings.split.train_ratio,
            effective_settings.split.validation_ratio,
            effective_settings.split.test_ratio,
        )
        model, threshold, threshold_report = _train_symbol_model(effective_settings, split.train, split.validation, feature_names)
        test_matrix = [[row.features[name] for name in feature_names] for row in split.test]
        test_probabilities = model.predict_probabilities(test_matrix)
        test_metrics = _evaluate_rows(effective_settings, split.test, test_probabilities, threshold)
        walk_forward = _walk_forward_symbol(effective_settings, symbol_rows, feature_names)
        decision, reasons = _decision_for_symbol(effective_settings, symbol, test_metrics, walk_forward)
        active_status = resolve_active_profile_entry(effective_settings, symbol)
        base_profile_snapshot = (
            asdict(active_status["resolved_profile"]) if active_status.get("resolved_profile") is not None else {}
        )
        base_profile_snapshot.setdefault("profile_id", active_status.get("active_profile_id", ""))
        base_profile_snapshot.setdefault("promotion_state", active_status.get("promotion_state", ""))
        manifest_path = None
        if decision != "REJECT_FOR_DEMO_EXECUTION" or symbol in separation.approved_demo_universe:
            artifact_dir = model_artifact_dir(effective_settings, symbol)
            artifact_dir.mkdir(parents=True, exist_ok=True)
            model_path = artifact_dir / "xgboost_model.json"
            metadata_path = artifact_dir / "xgboost_metadata.json"
            model.save(model_path, metadata_path, feature_names)
            manifest = build_model_artifact_manifest(
                settings=effective_settings,
                symbol=symbol,
                model_path=model_path,
                metadata_path=metadata_path,
                feature_names=feature_names,
                threshold=threshold,
                threshold_metric=threshold_report["metric_name"],
                threshold_value=threshold_report["metric_value"],
                model_variant="symbol_specific",
                source_run_dir=str(matrix_run_dir),
                base_profile_snapshot=base_profile_snapshot,
                evaluation_summary={
                    "test": test_metrics,
                    "walk_forward": walk_forward["aggregate"],
                },
            )
            manifest_path = write_model_artifact_manifest(artifact_dir / "model_artifact_manifest.json", manifest)

        symbol_report = {
            "decision": decision,
            "reasons": reasons,
            "row_count": len(symbol_rows),
            "split_summary": [asdict(item) for item in split.summaries],
            "best_iteration": model.best_iteration,
            "best_score": model.best_score,
            "threshold_report": threshold_report,
            "test_metrics": test_metrics,
            "walk_forward": walk_forward,
            "active_profile_status": {key: value for key, value in active_status.items() if key != "resolved_profile"},
            "approved_demo_portfolio_member": symbol in separation.approved_demo_universe,
            "active_portfolio_member": symbol in separation.active_portfolio,
            "model_artifact_manifest_path": str(manifest_path) if manifest_path is not None else None,
        }
        symbol_reports[symbol] = symbol_report
        demo_registry["symbols"][symbol] = {
            "symbol": symbol,
            "decision": decision,
            "approved_for_demo_execution": decision == "APPROVED_FOR_DEMO_EXECUTION",
            "active_for_demo_execution": False,
            "base_profile_id": active_status.get("active_profile_id", ""),
            "base_promotion_state": active_status.get("promotion_state", ""),
            "model_variant": "symbol_specific",
            "threshold": threshold,
            "stop_policy": base_profile_snapshot.get("stop_policy", ""),
            "target_policy": base_profile_snapshot.get("target_policy", ""),
            "model_artifact_manifest_path": str(manifest_path) if manifest_path is not None else "",
            "reasons": reasons,
        }
        if decision == "APPROVED_FOR_DEMO_EXECUTION" and approved_symbol is None:
            approved_symbol = symbol

    save_demo_execution_registry(effective_settings, demo_registry)
    matrix_report = json.loads((matrix_run_dir / "experiment_matrix_report.json").read_text(encoding="utf-8"))
    recommendation = {
        "chosen_structural_variant": selected_result["experiment_id"],
        "chosen_model_mode": selected_result["model_mode"],
        "approved_symbols": sorted(
            symbol for symbol, payload in symbol_reports.items() if payload["decision"] == "APPROVED_FOR_DEMO_EXECUTION"
        ),
        "candidate_symbols": sorted(
            symbol for symbol, payload in symbol_reports.items() if payload["decision"] == "CANDIDATE_FOR_DEMO_EXECUTION"
        ),
        "reject_symbols": sorted(
            symbol for symbol, payload in symbol_reports.items() if payload["decision"] == "REJECT_FOR_DEMO_EXECUTION"
        ),
        "dominant_cause": (
            "symbol_context_structural_issue"
            if selected_result["model_mode"] == "symbol_specific"
            else "insufficient_structural_gain"
        ),
        "promotion_recommendation": "use_symbol_specific_for_demo_candidate_selection_only",
    }
    write_json_report(
        run_dir,
        "structural_model_comparison_report.json",
        wrap_artifact(
            "structural_model_comparison",
            {
                "experiment_matrix_run_dir": str(matrix_run_dir),
                "selected_symbol_context_variant": matrix_report.get("selected_symbol_context_variant"),
                "selected_xgb_variant": matrix_report.get("selected_xgb_variant"),
                "experiments": matrix_report.get("experiments", {}),
            },
        ),
    )
    write_json_report(
        run_dir,
        "demo_execution_candidate_report.json",
        wrap_artifact(
            "demo_execution_candidate",
            {
                "chosen_structural_variant": selected_result["experiment_id"],
                "symbols": symbol_reports,
                "approved_symbols": recommendation["approved_symbols"],
                "candidate_symbols": recommendation["candidate_symbols"],
            },
        ),
    )
    write_json_report(
        run_dir,
        "structural_rework_report.json",
        wrap_artifact(
            "structural_rework",
            {
                "experiment_matrix_run_dir": str(matrix_run_dir),
                "chosen_result": selected_result["experiment_id"],
                "effective_config": selected_result["effective_config"],
                "recommendation": recommendation,
                "symbols": symbol_reports,
            },
        ),
    )
    write_json_report(
        run_dir,
        "technical_debt_avoidance_report.json",
        wrap_artifact(
            "technical_debt_avoidance",
            {
                "new_runtime_registry": "demo_execution_registry.json",
                "base_registry_untouched": True,
                "paper_pipeline_unchanged": True,
                "live_real_pipeline_unchanged": True,
                "readme_modified": False,
            },
        ),
    )
    logger.info(
        "structural_rework chosen=%s approved=%s candidate=%s run_dir=%s",
        selected_result["experiment_id"],
        len(recommendation["approved_symbols"]),
        len(recommendation["candidate_symbols"]),
        run_dir,
    )
    return 0
