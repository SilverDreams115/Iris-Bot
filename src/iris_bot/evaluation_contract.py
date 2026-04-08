from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Collection

from iris_bot.config import Settings
from iris_bot.contract_versions import EVALUATION_CONTRACT_VERSION, contract_fingerprint
from iris_bot.processed_dataset import ProcessedDataset, ProcessedRow
from iris_bot.symbols import SymbolStrategyProfile, row_allowed_by_profile


@dataclass(frozen=True)
class ExperimentReference:
    run_dir: Path
    model_path: Path
    report_path: Path
    threshold: float
    threshold_metric: str
    threshold_value: float
    feature_names: list[str]
    test_start_timestamp: str
    test_end_timestamp: str
    training_contract: dict[str, Any] | None = None
    evaluation_contract: dict[str, Any] | None = None
    training_contract_version: str | None = None
    evaluation_contract_version: str | None = None
    training_contract_hash: str | None = None
    evaluation_contract_hash: str | None = None


def locate_experiment_reference(settings: Settings) -> ExperimentReference:
    if settings.backtest.experiment_run_dir:
        run_dir = Path(settings.backtest.experiment_run_dir)
    else:
        candidates = sorted(settings.data.runs_dir.glob("*_experiment"))
        if not candidates:
            raise FileNotFoundError("No experiment runs found")
        run_dir = candidates[-1]

    report_path = run_dir / "experiment_report.json"
    model_path = run_dir / "models" / "xgboost_model.json"
    if not report_path.exists():
        raise FileNotFoundError(f"experiment_report.json not found in {run_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"xgboost_model.json not found in {run_dir}")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    xgb_section = payload.get("xgboost")
    if not isinstance(xgb_section, dict):
        raise FileNotFoundError("XGBoost metadata missing from experiment_report")
    threshold_section = xgb_section.get("threshold")
    if not isinstance(threshold_section, dict):
        raise FileNotFoundError("XGBoost threshold metadata missing from experiment_report")
    split_summary = payload.get("split_summary")
    if not isinstance(split_summary, list):
        raise FileNotFoundError("split_summary missing from experiment_report")
    test_summary = next(item for item in split_summary if item["name"] == "test")
    contract_hashes = payload.get("contract_hashes", {})
    if not isinstance(contract_hashes, dict):
        contract_hashes = {}
    return ExperimentReference(
        run_dir=run_dir,
        model_path=model_path,
        report_path=report_path,
        threshold=float(threshold_section["threshold"]),
        threshold_metric=str(threshold_section["metric_name"]),
        threshold_value=float(threshold_section["metric_value"]),
        feature_names=list(payload["feature_names"]),
        test_start_timestamp=str(test_summary["start_timestamp"]),
        test_end_timestamp=str(test_summary["end_timestamp"]),
        training_contract=payload.get("training_contract") if isinstance(payload.get("training_contract"), dict) else None,
        evaluation_contract=payload.get("evaluation_contract") if isinstance(payload.get("evaluation_contract"), dict) else None,
        training_contract_version=str(payload.get("training_contract_version")) if payload.get("training_contract_version") else None,
        evaluation_contract_version=str(payload.get("evaluation_contract_version")) if payload.get("evaluation_contract_version") else None,
        training_contract_hash=str(contract_hashes.get("training_contract")) if contract_hashes.get("training_contract") else None,
        evaluation_contract_hash=str(contract_hashes.get("evaluation_contract")) if contract_hashes.get("evaluation_contract") else None,
    )


def filter_reference_rows(
    dataset: ProcessedDataset,
    settings: Settings,
    reference: ExperimentReference,
) -> list[ProcessedRow]:
    test_start = datetime.fromisoformat(reference.test_start_timestamp)
    test_end = datetime.fromisoformat(reference.test_end_timestamp)
    rows = [
        row
        for row in dataset.rows
        if row.timeframe == settings.trading.primary_timeframe
        and row.symbol in set(settings.trading.symbols)
        and test_start <= row.timestamp <= test_end
    ]
    rows.sort(key=lambda r: (r.timestamp, r.symbol))
    return rows


def filter_rows_by_symbol_profiles(
    rows: list[ProcessedRow],
    symbol_profiles: dict[str, SymbolStrategyProfile],
    *,
    allowed_states: Collection[str],
) -> list[ProcessedRow]:
    tradable_states = set(allowed_states)
    return [
        row
        for row in rows
        if symbol_profiles.get(row.symbol) is None
        or (
            symbol_profiles[row.symbol].enabled_state in tradable_states
            and row_allowed_by_profile(symbol_profiles[row.symbol], row.timestamp, row.timeframe)
        )
    ]


def resolve_threshold_by_symbol(
    base_threshold: float,
    symbol_profiles: dict[str, SymbolStrategyProfile],
) -> dict[str, float]:
    return {
        symbol: max(base_threshold, profile.threshold)
        for symbol, profile in symbol_profiles.items()
    }


def build_evaluation_contract(
    settings: Settings,
    *,
    evaluation_mode: str,
    threshold_source: str,
    threshold: float,
    threshold_metric: str,
    threshold_value: float,
    threshold_by_symbol: dict[str, float],
    profile_gating_mode: str,
    allowed_profile_states: Collection[str],
    consistency_requires_equity_curve: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": EVALUATION_CONTRACT_VERSION,
        "evaluation_mode": evaluation_mode,
        "threshold_selection": {
            "source": threshold_source,
            "objective_metric": settings.threshold.objective_metric,
            "threshold": threshold,
            "selected_metric_name": threshold_metric,
            "selected_metric_value": threshold_value,
            "grid": list(settings.threshold.grid),
            "refinement_steps": settings.threshold.refinement_steps,
        },
        "threshold_application": {
            "policy": "max_global_and_profile_threshold" if threshold_by_symbol else "global_threshold_only",
            "base_threshold": threshold,
            "threshold_by_symbol": threshold_by_symbol,
        },
        "signal_generation": {
            "policy": "probability_threshold_multiclass",
            "entry_timing": "next_bar_open",
            "allow_long": settings.trading.allow_long and settings.backtest.allow_long,
            "allow_short": settings.trading.allow_short and settings.backtest.allow_short,
            "one_position_per_symbol": settings.trading.one_position_per_symbol,
        },
        "profile_gating": {
            "mode": profile_gating_mode,
            "allowed_states": sorted(set(allowed_profile_states)),
        },
        "execution_semantics": {
            "intrabar_policy": settings.backtest.intrabar_policy,
            "max_holding_bars": settings.backtest.max_holding_bars,
            "stop_policy": settings.exit_policy.stop_policy,
            "target_policy": settings.exit_policy.target_policy,
        },
        "consistency_policy": {
            "requires_equity_curve": consistency_requires_equity_curve,
            "checks": "verify_engine_consistency",
        },
    }
    payload["contract_hash"] = contract_fingerprint(payload)
    return payload
