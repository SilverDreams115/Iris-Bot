from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_bot.artifacts import wrap_artifact
from iris_bot.config import Settings
from iris_bot.data import group_bars
from iris_bot.demo_execution_registry import (
    activate_demo_execution_symbol,
    load_demo_execution_registry,
)
from iris_bot.demo_live_probe import _is_demo_account
from iris_bot.governance import _latest_lifecycle_evidence, _lifecycle_evidence_age_hours
from iris_bot.governance_active import resolve_active_profile_entry
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.model_artifacts import load_validated_model, validate_model_artifact
from iris_bot.mt5 import MT5Client, OrderRequest, OrderResult
from iris_bot.operational import atomic_write_json
from iris_bot.processed_dataset import _build_cross_returns_index, _compute_feature_row
from iris_bot.thresholds import apply_probability_threshold


def _runtime_state_path(settings: Settings) -> Path:
    return settings.data.runtime_dir / settings.operational.persistence_state_filename


def _read_runtime_state(settings: Settings) -> dict[str, Any]:
    path = _runtime_state_path(settings)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _active_demo_symbol(settings: Settings) -> tuple[str, dict[str, Any], dict[str, Any]]:
    registry = load_demo_execution_registry(settings)
    target = settings.demo_execution.target_symbol or str(registry.get("active_symbol", ""))
    if not target:
        return "", {}, registry
    entry = dict((registry.get("symbols", {}) or {}).get(target, {}))
    return target, entry, registry


def _latest_row_features(settings: Settings, client: MT5Client, symbol: str, feature_names: list[str]) -> tuple[dict[str, float], dict[str, Any]]:
    timeframe = settings.trading.primary_timeframe
    bars: list[Any] = []
    for other_symbol in settings.trading.symbols:
        bars.extend(client.fetch_historical_bars(other_symbol, timeframe, settings.mt5.history_bars))
    grouped = group_bars(bars)
    series = grouped.get((symbol, timeframe), [])
    if len(series) < 20:
        raise RuntimeError(f"insufficient_history_for_features:{symbol}:{len(series)}")
    cross_returns = _build_cross_returns_index(grouped)
    index = len(series) - 1
    ts_iso = series[index].timestamp.isoformat()
    features = _compute_feature_row(series, index, cross_returns=cross_returns.get(ts_iso, {}))
    runtime_info = {
        "timestamp": ts_iso,
        "symbol": symbol,
        "timeframe": timeframe,
        "available_feature_names": sorted(features.keys()),
    }
    missing = [name for name in feature_names if name not in features]
    if missing:
        raise RuntimeError(f"missing_runtime_features:{','.join(missing)}")
    return {name: features[name] for name in feature_names}, runtime_info


def _classify_order_result(result: OrderResult) -> str:
    comment = (result.comment or "").lower()
    if result.accepted:
        if result.volume is not None and result.volume > 0.0:
            return "filled"
        return "sent"
    if "requote" in comment:
        return "requote"
    if "invalid" in comment or result.retcode in {10013, 10014, 10015, 10016}:
        return "invalid"
    if "partial" in comment:
        return "partially_filled"
    if "reject" in comment or result.retcode in {10006}:
        return "rejected"
    return "broker_error"


def _find_position(snapshot: dict[str, Any], symbol: str) -> dict[str, Any] | None:
    for position in snapshot.get("positions", []):
        if str(position.get("symbol", "")) == symbol:
            return dict(position)
    return None


def validate_model_artifact_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "validate_model_artifact")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    symbol, entry, _ = _active_demo_symbol(settings)
    if not symbol:
        payload = {"ok": False, "reasons": ["no_demo_execution_target_symbol"]}
        write_json_report(run_dir, "model_load_validation_report.json", wrap_artifact("model_load_validation", payload))
        logger.error("validate_model_artifact blocked: no_demo_execution_target_symbol")
        return 2
    report = validate_model_artifact(
        settings,
        symbol=symbol,
        manifest_path=Path(str(entry.get("model_artifact_manifest_path", ""))),
    )
    write_json_report(run_dir, "model_load_validation_report.json", wrap_artifact("model_load_validation", report))
    logger.info("validate_model_artifact symbol=%s ok=%s run_dir=%s", symbol, report["ok"], run_dir)
    return 0 if report["ok"] else 2


def activate_demo_execution_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "activate_demo_execution")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    if not settings.demo_execution.enabled:
        payload = {"ok": False, "reason": "demo_execution_disabled_in_config"}
        write_json_report(run_dir, "demo_execution_status_report.json", wrap_artifact("demo_execution_status", payload))
        logger.error("activate_demo_execution blocked: demo_execution_disabled_in_config")
        return 2
    if not settings.demo_execution.target_symbol:
        payload = {"ok": False, "reason": "demo_execution_target_symbol_missing"}
        write_json_report(run_dir, "demo_execution_status_report.json", wrap_artifact("demo_execution_status", payload))
        logger.error("activate_demo_execution blocked: demo_execution_target_symbol_missing")
        return 2
    ok, payload = activate_demo_execution_symbol(settings, settings.demo_execution.target_symbol)
    write_json_report(run_dir, "demo_execution_status_report.json", wrap_artifact("demo_execution_status", payload))
    logger.info("activate_demo_execution symbol=%s ok=%s run_dir=%s", settings.demo_execution.target_symbol, ok, run_dir)
    return 0 if ok else 2


def demo_execution_status_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "demo_execution_status")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    registry = load_demo_execution_registry(settings)
    latest_runs = sorted(settings.data.runs_dir.glob("*_run_demo_execution"))
    payload = {
        "registry": registry,
        "latest_demo_execution_run": str(latest_runs[-1]) if latest_runs else None,
    }
    write_json_report(run_dir, "demo_execution_status_report.json", wrap_artifact("demo_execution_status", payload))
    logger.info("demo_execution_status active_symbol=%s run_dir=%s", registry.get("active_symbol", ""), run_dir)
    return 0


def demo_execution_preflight_payload(settings: Settings, client: MT5Client | None = None) -> dict[str, Any]:
    symbol, entry, registry = _active_demo_symbol(settings)
    checks: dict[str, dict[str, Any]] = {}
    checks["config_enabled"] = {
        "ok": settings.demo_execution.enabled,
        "reason": "ok" if settings.demo_execution.enabled else "demo_execution_disabled_in_config",
    }
    checks["target_symbol"] = {
        "ok": bool(symbol),
        "reason": "ok" if symbol else "no_demo_execution_target_symbol",
    }
    checks["registry_gate"] = {
        "ok": bool(registry.get("gate_open", False) and entry.get("active_for_demo_execution", False)),
        "reason": "ok" if bool(registry.get("gate_open", False) and entry.get("active_for_demo_execution", False)) else "final_demo_execution_gate_closed",
    }
    checks["approved_for_demo_execution"] = {
        "ok": entry.get("decision") == "APPROVED_FOR_DEMO_EXECUTION",
        "reason": "ok" if entry.get("decision") == "APPROVED_FOR_DEMO_EXECUTION" else "symbol_not_approved_for_demo_execution",
    }
    if symbol:
        model_report = validate_model_artifact(
            settings,
            symbol=symbol,
            manifest_path=Path(str(entry.get("model_artifact_manifest_path", ""))),
        )
    else:
        model_report = {"ok": False, "reasons": ["no_symbol_selected"]}
    checks["model_artifact"] = {
        "ok": model_report.get("ok", False),
        "reason": "ok" if model_report.get("ok", False) else ",".join(model_report.get("reasons", ["model_validation_failed"])),
    }
    active_profile = resolve_active_profile_entry(settings, symbol) if symbol else {"ok": False, "reasons": ["no_symbol_selected"]}
    checks["active_profile"] = {
        "ok": active_profile.get("ok", False) and active_profile.get("promotion_state") == "approved_demo",
        "reason": "ok" if active_profile.get("ok", False) and active_profile.get("promotion_state") == "approved_demo" else "active_profile_invalid",
    }
    lifecycle = _latest_lifecycle_evidence(settings)
    lifecycle_age = _lifecycle_evidence_age_hours(lifecycle)
    checks["lifecycle"] = {
        "ok": lifecycle is not None and lifecycle_age is not None and lifecycle_age <= settings.approved_demo_gate.lifecycle_max_age_hours,
        "reason": "ok" if lifecycle is not None and lifecycle_age is not None and lifecycle_age <= settings.approved_demo_gate.lifecycle_max_age_hours else "lifecycle_missing_or_stale",
    }
    runtime_state = _read_runtime_state(settings)
    blocked_reasons = list(runtime_state.get("blocked_reasons", [])) if isinstance(runtime_state, dict) else []
    critical_discrepancies = int((((runtime_state.get("broker_sync_status") or {}).get("critical_discrepancy_count", 0)) if isinstance(runtime_state, dict) else 0) or 0)
    checks["operational_state"] = {
        "ok": not blocked_reasons and critical_discrepancies == 0,
        "reason": "ok" if not blocked_reasons and critical_discrepancies == 0 else "operational_state_blocked",
    }

    broker_checks: dict[str, Any] = {"connected": False}
    if client is None:
        client = MT5Client(settings.mt5)
    broker_connected = client.connect()
    broker_checks["connected"] = broker_connected
    if broker_connected:
        account_info = client.account_info() or {}
        is_demo, demo_indicators = _is_demo_account(account_info)
        broker_checks["account_info"] = account_info
        broker_checks["demo_guard"] = {"ok": is_demo, "indicators": demo_indicators}
        checks["demo_account"] = {"ok": is_demo, "reason": "ok" if is_demo else "account_not_demo"}
        if symbol:
            validation = client.check((symbol,))
            symbol_report = validation.symbols.get(symbol, {})
            issues = symbol_report.get("issues", [])
            checks["symbol_validation"] = {
                "ok": validation.connected and not issues,
                "reason": "ok" if validation.connected and not issues else "symbol_validation_failed",
                "details": symbol_report,
            }
            try:
                if symbol and model_report.get("ok", False):
                    manifest = model_report["manifest"]
                    features, runtime_info = _latest_row_features(settings, client, symbol, list(manifest.get("feature_names", [])))
                    checks["inference_ready"] = {
                        "ok": True,
                        "reason": "ok",
                        "runtime_info": runtime_info,
                        "feature_count": len(features),
                    }
                else:
                    checks["inference_ready"] = {"ok": False, "reason": "model_not_validated"}
            except Exception as exc:  # noqa: BLE001
                checks["inference_ready"] = {"ok": False, "reason": f"inference_feature_build_failed:{exc}"}
        else:
            checks["symbol_validation"] = {"ok": False, "reason": "no_symbol_selected"}
            checks["inference_ready"] = {"ok": False, "reason": "no_symbol_selected"}
    else:
        checks["demo_account"] = {"ok": False, "reason": "mt5_connect_failed"}
        checks["symbol_validation"] = {"ok": False, "reason": "mt5_connect_failed"}
        checks["inference_ready"] = {"ok": False, "reason": "mt5_connect_failed"}

    ok = all(item.get("ok", False) for item in checks.values())
    return {
        "ok": ok,
        "symbol": symbol,
        "registry_entry": entry,
        "checks": checks,
        "model_load_validation": model_report,
        "broker_checks": broker_checks,
        "blocked_for_demo_execution": not ok,
    }


def demo_execution_preflight_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "demo_execution_preflight")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    payload = demo_execution_preflight_payload(settings)
    write_json_report(
        run_dir,
        "demo_broker_execution_preflight_report.json",
        wrap_artifact("demo_broker_execution_preflight", payload),
    )
    write_json_report(
        run_dir,
        "inference_preflight_report.json",
        wrap_artifact("inference_preflight", payload.get("checks", {}).get("inference_ready", {})),
    )
    logger.info("demo_execution_preflight symbol=%s ok=%s run_dir=%s", payload.get("symbol", ""), payload["ok"], run_dir)
    return 0 if payload["ok"] else 2


def run_demo_execution_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "run_demo_execution")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    client = MT5Client(settings.mt5)
    preflight = demo_execution_preflight_payload(settings, client=client)
    write_json_report(
        run_dir,
        "demo_broker_execution_preflight_report.json",
        wrap_artifact("demo_broker_execution_preflight", preflight),
    )
    if not preflight["ok"]:
        logger.error("run_demo_execution blocked symbol=%s", preflight.get("symbol", ""))
        return 2

    symbol = str(preflight["symbol"])
    entry = dict(preflight["registry_entry"])
    manifest_path = Path(str(entry.get("model_artifact_manifest_path", "")))
    model, model_report = load_validated_model(settings, symbol=symbol, manifest_path=manifest_path)
    write_json_report(run_dir, "model_load_validation_report.json", wrap_artifact("model_load_validation", model_report))
    if model is None:
        logger.error("run_demo_execution blocked model_invalid symbol=%s", symbol)
        return 2

    manifest = model_report["manifest"]
    feature_names = list(manifest["feature_names"])
    features, runtime_info = _latest_row_features(settings, client, symbol, feature_names)
    matrix = [[features[name] for name in feature_names]]
    probabilities = model.predict_probabilities(matrix)[0]
    threshold = float(entry.get("threshold", manifest.get("threshold", 1.0)))
    signal = apply_probability_threshold([probabilities], threshold)[0]
    inference_payload = {
        "ok": signal != 0,
        "symbol": symbol,
        "signal": signal,
        "threshold": threshold,
        "probabilities": probabilities,
        "runtime_info": runtime_info,
    }
    write_json_report(run_dir, "inference_preflight_report.json", wrap_artifact("inference_preflight", inference_payload))
    if signal == 0:
        logger.error("run_demo_execution blocked no_trade signal=%s symbol=%s", signal, symbol)
        return 3

    tick_snapshot = client.broker_state_snapshot((symbol,)).to_dict()
    account_info = client.account_info() or {}
    entry_price = None
    if signal == 1:
        entry_price = float((client._mt5.symbol_info_tick(symbol)).ask)  # noqa: SLF001
    else:
        entry_price = float((client._mt5.symbol_info_tick(symbol)).bid)  # noqa: SLF001
    stop_distance = max(entry_price * 0.0010, 0.0005)
    take_profit_distance = max(entry_price * 0.0015, 0.0008)
    stop_loss = entry_price - stop_distance if signal == 1 else entry_price + stop_distance
    take_profit = entry_price + take_profit_distance if signal == 1 else entry_price - take_profit_distance
    order_request = OrderRequest(
        symbol=symbol,
        side="buy" if signal == 1 else "sell",
        volume=settings.backtest.min_lot,
        stop_loss=round(stop_loss, 8),
        take_profit=round(take_profit, 8),
        price=round(entry_price, 8),
        deviation=settings.demo_execution.deviation_points,
    )
    open_result = client.send_market_order(order_request)
    open_status = _classify_order_result(open_result)
    after_open_snapshot = client.broker_state_snapshot((symbol,)).to_dict()
    opened_position = _find_position(after_open_snapshot, symbol)
    close_result_payload: dict[str, Any] | None = None
    if opened_position is not None and settings.demo_execution.auto_close_after_entry:
        close_result = client.close_position(
            ticket=int(opened_position.get("ticket", 0)),
            symbol=symbol,
            volume=float(opened_position.get("volume", settings.backtest.min_lot)),
            side="buy" if int(opened_position.get("type", 0)) == 0 else "sell",
        )
        close_result_payload = {**close_result.to_dict(), "status": _classify_order_result(close_result)}
    final_snapshot = client.broker_state_snapshot((symbol,)).to_dict()
    order_trace = {
        "symbol": symbol,
        "account_info": account_info,
        "request": {
            "symbol": order_request.symbol,
            "side": order_request.side,
            "volume": order_request.volume,
            "price": order_request.price,
            "sl": order_request.stop_loss,
            "tp": order_request.take_profit,
            "deviation": settings.demo_execution.deviation_points,
            "magic_number": settings.mt5.magic_number,
            "comment_tag": settings.mt5.comment_tag,
        },
        "open_result": {**open_result.to_dict(), "status": open_status},
        "opened_position": opened_position,
        "close_result": close_result_payload,
        "snapshots": {
            "before_open": tick_snapshot,
            "after_open": after_open_snapshot,
            "final": final_snapshot,
        },
    }
    reconciliation = {
        "ok": open_result.accepted and ((opened_position is not None) or open_status in {"filled", "sent"}),
        "symbol": symbol,
        "intent_matches_symbol": str(open_result.request.get("symbol", "")) == symbol,
        "intent_matches_side": str(open_result.request.get("type", "")) != "",
        "position_found_after_open": opened_position is not None,
        "auto_close_applied": settings.demo_execution.auto_close_after_entry,
        "remaining_positions": final_snapshot.get("positions", []),
    }
    execution_report = {
        "ok": reconciliation["ok"],
        "symbol": symbol,
        "signal": signal,
        "probabilities": probabilities,
        "threshold": threshold,
        "model_variant": entry.get("model_variant", ""),
        "open_status": open_status,
        "open_result": open_result.to_dict(),
        "close_result": close_result_payload,
    }
    write_json_report(run_dir, "broker_order_trace.json", wrap_artifact("broker_order_trace", order_trace))
    write_json_report(run_dir, "live_intent_to_broker_mapping.json", order_trace)
    write_json_report(run_dir, "post_trade_reconciliation_report.json", wrap_artifact("post_trade_reconciliation", reconciliation))
    write_json_report(run_dir, "demo_execution_report.json", wrap_artifact("demo_execution", execution_report))
    logger.info("run_demo_execution symbol=%s open_status=%s run_dir=%s", symbol, open_status, run_dir)
    client.shutdown()
    return 0 if reconciliation["ok"] else 4
