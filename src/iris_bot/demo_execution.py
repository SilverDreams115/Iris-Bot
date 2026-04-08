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
from iris_bot.demo_forward_evidence import build_demo_session_evidence, write_demo_session_evidence
from iris_bot.demo_live_probe import _is_demo_account
from iris_bot.demo_session_series import ensure_active_demo_session_series, record_demo_session_result
from iris_bot.exits import SymbolExitProfile, build_exit_policies
from iris_bot.governance import _latest_lifecycle_evidence, _lifecycle_evidence_age_hours
from iris_bot.governance_active import resolve_active_profile_entry
from iris_bot.kill_switch import build_default_circuit_breaker_conditions
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.model_artifacts import load_validated_model, validate_model_artifact
from iris_bot.mt5 import MT5Client, OrderRequest, OrderResult
from iris_bot.processed_dataset import ProcessedRow, _build_cross_returns_index, _compute_feature_row
from iris_bot.operational import PendingIntent
from iris_bot.resilient_state import persist_runtime_state, state_from_dict
from iris_bot.session_discipline import review_demo_session, session_startup_check
from iris_bot.thresholds import apply_probability_threshold


def _runtime_state_path(settings: Settings) -> Path:
    return settings.data.runtime_dir / settings.operational.persistence_state_filename


def _read_runtime_state(settings: Settings) -> dict[str, Any]:
    path = _runtime_state_path(settings)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _register_open_position_in_runtime_state(
    settings: Settings,
    *,
    symbol: str,
    side: str,
    session_started_at: str,
    entry: dict[str, Any],
) -> None:
    """Register an opened demo position as a pending intent in runtime_state.

    This ensures lifecycle reconciliation sees local evidence for broker activity
    and does not flag the position as broker_event_without_local_intent.
    """
    path = _runtime_state_path(settings)
    raw = _read_runtime_state(settings)
    state_payload = raw.get("state")
    try:
        state = state_from_dict(state_payload) if isinstance(state_payload, dict) else None
    except Exception:  # noqa: BLE001
        state = None

    if state is None:
        from iris_bot.operational import AccountState, PaperEngineState
        state = PaperEngineState(account_state=AccountState(1000.0, 1000.0, 1000.0))

    # Remove any stale pending intent for this symbol before adding the new one.
    state.pending_intents = [p for p in state.pending_intents if p.symbol != symbol]
    state.pending_intents.append(
        PendingIntent(
            symbol=symbol,
            created_at=session_started_at,
            signal_timestamp=session_started_at,
            side=side,
            volume_lots=float(settings.backtest.min_lot),
            active_profile_id=str(entry.get("base_profile_id", "")),
            model_variant=str(entry.get("model_variant", "")),
            promotion_state=str(entry.get("base_promotion_state", "")),
        )
    )
    persist_runtime_state(path, state, raw.get("latest_broker_sync_result") or {})


def _clear_stale_pending_intent_if_position_closed(
    settings: Settings,
    *,
    symbol: str,
    tick_snapshot: dict[str, Any],
) -> None:
    """Remove the pending intent for symbol when broker confirms no open position exists.

    Called after each tick_snapshot so that runtime_state stays consistent with broker
    reality after a position closes naturally via TP/SL between demo execution runs.
    No-ops if the position is still open or if no pending intent exists for the symbol.
    """
    if _find_position(tick_snapshot, symbol) is not None:
        return  # position still open — nothing to clean
    path = _runtime_state_path(settings)
    raw = _read_runtime_state(settings)
    state_payload = raw.get("state")
    try:
        state = state_from_dict(state_payload) if isinstance(state_payload, dict) else None
    except Exception:  # noqa: BLE001
        state = None
    if state is None:
        return
    before = len(state.pending_intents)
    state.pending_intents = [p for p in state.pending_intents if p.symbol != symbol]
    if len(state.pending_intents) < before:
        persist_runtime_state(path, state, raw.get("latest_broker_sync_result") or {})


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


def _position_lifetime_seconds(opened_position: dict[str, Any] | None, closed_trades: list[dict[str, Any]]) -> float:
    if opened_position is None:
        return 0.0
    open_time = opened_position.get("time")
    ticket = int(opened_position.get("ticket", 0) or 0)
    if not isinstance(open_time, (int, float)) or ticket <= 0:
        return 0.0
    matching_times: list[float] = []
    for trade in closed_trades:
        if int(trade.get("position_id", trade.get("position", 0)) or 0) != ticket:
            continue
        trade_time = trade.get("time")
        if isinstance(trade_time, (int, float)) and float(trade_time) >= float(open_time):
            matching_times.append(float(trade_time))
    if not matching_times:
        return 0.0
    return round(max(matching_times) - float(open_time), 4)


def _realized_pnl_usd(opened_position: dict[str, Any] | None, closed_trades: list[dict[str, Any]]) -> float:
    if opened_position is None:
        return 0.0
    ticket = int(opened_position.get("ticket", 0) or 0)
    if ticket <= 0:
        return 0.0
    pnl = 0.0
    for trade in closed_trades:
        if int(trade.get("position_id", trade.get("position", 0)) or 0) == ticket:
            pnl += float(trade.get("profit", 0.0) or 0.0)
    return round(pnl, 8)


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
    # runtime_state is the outer envelope: {"saved_at": ..., "state": {...}}
    # blocked_reasons and broker_sync_status live inside "state", not at top level.
    state_payload = runtime_state.get("state", {}) if isinstance(runtime_state, dict) else {}
    blocked_reasons = list(state_payload.get("blocked_reasons", []))
    critical_discrepancies = int(
        (state_payload.get("broker_sync_status") or {}).get("critical_discrepancy_count", 0) or 0
    )
    kill_switch_active = any(r.startswith("kill_switch:") for r in blocked_reasons)
    no_trade_active = any(
        r.startswith("no_trade_mode:") or r.startswith("kill_switch:") for r in blocked_reasons
    )
    # Circuit breaker pre-check: evaluate conditions against live state (if restorable)
    circuit_breaker_triggered = False
    circuit_breaker_reason = ""
    try:
        restored_state = state_from_dict(state_payload) if state_payload else None
        if restored_state is not None:
            conditions = build_default_circuit_breaker_conditions()
            for cond in conditions:
                if cond.check(restored_state):
                    circuit_breaker_triggered = True
                    circuit_breaker_reason = cond.reason
                    break
    except Exception:  # noqa: BLE001
        pass  # If state can't be parsed, we fall back to blocked_reasons check

    operational_ok = (
        not blocked_reasons
        and critical_discrepancies == 0
        and not kill_switch_active
        and not no_trade_active
        and not circuit_breaker_triggered
    )
    checks["operational_state"] = {
        "ok": operational_ok,
        "reason": (
            "ok" if operational_ok else
            "kill_switch_active" if kill_switch_active else
            "no_trade_mode_active" if no_trade_active else
            f"circuit_breaker_triggered:{circuit_breaker_reason}" if circuit_breaker_triggered else
            "operational_state_blocked"
        ),
        "blocked_reasons": blocked_reasons,
        "kill_switch_active": kill_switch_active,
        "no_trade_mode_active": no_trade_active,
        "circuit_breaker_triggered": circuit_breaker_triggered,
        "critical_discrepancy_count": critical_discrepancies,
    }
    checks["max_symbols_per_run"] = {
        "ok": bool(symbol),
        "reason": "ok" if symbol else "no_symbol_selected",
        "symbols_configured": 1 if symbol else 0,
        "max_symbols_per_run": 1,  # strictly one symbol per demo execution run
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
    session_started_at = datetime.now(tz=UTC).isoformat()
    session_id = run_dir.name
    startup_report = session_startup_check(settings)
    write_json_report(run_dir, "session_startup_report.json", wrap_artifact("demo_session_review", startup_report.to_dict()))
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
    session_series = ensure_active_demo_session_series(settings, symbol=symbol)
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
    # Resolve threshold: base_profile_snapshot is the primary source of truth because
    # it reflects the threshold at which the profile was validated in backtest/endurance.
    # Registry and manifest thresholds may diverge when the model was retrained or the
    # profile was updated independently.
    profile_snapshot = dict(manifest.get("base_profile_snapshot", {}))
    _registry_threshold = float(entry.get("threshold", 0) or 0)
    _manifest_threshold = float(manifest.get("threshold", 0) or 0)
    _profile_threshold = float(profile_snapshot.get("threshold", 0) or 0)
    threshold = _profile_threshold or _manifest_threshold or _registry_threshold or 1.0
    if _registry_threshold and abs(_registry_threshold - threshold) > 1e-9:
        logger.warning(
            "threshold_discrepancy symbol=%s registry=%.4f using=%.4f source=base_profile_snapshot",
            symbol, _registry_threshold, threshold,
        )
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
    # Always fetch broker state and clean up stale pending intents regardless of signal,
    # so runtime_state stays consistent with broker reality even on no-trade sessions.
    tick_snapshot = client.broker_state_snapshot((symbol,)).to_dict()
    _clear_stale_pending_intent_if_position_closed(settings, symbol=symbol, tick_snapshot=tick_snapshot)

    if signal == 0:
        logger.error("run_demo_execution blocked no_trade signal=%s symbol=%s", signal, symbol)
        return 3

    if settings.trading.one_position_per_symbol:
        existing_position = _find_position(tick_snapshot, symbol)
        if existing_position is not None:
            block_payload = {
                "ok": False,
                "reason": "position_already_open",
                "symbol": symbol,
                "existing_ticket": existing_position.get("ticket"),
                "existing_volume": existing_position.get("volume"),
                "one_position_per_symbol": True,
            }
            write_json_report(run_dir, "position_already_open_report.json", wrap_artifact("position_already_open", block_payload))
            logger.error(
                "run_demo_execution blocked position_already_open symbol=%s ticket=%s",
                symbol,
                existing_position.get("ticket"),
            )
            client.shutdown()
            return 5  # 5 = position_already_open (distinct from 3 = no_trade_signal)
    account_info = client.account_info() or {}
    terminal = client._mt5
    if terminal is None:
        logger.error("run_demo_execution blocked: terminal_not_initialized symbol=%s", symbol)
        client.shutdown()
        return 2
    tick = terminal.symbol_info_tick(symbol)
    if tick is None:
        logger.error("run_demo_execution blocked: tick_unavailable symbol=%s", symbol)
        client.shutdown()
        return 2
    entry_price = float(tick.ask) if signal == 1 else float(tick.bid)

    # Derive SL/TP using the same exit engine as the validated backtest.
    # profile_snapshot already resolved above (threshold section) — reused here.
    # Source of truth: stop_policy / target_policy from the demo execution registry entry,
    # profile parameters from the model manifest's base_profile_snapshot,
    # multipliers and floors from settings.risk / settings.backtest.
    # Features (atr_10, atr_5, rolling_volatility_*) already computed above — no extra fetch.
    exit_profile = SymbolExitProfile(
        stop_policy=str(entry.get("stop_policy", profile_snapshot.get("stop_policy", "static"))),
        target_policy=str(entry.get("target_policy", profile_snapshot.get("target_policy", "static"))),
        stop_atr_multiplier=float(profile_snapshot.get("stop_atr_multiplier", settings.risk.atr_stop_loss_multiplier)),
        target_atr_multiplier=float(profile_snapshot.get("target_atr_multiplier", settings.risk.atr_take_profit_multiplier)),
        stop_min_pct=float(profile_snapshot.get("stop_min_pct", settings.backtest.fixed_stop_loss_pct)) if "stop_min_pct" in profile_snapshot else None,
        stop_max_pct=float(profile_snapshot.get("stop_max_pct", 0.01)) if "stop_max_pct" in profile_snapshot else None,
        target_min_pct=float(profile_snapshot.get("target_min_pct", settings.backtest.fixed_take_profit_pct)) if "target_min_pct" in profile_snapshot else None,
        target_max_pct=float(profile_snapshot.get("target_max_pct", 0.02)) if "target_max_pct" in profile_snapshot else None,
    )
    stop_policy_obj, target_policy_obj = build_exit_policies(exit_profile.stop_policy, exit_profile.target_policy)
    feature_row = ProcessedRow(
        timestamp=datetime.now(tz=UTC),
        symbol=symbol,
        timeframe=settings.trading.primary_timeframe,
        open=entry_price,
        high=entry_price,
        low=entry_price,
        close=entry_price,
        volume=0.0,
        label=0,
        label_reason="runtime",
        horizon_end_timestamp="",
        features=features,
    )
    sl_level = stop_policy_obj.stop_loss_price(
        feature_row, entry_price, signal, settings.backtest, settings.risk, settings.dynamic_exits, exit_profile
    )
    tp_level = target_policy_obj.take_profit_price(
        feature_row, entry_price, signal, settings.backtest, settings.risk, settings.dynamic_exits, exit_profile
    )
    order_request = OrderRequest(
        symbol=symbol,
        side="buy" if signal == 1 else "sell",
        volume=settings.backtest.min_lot,
        stop_loss=round(sl_level.price, 8),
        take_profit=round(tp_level.price, 8),
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
    if opened_position is not None and open_result.accepted and not settings.demo_execution.auto_close_after_entry:
        _register_open_position_in_runtime_state(
            settings,
            symbol=symbol,
            side=order_request.side,
            session_started_at=session_started_at,
            entry=entry,
        )
    final_snapshot = client.broker_state_snapshot((symbol,)).to_dict()
    closed_trades = list(final_snapshot.get("closed_trades", []))
    order_trace: dict[str, object] = {
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
        "exit_policy": {
            "stop_policy": exit_profile.stop_policy,
            "target_policy": exit_profile.target_policy,
            "sl_details": sl_level.details,
            "tp_details": tp_level.details,
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
    reconciliation: dict[str, object] = {
        "ok": open_result.accepted and ((opened_position is not None) or open_status in {"filled", "sent"}),
        "symbol": symbol,
        "intent_matches_symbol": str(open_result.request.get("symbol", "")) == symbol,
        "intent_matches_side": str(open_result.request.get("type", "")) != "",
        "position_found_after_open": opened_position is not None,
        "auto_close_applied": settings.demo_execution.auto_close_after_entry,
        "remaining_positions": final_snapshot.get("positions", []),
    }
    execution_report: dict[str, object] = {
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
    session_evidence = build_demo_session_evidence(
        session_id=session_id,
        session_series_id=session_series.session_series_id,
        series_position=len(session_series.session_ids) + 1,
        symbol=symbol,
        start_time=session_started_at,
        end_time=datetime.now(tz=UTC).isoformat(),
        preflight_report=preflight,
        signals_evaluated=1,
        no_trade_signals=0,
        blocked_signals=0,
        trades_opened=1 if open_result.accepted else 0,
        trades_closed=1 if close_result_payload is not None and bool(close_result_payload.get("accepted", False)) else 0,
        orders_sent=1,
        orders_rejected=0 if open_result.accepted else 1,
        session_decision_log=[
            {
                "action": "execute",
                "reason": "signal_passed_threshold",
                "signal": signal,
                "threshold": threshold,
            }
        ],
        final_state_summary={
            "kill_switch_active": bool(preflight["checks"]["operational_state"].get("kill_switch_active", False)),
            "circuit_breaker_triggered": bool(preflight["checks"]["operational_state"].get("circuit_breaker_triggered", False)),
            "remaining_positions": len(final_snapshot.get("positions", [])),
            "blocked_reasons": list(preflight["checks"]["operational_state"].get("blocked_reasons", [])),
        },
        signal_summary={
            "signals_generated": 1,
            "generated_signal": signal,
            "probabilities": probabilities,
        },
        execution_summary={
            "orders_sent": 1,
            "orders_rejected": 0 if open_result.accepted else 1,
            "decisions_executed": 1 if open_result.accepted else 0,
            "open_status": open_status,
            "close_status": close_result_payload.get("status") if close_result_payload else None,
        },
        trade_summary={
            "trades_opened": 1 if open_result.accepted else 0,
            "trades_closed": 1 if close_result_payload is not None and bool(close_result_payload.get("accepted", False)) else 0,
            "opened_position_ticket": opened_position.get("ticket") if opened_position else None,
        },
        performance_summary={
            "realized_pnl_usd": _realized_pnl_usd(opened_position, closed_trades),
            "position_lifetime_seconds": _position_lifetime_seconds(opened_position, closed_trades),
        },
        divergence_summary={
            "divergence_events": 0 if reconciliation["ok"] else 1,
            "reconcile_events": 1,
            "broker_local_divergences": [] if reconciliation["ok"] else ["post_trade_reconciliation_failed"],
        },
        restore_recovery_summary={
            "restore_events": 1 if startup_report.details.get("restore") else 0,
            "recovery_events": 0,
            "recovery_details": [],
        },
        artifact_paths={
            "preflight_report": str(run_dir / "demo_broker_execution_preflight_report.json"),
            "execution_report": str(run_dir / "demo_execution_report.json"),
            "reconciliation_report": str(run_dir / "post_trade_reconciliation_report.json"),
        },
    )
    session_evidence_path = run_dir / "demo_session_evidence.json"
    write_demo_session_evidence(session_evidence_path, session_evidence)
    session_review = review_demo_session(
        session_id=session_id,
        session_series_id=session_series.session_series_id,
        session_evidence={**session_evidence.to_dict(), "artifact_paths": {**session_evidence.artifact_paths, "session_evidence": str(session_evidence_path)}},
        startup_report=startup_report,
    )
    session_review_path = write_json_report(run_dir, "demo_session_review.json", wrap_artifact("demo_session_review", session_review.to_dict()))
    series_update = record_demo_session_result(
        settings,
        session_series_id=session_series.session_series_id,
        session_id=session_id,
        session_evidence_path=session_evidence_path,
        session_review_path=session_review_path,
        session_evidence_payload={**session_evidence.to_dict(), "artifact_paths": {**session_evidence.artifact_paths, "session_evidence": str(session_evidence_path)}},
        session_review_payload=session_review.to_dict(),
    )
    write_json_report(run_dir, "demo_session_series_report.json", wrap_artifact("demo_session_series", series_update))
    logger.info("run_demo_execution symbol=%s open_status=%s run_dir=%s", symbol, open_status, run_dir)
    client.shutdown()
    return 0 if reconciliation["ok"] else 4
