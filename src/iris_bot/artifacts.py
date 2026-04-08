from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


ARTIFACT_SCHEMA_VERSIONS = {
    "strategy_profiles": 2,
    "strategy_validation": 2,
    "symbol_enablement": 2,
    "model_comparison": 2,
    "threshold_report": 2,
    "dynamic_exit_report": 2,
    "corrective_audit": 1,
    "strategy_profile_registry": 1,
    "strategy_profile_validation": 1,
    "strategy_profile_promotion": 1,
    "active_strategy_status": 1,
    "symbol_endurance": 1,
    "enabled_symbols_soak": 1,
    "symbol_stability": 1,
    "lifecycle_reconciliation": 1,
    # Phase 4 blindaje artifacts
    "active_strategy_profiles": 1,
    "evidence_store_manifest": 1,
    "governance_lock_audit": 1,
    "approved_demo_gate_audit": 1,
    "active_portfolio_status": 1,
    "active_universe_status": 1,
    "active_strategy_materialization": 1,
    "demo_execution_readiness": 1,
    "technical_debt_avoidance": 1,
    "structural_rework": 1,
    "structural_model_comparison": 1,
    "demo_execution_candidate": 1,
    "model_artifact_manifest": 1,
    "model_load_validation": 1,
    "inference_preflight": 1,
    "demo_broker_execution_preflight": 1,
    "demo_execution_registry": 1,
    "demo_execution": 1,
    "broker_order_trace": 1,
    "post_trade_reconciliation": 1,
    "demo_execution_status": 1,
    "demo_session_evidence": 1,
    "demo_session_series": 1,
    "demo_session_review": 1,
    "demo_series_review": 1,
    "demo_forward_runbook": 1,
    "prolonged_serious_demo_gate": 1,
    # Symbol-focused quantitative rework artifacts
    "symbol_focus_diagnostic": 1,
    "structural_variant_comparison": 1,
    "feature_signal_analysis": 1,
    "label_exit_interaction": 1,
    "walkforward_stability": 1,
    "threshold_trade_density": 1,
    "structural_rework_recommendation": 1,
    "expanded_history_report": 1,
    "regime_feature_diagnostic_report": 1,
    "regime_aware_experiment_matrix_report": 1,
    "per_regime_performance_report": 1,
    "symbol_focus_rework_report": 1,
    "symbol_secondary_comparison_report": 1,
    "demo_execution_candidate_report": 1,
    "structural_rework_recommendation_report": 1,
    "mt5_research_runtime_report": 1,
    "mt5_research_preflight_report": 1,
    "mt5_research_execution_report": 1,
    "environment_provenance_report": 1,
    "baseline_edge_diagnostic_report": 1,
    "label_noise_report": 1,
    "horizon_exit_alignment_report": 1,
    "regime_value_report": 1,
    "class_separability_report": 1,
    "costly_error_analysis_report": 1,
    "hypothesis_matrix_report": 1,
    "edge_diagnosis_recommendation_report": 1,
    "horizon_alignment_report": 1,
    "timeout_label_impact_report": 1,
    "tp_sl_alignment_report": 1,
    "label_horizon_exit_matrix_report": 1,
    "trade_duration_distribution_report": 1,
    "threshold_utility_report": 1,
    "edge_realignment_recommendation_report": 1,
    "exit_lifecycle_diagnostic_report": 1,
    "timeout_timeexit_reduction_report": 1,
    "h12_exit_variant_matrix_report": 1,
    "trade_count_preservation_report": 1,
    "edge_lifecycle_recommendation_report": 1,
    "governance_evidence_ingest": 1,
}


@dataclass(frozen=True)
class ArtifactEnvelope:
    artifact_type: str
    schema_version: int
    generated_at: str
    artifact_version: str
    checksum: str
    payload: dict[str, Any]
    compatibility: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "artifact_version": self.artifact_version,
            "generated_at": self.generated_at,
            "checksum": self.checksum,
            "compatibility": self.compatibility,
            "provenance": self.provenance,
            "payload": self.payload,
        }


def _checksum_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_json_dict(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return cast(dict[str, Any], raw)


def build_artifact_provenance(
    *,
    run_dir: Path | None = None,
    source_run_id: str | None = None,
    parent_run_id: str | None = None,
    lineage_id: str | None = None,
    training_contract_version: str | None = None,
    evaluation_contract_version: str | None = None,
    policy_version: str | None = None,
    policy_source: str | None = None,
    contract_hashes: dict[str, str] | None = None,
    correlation_keys: dict[str, str] | None = None,
    references: dict[str, str] | None = None,
    integrity: dict[str, str] | None = None,
    materialized_at: str | None = None,
    ingested_at: str | None = None,
) -> dict[str, Any]:
    run_id = run_dir.name if run_dir is not None else ""
    resolved_source_run_id = source_run_id or run_id
    resolved_lineage_id = lineage_id or parent_run_id or resolved_source_run_id or run_id
    return {
        "run_id": run_id,
        "source_run_id": resolved_source_run_id,
        "parent_run_id": parent_run_id or "",
        "lineage_id": resolved_lineage_id,
        "training_contract_version": training_contract_version,
        "evaluation_contract_version": evaluation_contract_version,
        "policy_version": policy_version,
        "policy_source": policy_source,
        "contract_hashes": dict(contract_hashes or {}),
        "correlation_keys": dict(correlation_keys or {}),
        "references": dict(references or {}),
        "integrity": dict(integrity or {}),
        "materialized_at": materialized_at or datetime.now(tz=UTC).isoformat(),
        "ingested_at": ingested_at,
    }


def attach_artifact_provenance(payload: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(payload)
    enriched["artifact_provenance"] = provenance
    return enriched


def wrap_artifact(
    artifact_type: str,
    payload: dict[str, Any],
    compatibility: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    version = ARTIFACT_SCHEMA_VERSIONS.get(artifact_type, 1)
    envelope = ArtifactEnvelope(
        artifact_type=artifact_type,
        schema_version=version,
        generated_at=datetime.now(tz=UTC).isoformat(),
        artifact_version=f"{artifact_type}.v{version}",
        checksum=_checksum_payload(payload),
        payload=payload,
        compatibility=compatibility or {},
        provenance=provenance or {},
    )
    return envelope.to_dict()


def read_artifact_payload(path: Path, expected_type: str | None = None) -> dict[str, Any]:
    payload = _load_json_dict(path)
    if "artifact_type" not in payload:
        return payload
    if expected_type is not None and payload.get("artifact_type") != expected_type:
        raise ValueError(f"artifact_type mismatch: expected {expected_type}, got {payload.get('artifact_type')}")
    checksum = payload.get("checksum")
    current = _checksum_payload(payload.get("payload", {}))
    if checksum != current:
        raise ValueError(f"artifact checksum mismatch for {path}")
    return cast(dict[str, Any], payload.get("payload", {}))


def artifact_schema_report(path: Path, expected_type: str | None = None) -> dict[str, Any]:
    payload = _load_json_dict(path)
    if "artifact_type" not in payload:
        return {
            "path": str(path),
            "versioned": False,
            "ok": True,
            "artifact_type": expected_type,
            "schema_version": None,
        }
    ok = expected_type is None or payload.get("artifact_type") == expected_type
    checksum_ok = payload.get("checksum") == _checksum_payload(payload.get("payload", {}))
    return {
        "path": str(path),
        "versioned": True,
        "ok": ok and checksum_ok,
        "artifact_type": payload.get("artifact_type"),
        "schema_version": payload.get("schema_version"),
        "artifact_version": payload.get("artifact_version"),
        "checksum_ok": checksum_ok,
        "compatibility": payload.get("compatibility", {}),
        "provenance": payload.get("provenance", {}),
    }
