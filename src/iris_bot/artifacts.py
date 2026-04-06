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
}


@dataclass(frozen=True)
class ArtifactEnvelope:
    artifact_type: str
    schema_version: int
    generated_at: str
    checksum: str
    payload: dict[str, Any]
    compatibility: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "checksum": self.checksum,
            "compatibility": self.compatibility,
            "payload": self.payload,
        }


def _checksum_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _load_json_dict(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return cast(dict[str, Any], raw)


def wrap_artifact(
    artifact_type: str,
    payload: dict[str, Any],
    compatibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    version = ARTIFACT_SCHEMA_VERSIONS.get(artifact_type, 1)
    envelope = ArtifactEnvelope(
        artifact_type=artifact_type,
        schema_version=version,
        generated_at=datetime.now(tz=UTC).isoformat(),
        checksum=_checksum_payload(payload),
        payload=payload,
        compatibility=compatibility or {},
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
        "checksum_ok": checksum_ok,
        "compatibility": payload.get("compatibility", {}),
    }
