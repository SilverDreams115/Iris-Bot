from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from iris_bot.artifacts import read_artifact_payload, wrap_artifact
from iris_bot.config import Settings
from iris_bot.durable_io import durable_write_json
from iris_bot.governance_active import resolve_active_profile_entry
from iris_bot.logging_utils import write_json_report
from iris_bot.xgb_model import XGBoostMultiClassModel


MODEL_ARTIFACT_SCHEMA_VERSION = 1


def model_artifact_root(settings: Settings) -> Path:
    return settings.data.runtime_dir / "demo_execution_models"


def model_artifact_dir(settings: Settings, symbol: str) -> Path:
    return model_artifact_root(settings) / symbol


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model_artifact_manifest(
    *,
    settings: Settings,
    symbol: str,
    model_path: Path,
    metadata_path: Path,
    feature_names: list[str],
    threshold: float,
    threshold_metric: str,
    threshold_value: float,
    model_variant: str,
    source_run_dir: str,
    base_profile_snapshot: dict[str, Any],
    evaluation_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
        "symbol": symbol,
        "model_variant": model_variant,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "model_sha256": _sha256_file(model_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "threshold": threshold,
        "threshold_metric": threshold_metric,
        "threshold_value": threshold_value,
        "source_run_dir": source_run_dir,
        "base_profile_snapshot": base_profile_snapshot,
        "runtime_compatibility": {
            "primary_timeframe": settings.trading.primary_timeframe,
            "stop_policy": base_profile_snapshot.get("stop_policy"),
            "target_policy": base_profile_snapshot.get("target_policy"),
        },
        "evaluation_summary": evaluation_summary,
    }


def write_model_artifact_manifest(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    durable_write_json(
        path,
        wrap_artifact(
            "model_artifact_manifest",
            payload,
            compatibility={"loader": "load_model_artifact_manifest"},
        ),
    )
    return path


def load_model_artifact_manifest(path: Path) -> dict[str, Any]:
    return read_artifact_payload(path, expected_type="model_artifact_manifest")


def _resolve_runtime_project_path(settings: Settings, path_value: str) -> Path:
    candidate = Path(path_value) if path_value else Path()
    if not path_value:
        return candidate
    if candidate.exists():
        return candidate
    normalized = path_value.replace("\\", "/")
    project_marker = f"/{settings.project_root.name}/"
    if project_marker in normalized:
        suffix = normalized.split(project_marker, 1)[1]
        remapped = settings.project_root / Path(suffix)
        if remapped.exists():
            return remapped
    for anchor in ("/data/", "/runs/", "/config/", "/src/"):
        if anchor not in normalized:
            continue
        suffix = normalized.split(anchor, 1)[1]
        remapped = settings.project_root / Path(anchor.strip("/") + "/" + suffix)
        if remapped.exists():
            return remapped
    return candidate


def validate_model_artifact(
    settings: Settings,
    *,
    symbol: str,
    manifest_path: Path,
    require_active_profile: bool = True,
) -> dict[str, Any]:
    manifest_path = _resolve_runtime_project_path(settings, str(manifest_path))
    reasons: list[str] = []
    warnings: list[str] = []
    manifest_exists = manifest_path.exists()
    manifest: dict[str, Any] = {}
    if not manifest_exists:
        reasons.append("manifest_missing")
    else:
        try:
            manifest = load_model_artifact_manifest(manifest_path)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"manifest_unreadable:{exc}")

    model_path = _resolve_runtime_project_path(settings, str(manifest.get("model_path", ""))) if manifest else Path()
    metadata_path = _resolve_runtime_project_path(settings, str(manifest.get("metadata_path", ""))) if manifest else Path()
    metadata: dict[str, Any] = {}
    if manifest:
        manifest = dict(manifest)
        manifest["model_path"] = str(model_path)
        manifest["metadata_path"] = str(metadata_path)
        if int(manifest.get("schema_version", -1)) != MODEL_ARTIFACT_SCHEMA_VERSION:
            reasons.append("manifest_schema_version_incompatible")
        if str(manifest.get("symbol", "")) != symbol:
            reasons.append("manifest_symbol_mismatch")
        if not model_path.exists():
            reasons.append("model_file_missing")
        if not metadata_path.exists():
            reasons.append("metadata_file_missing")
        if model_path.exists() and manifest.get("model_sha256") != _sha256_file(model_path):
            reasons.append("model_checksum_mismatch")
        if metadata_path.exists() and manifest.get("metadata_sha256") != _sha256_file(metadata_path):
            reasons.append("metadata_checksum_mismatch")
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                reasons.append(f"metadata_unreadable:{exc}")
        if metadata:
            if list(metadata.get("feature_names", [])) != list(manifest.get("feature_names", [])):
                reasons.append("feature_names_mismatch")

    profile_status = resolve_active_profile_entry(settings, symbol)
    if require_active_profile and not profile_status["ok"]:
        reasons.append("active_profile_invalid")
    if profile_status["ok"] and manifest:
        base_profile_snapshot = manifest.get("base_profile_snapshot", {})
        if base_profile_snapshot:
            if str(base_profile_snapshot.get("profile_id", "")) != str(profile_status.get("active_profile_id", "")):
                warnings.append("base_profile_snapshot_differs_from_current_active_profile")
            if str(base_profile_snapshot.get("promotion_state", "")) != "approved_demo":
                reasons.append("base_profile_snapshot_not_approved_demo")

    return {
        "ok": not reasons,
        "symbol": symbol,
        "manifest_path": str(manifest_path),
        "model_path": str(model_path) if manifest else "",
        "metadata_path": str(metadata_path) if manifest else "",
        "manifest": manifest,
        "metadata": metadata,
        "active_profile_status": {key: value for key, value in profile_status.items() if key != "resolved_profile"},
        "reasons": reasons,
        "warnings": warnings,
    }


def load_validated_model(
    settings: Settings,
    *,
    symbol: str,
    manifest_path: Path,
) -> tuple[XGBoostMultiClassModel | None, dict[str, Any]]:
    report = validate_model_artifact(settings, symbol=symbol, manifest_path=manifest_path, require_active_profile=True)
    if not report["ok"]:
        return None, report
    manifest = report["manifest"]
    model = XGBoostMultiClassModel(settings.xgboost)
    try:
        model.load(Path(str(manifest["model_path"])))
    except Exception as exc:  # noqa: BLE001
        report["ok"] = False
        report["reasons"].append(f"model_load_failed:{exc}")
        return None, report
    return model, report


def write_model_validation_report(run_dir: Path, filename: str, payload: dict[str, Any]) -> Path:
    return write_json_report(run_dir, filename, wrap_artifact("model_load_validation", payload))
