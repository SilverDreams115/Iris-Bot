#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: scripts/run_mt5_research_windows.sh <validate-runtime|fetch-extended-history|audit-regime-features|run-regime-aware-rework|compare-regime-experiments|evaluate-demo-candidate>" >&2
  exit 2
fi

COMMAND="$1"
case "$COMMAND" in
  validate-runtime|fetch-extended-history|audit-regime-features|run-regime-aware-rework|compare-regime-experiments|evaluate-demo-candidate) ;;
  *)
    echo "unsupported MT5 research command: $COMMAND" >&2
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WINDOWS_WORKSPACE_WIN="${IRIS_MT5_RESEARCH_WINDOWS_WORKSPACE:-C:\\Temp\\IRIS-Bot-mt5-runtime\\workspace}"
WINDOWS_WORKSPACE_WSL="${IRIS_MT5_RESEARCH_WINDOWS_WORKSPACE_WSL:-/mnt/c/Temp/IRIS-Bot-mt5-runtime/workspace}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOCK_FILE="$PROJECT_ROOT/.mt5_research_windows.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "mt5 research runner is already active: $LOCK_FILE" >&2
  exit 3
fi

BRIDGE_RUN_DIR="$("$PROJECT_ROOT/.venv/bin/python" - <<'PY'
from iris_bot.config import load_settings
from iris_bot.logging_utils import build_run_directory
settings = load_settings()
print(build_run_directory(settings.data.runs_dir, "mt5_research_bridge").as_posix())
PY
)"

mkdir -p "$BRIDGE_RUN_DIR"

WSL_MT5_IMPORTABLE="$("$PROJECT_ROOT/.venv/bin/python" - <<'PY'
from importlib.util import find_spec
print(bool(find_spec("MetaTrader5")))
PY
)"

export IRIS_MT5_RESEARCH_BRIDGE_RUN_DIR="$BRIDGE_RUN_DIR"
export IRIS_MT5_RESEARCH_COMMAND="$COMMAND"
export IRIS_MT5_RESEARCH_PROJECT_ROOT="$PROJECT_ROOT"
export IRIS_MT5_RESEARCH_WINDOWS_WORKSPACE_WIN="$WINDOWS_WORKSPACE_WIN"
export IRIS_MT5_RESEARCH_WINDOWS_WORKSPACE_WSL="$WINDOWS_WORKSPACE_WSL"
export IRIS_MT5_RESEARCH_WSL_MT5_IMPORTABLE="$WSL_MT5_IMPORTABLE"

"$PROJECT_ROOT/.venv/bin/python" - <<'PY'
import json
import os
from iris_bot.artifacts import wrap_artifact

payload = {
    "command": os.environ["IRIS_MT5_RESEARCH_COMMAND"],
    "workspace_root_windows": os.environ["IRIS_MT5_RESEARCH_WINDOWS_WORKSPACE_WIN"],
    "workspace_root_wsl": os.environ["IRIS_MT5_RESEARCH_WINDOWS_WORKSPACE_WSL"],
    "wsl_runtime": {
        "python_executable": os.environ["IRIS_MT5_RESEARCH_PROJECT_ROOT"] + "/.venv/bin/python",
        "project_root": os.environ["IRIS_MT5_RESEARCH_PROJECT_ROOT"],
        "metatrader5_importable": os.environ["IRIS_MT5_RESEARCH_WSL_MT5_IMPORTABLE"] == "True",
    },
    "windows_runtime": {
        "evidence_source": "synced_windows_preflight_report",
    },
    "root_cause": {
        "wsl_python_missing_metatrader5": os.environ["IRIS_MT5_RESEARCH_WSL_MT5_IMPORTABLE"] != "True",
        "primary_blockers": [
            item for item, active in (
                ("wsl_python_missing_metatrader5", os.environ["IRIS_MT5_RESEARCH_WSL_MT5_IMPORTABLE"] != "True"),
                ("unc_repo_execution_is_not_canonical", True),
                ("python_subprocess_windows_interop_is_not_canonical", True),
            ) if active
        ],
        "canonical_recommendation": "shell_wrapper_wsl_tar_bridge_to_windows_native_python",
    },
}
path = os.path.join(os.environ["IRIS_MT5_RESEARCH_BRIDGE_RUN_DIR"], "mt5_research_runtime_report.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(wrap_artifact("mt5_research_runtime_report", payload), handle, indent=2, sort_keys=True)
PY

"$PROJECT_ROOT/.venv/bin/python" - <<'PY'
import os
import shutil
workspace = os.environ["IRIS_MT5_RESEARCH_WINDOWS_WORKSPACE_WSL"]
if os.path.exists(workspace):
    shutil.rmtree(workspace)
os.makedirs(workspace, exist_ok=True)
PY
cp -r "$PROJECT_ROOT/src" "$WINDOWS_WORKSPACE_WSL/"
cp -r "$PROJECT_ROOT/data/raw" "$WINDOWS_WORKSPACE_WSL/data_raw"
cp -r "$PROJECT_ROOT/data/processed" "$WINDOWS_WORKSPACE_WSL/data_processed"
cp "$PROJECT_ROOT/pyproject.toml" "$WINDOWS_WORKSPACE_WSL/"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  cp "$PROJECT_ROOT/.env" "$WINDOWS_WORKSPACE_WSL/"
fi
"$PROJECT_ROOT/.venv/bin/python" - <<'PY'
import os
import shutil
workspace = os.environ["IRIS_MT5_RESEARCH_WINDOWS_WORKSPACE_WSL"]
os.makedirs(os.path.join(workspace, "data"), exist_ok=True)
src_raw = os.path.join(workspace, "data_raw")
src_processed = os.path.join(workspace, "data_processed")
dst_raw = os.path.join(workspace, "data", "raw")
dst_processed = os.path.join(workspace, "data", "processed")
if os.path.exists(dst_raw):
    shutil.rmtree(dst_raw)
if os.path.exists(dst_processed):
    shutil.rmtree(dst_processed)
shutil.move(src_raw, dst_raw)
shutil.move(src_processed, dst_processed)
for suffix in (
    "_run_regime_aware_rework",
    "_audit_regime_features",
    "_fetch_extended_history",
    "_compare_regime_experiments",
    "_evaluate_demo_candidate",
):
    for item in sorted(os.listdir(os.path.join(os.environ["IRIS_MT5_RESEARCH_PROJECT_ROOT"], "runs"))):
        if item.endswith(suffix):
            source = os.path.join(os.environ["IRIS_MT5_RESEARCH_PROJECT_ROOT"], "runs", item)
            target_dir = os.path.join(workspace, "runs")
            os.makedirs(target_dir, exist_ok=True)
            target = os.path.join(target_dir, item)
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.copytree(source, target)
PY

RUNTIME_PROVENANCE_JSON="$("$PROJECT_ROOT/.venv/bin/python" - <<'PY'
import json
import os
from datetime import UTC, datetime
payload = {
    "host_runtime": "shell_wrapper_wsl_tar_bridge_to_windows_native_python",
    "command": os.environ["IRIS_MT5_RESEARCH_COMMAND"],
    "project_root_wsl": os.environ["IRIS_MT5_RESEARCH_PROJECT_ROOT"],
    "workspace_root_windows": os.environ["IRIS_MT5_RESEARCH_WINDOWS_WORKSPACE_WIN"],
    "workspace_root_wsl": os.environ["IRIS_MT5_RESEARCH_WINDOWS_WORKSPACE_WSL"],
    "python_executable_wsl": os.environ["IRIS_MT5_RESEARCH_PROJECT_ROOT"] + "/.venv/bin/python",
    "python_executable_windows": "collected_by_windows_runtime",
    "sync_mode": "wsl_tar_bridge",
    "invoked_at": datetime.now(tz=UTC).isoformat(),
}
print(json.dumps(payload, sort_keys=True))
PY
)"

WINDOWS_RUN_MODE="run"
WINDOWS_DELEGATED_COMMAND="$COMMAND"
if [[ "$COMMAND" == "validate-runtime" ]]; then
  WINDOWS_RUN_MODE="validate-runtime"
  WINDOWS_DELEGATED_COMMAND="fetch-extended-history"
fi

WINDOWS_EXEC_SCRIPT="\$env:PYTHONPATH = '$WINDOWS_WORKSPACE_WIN\\src'; \$env:IRIS_MT5_RESEARCH_PROVENANCE_JSON = '$RUNTIME_PROVENANCE_JSON'; Set-Location '$WINDOWS_WORKSPACE_WIN'; python -m iris_bot.mt5_research_runtime $WINDOWS_RUN_MODE --delegated-command '$WINDOWS_DELEGATED_COMMAND' --run-id '$RUN_ID'"
set +e
powershell.exe -NoProfile -Command "$WINDOWS_EXEC_SCRIPT"
WINDOWS_EXIT_CODE=$?
set -e

MARKER_PATH="$WINDOWS_WORKSPACE_WSL/.mt5_research_last_execution.json"
if [[ ! -f "$MARKER_PATH" ]]; then
  echo "missing execution marker: $MARKER_PATH" >&2
  exit 1
fi

export MARKER_PATH
EXEC_RUN_DIR_NAME="$("$PROJECT_ROOT/.venv/bin/python" - <<'PY'
import json
import os
with open(os.environ["MARKER_PATH"], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload["run_dir_name"])
print(payload.get("delegated_run_dir_name") or "")
print(payload["exit_code"])
PY
)"
EXEC_RUN_DIR="$(printf '%s\n' "$EXEC_RUN_DIR_NAME" | sed -n '1p')"
DELEGATED_RUN_DIR="$(printf '%s\n' "$EXEC_RUN_DIR_NAME" | sed -n '2p')"
DELEGATED_EXIT_CODE="$(printf '%s\n' "$EXEC_RUN_DIR_NAME" | sed -n '3p')"

SYNC_ITEMS=("runs/$EXEC_RUN_DIR")
if [[ -n "$DELEGATED_RUN_DIR" ]]; then
  SYNC_ITEMS+=("runs/$DELEGATED_RUN_DIR")
fi
if [[ "$COMMAND" == "fetch-extended-history" ]]; then
  SYNC_ITEMS+=("data/raw/market_extended.csv" "data/raw/market_extended.csv.metadata.json")
fi

SYNC_JOINED=""
for item in "${SYNC_ITEMS[@]}"; do
  if [[ -n "$SYNC_JOINED" ]]; then
    SYNC_JOINED+=" "
  fi
  SYNC_JOINED+="$item"
done

export SYNC_JOINED
"$PROJECT_ROOT/.venv/bin/python" - <<'PY'
import os
import shutil

project_root = os.environ["IRIS_MT5_RESEARCH_PROJECT_ROOT"]
workspace = os.environ["IRIS_MT5_RESEARCH_WINDOWS_WORKSPACE_WSL"]
items = os.environ["SYNC_JOINED"].split()
for item in items:
    source = os.path.join(workspace, item)
    target = os.path.join(project_root, item)
    if not os.path.exists(source):
        continue
    if os.path.isdir(source):
        if os.path.exists(target):
            shutil.rmtree(target)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copytree(source, target)
    else:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(source, target)
PY

export EXEC_RUN_DIR
export DELEGATED_RUN_DIR
export WINDOWS_EXIT_CODE
export DELEGATED_EXIT_CODE
"$PROJECT_ROOT/.venv/bin/python" - <<'PY'
import json
import os
from iris_bot.artifacts import wrap_artifact

payload = {
    "command": os.environ["IRIS_MT5_RESEARCH_COMMAND"],
    "workspace_root_windows": os.environ["IRIS_MT5_RESEARCH_WINDOWS_WORKSPACE_WIN"],
    "execution_run_dir_name": os.environ["EXEC_RUN_DIR"],
    "delegated_run_dir_name": os.environ["DELEGATED_RUN_DIR"],
    "windows_exit_code": int(os.environ["WINDOWS_EXIT_CODE"]),
    "delegated_exit_code": int(os.environ["DELEGATED_EXIT_CODE"]),
}
path = os.path.join(os.environ["IRIS_MT5_RESEARCH_BRIDGE_RUN_DIR"], "mt5_research_execution_report.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(wrap_artifact("mt5_research_execution_report", payload), handle, indent=2, sort_keys=True)
path = os.path.join(os.environ["IRIS_MT5_RESEARCH_BRIDGE_RUN_DIR"], "technical_debt_avoidance_report.json")
with open(path, "w", encoding="utf-8") as handle:
    json.dump(
        wrap_artifact(
            "technical_debt_avoidance",
            {
                "single_recommended_flow": "shell_wrapper_wsl_tar_bridge_to_windows_native_python",
                "avoided_shortcuts": [
                    "no_unc_repo_execution",
                    "no_python_subprocess_bridge_to_windows",
                    "no_duplicate_quant_logic_in_wrapper",
                    "no_secret_persistence_in_repo",
                ],
            },
        ),
        handle,
        indent=2,
        sort_keys=True,
    )
PY

exit "$DELEGATED_EXIT_CODE"
