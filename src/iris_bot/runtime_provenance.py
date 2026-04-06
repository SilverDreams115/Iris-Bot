from __future__ import annotations

import json
import os
from typing import Any


_ENV_NAME = "IRIS_MT5_RESEARCH_PROVENANCE_JSON"


def load_runtime_provenance_from_env() -> dict[str, Any]:
    raw = os.getenv(_ENV_NAME, "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_payload_invalid": True}
    return payload if isinstance(payload, dict) else {"raw_payload_invalid": True}


def runtime_provenance_env_name() -> str:
    return _ENV_NAME
