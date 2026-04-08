from __future__ import annotations

import hashlib
import json
from typing import Any


TRAINING_CONTRACT_VERSION = "1.0"
EVALUATION_CONTRACT_VERSION = "1.0"


def contract_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def contract_bundle_fingerprint(
    training_contract: dict[str, Any],
    evaluation_contract: dict[str, Any],
) -> str:
    return contract_fingerprint(
        {
            "training_contract_version": TRAINING_CONTRACT_VERSION,
            "evaluation_contract_version": EVALUATION_CONTRACT_VERSION,
            "training_contract": training_contract,
            "evaluation_contract": evaluation_contract,
        }
    )
