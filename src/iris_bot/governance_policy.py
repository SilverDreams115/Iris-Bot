from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iris_bot.config import Settings


_BUNDLED_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "governance_policy.json"


@dataclass(frozen=True)
class SymbolGovernanceRule:
    symbol: str
    rule_id: str
    decision_reason: str
    promotion_review: str
    activation_readiness: str
    portfolio_eligibility: str
    policy_version: str
    policy_source: str

    @property
    def promotion_allowed(self) -> bool:
        return self.promotion_review != "block"

    @property
    def deliberately_blocked(self) -> bool:
        return self.portfolio_eligibility == "deliberately_blocked"

    def to_dict(self) -> dict[str, str]:
        return {
            "symbol": self.symbol,
            "rule_id": self.rule_id,
            "decision_reason": self.decision_reason,
            "promotion_review": self.promotion_review,
            "activation_readiness": self.activation_readiness,
            "portfolio_eligibility": self.portfolio_eligibility,
            "policy_version": self.policy_version,
            "policy_source": self.policy_source,
        }


def _resolve_policy_path(settings: Settings) -> Path:
    candidate = settings.project_root / "config" / settings.governance.policy_filename
    if candidate.exists():
        return candidate
    return _BUNDLED_POLICY_PATH


def _load_policy_document(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Governance policy must be a JSON object: {path}")
    symbol_rules = raw.get("symbol_rules", {})
    if not isinstance(symbol_rules, dict):
        raise ValueError(f"Governance policy symbol_rules must be an object: {path}")
    return raw


def load_governance_policy(settings: Settings) -> dict[str, Any]:
    path = _resolve_policy_path(settings)
    payload = _load_policy_document(path)
    return {
        "schema_version": int(payload.get("schema_version", 1)),
        "policy_version": str(payload.get("policy_version", "unknown")),
        "policy_source": str(path),
        "symbol_rules": payload.get("symbol_rules", {}),
    }


def symbol_governance_rule(settings: Settings, symbol: str) -> SymbolGovernanceRule | None:
    policy = load_governance_policy(settings)
    raw = policy["symbol_rules"].get(symbol)
    if not isinstance(raw, dict):
        return None
    return SymbolGovernanceRule(
        symbol=symbol,
        rule_id=str(raw.get("rule_id", f"policy_rule_for_{symbol.lower()}")),
        decision_reason=str(raw.get("decision_reason", "policy_rule_applied")),
        promotion_review=str(raw.get("promotion_review", "allow")),
        activation_readiness=str(raw.get("activation_readiness", "active")),
        portfolio_eligibility=str(raw.get("portfolio_eligibility", "eligible")),
        policy_version=str(policy["policy_version"]),
        policy_source=str(policy["policy_source"]),
    )


def deliberately_blocked_symbols(settings: Settings) -> dict[str, str]:
    policy = load_governance_policy(settings)
    blocked: dict[str, str] = {}
    for symbol, raw in policy["symbol_rules"].items():
        if not isinstance(raw, dict):
            continue
        if raw.get("portfolio_eligibility") == "deliberately_blocked":
            blocked[str(symbol)] = str(raw.get("decision_reason", "policy_blocked"))
    return blocked


def bundled_deliberately_blocked_symbols() -> dict[str, str]:
    payload = _load_policy_document(_BUNDLED_POLICY_PATH)
    blocked: dict[str, str] = {}
    symbol_rules = payload.get("symbol_rules", {})
    if not isinstance(symbol_rules, dict):
        return blocked
    for symbol, raw in symbol_rules.items():
        if not isinstance(raw, dict):
            continue
        if raw.get("portfolio_eligibility") == "deliberately_blocked":
            blocked[str(symbol)] = str(raw.get("decision_reason", "policy_blocked"))
    return blocked


def deliberately_blocked_symbol_details(settings: Settings) -> dict[str, dict[str, str]]:
    details: dict[str, dict[str, str]] = {}
    for symbol, reason in deliberately_blocked_symbols(settings).items():
        rule = symbol_governance_rule(settings, symbol)
        if rule is None:
            continue
        details[symbol] = {
            "rule_id": rule.rule_id,
            "decision_reason": reason,
            "policy_version": rule.policy_version,
            "policy_source": rule.policy_source,
        }
    return details
