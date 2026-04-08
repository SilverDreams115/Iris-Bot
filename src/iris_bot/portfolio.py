"""
Portfolio and universe separation for IRIS-Bot.

Problem solved:
  active_strategy_status mixes USDJPY (deliberately blocked out-of-scope) with
  EURUSD/GBPUSD/AUDUSD (actively approved for demo). This makes global status
  reports noisy and misleading — a symbol blocked by design looks the same as
  a symbol blocked due to failed validation.

Solution:
  Explicit separation between:
    full_universe        — all symbols configured in settings.trading.symbols
    eligible_universe    — symbols not deliberately excluded (e.g., USDJPY out-of-scope)
    approved_demo        — symbols with active approved_demo profile in registry
    active_portfolio     — symbols currently enabled and approved (the operational set)
    deliberately_blocked — symbols excluded by design with reason

Key invariant:
  active_portfolio ⊆ approved_demo ⊆ eligible_universe ⊆ full_universe
  deliberately_blocked ∩ active_portfolio = ∅ (always)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from iris_bot.config import Settings
from iris_bot.governance_policy import (
    bundled_deliberately_blocked_symbols,
    deliberately_blocked_symbol_details,
    deliberately_blocked_symbols,
    load_governance_policy,
)


_PERMANENTLY_EXCLUDED: dict[str, str] = bundled_deliberately_blocked_symbols()


@dataclass(frozen=True)
class PortfolioSeparation:
    full_universe: tuple[str, ...]
    eligible_universe: tuple[str, ...]
    approved_demo_universe: tuple[str, ...]
    active_portfolio: tuple[str, ...]
    deliberately_blocked: dict[str, str]  # symbol → reason
    deliberately_blocked_details: dict[str, dict[str, str]]
    registry_active_profiles: dict[str, str]  # symbol → profile_id
    policy_version: str
    policy_source: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_universe": list(self.full_universe),
            "eligible_universe": list(self.eligible_universe),
            "approved_demo_universe": list(self.approved_demo_universe),
            "active_portfolio": list(self.active_portfolio),
            "deliberately_blocked": dict(self.deliberately_blocked),
            "deliberately_blocked_details": {
                symbol: dict(details) for symbol, details in self.deliberately_blocked_details.items()
            },
            "registry_active_profiles": dict(self.registry_active_profiles),
            "policy_version": self.policy_version,
            "policy_source": self.policy_source,
            "generated_at": self.generated_at,
            "counts": {
                "full_universe": len(self.full_universe),
                "eligible_universe": len(self.eligible_universe),
                "approved_demo_universe": len(self.approved_demo_universe),
                "active_portfolio": len(self.active_portfolio),
                "deliberately_blocked": len(self.deliberately_blocked),
            },
        }

    def portfolio_impact_of_blocked(self) -> dict[str, bool]:
        """
        For each deliberately blocked symbol, indicates whether it would
        otherwise be in the active portfolio. Useful for understanding
        whether a blockage is suppressing real operational capacity.
        """
        return {
            symbol: symbol in self.approved_demo_universe or symbol in self.active_portfolio
            for symbol in self.deliberately_blocked
        }


def build_portfolio_separation(
    settings: Settings,
    registry: dict[str, Any],
) -> PortfolioSeparation:
    """
    Builds an explicit PortfolioSeparation from current settings and registry state.

    Does not perform I/O beyond reading the provided registry dict.
    """
    full_universe = tuple(settings.trading.symbols)
    policy = load_governance_policy(settings)

    # Eligible = not permanently excluded
    blocked_map = deliberately_blocked_symbols(settings)
    eligible_universe = tuple(
        s for s in full_universe if s not in blocked_map
    )

    # deliberately_blocked = all permanently excluded symbols that are in the full universe
    deliberately_blocked = {
        s: reason
        for s, reason in blocked_map.items()
        if s in full_universe
    }
    deliberately_blocked_details = {
        s: details
        for s, details in deliberately_blocked_symbol_details(settings).items()
        if s in full_universe
    }

    # approved_demo = symbols that have an active profile with promotion_state == "approved_demo"
    registry_active_profiles: dict[str, str] = {}
    approved_demo_universe: list[str] = []
    for symbol in eligible_universe:
        active_id = registry.get("active_profiles", {}).get(symbol, "")
        if not active_id:
            continue
        registry_active_profiles[symbol] = active_id
        # Find the entry
        for entry in registry.get("profiles", {}).get(symbol, []):
            if entry.get("profile_id") == active_id:
                if entry.get("promotion_state") == "approved_demo":
                    approved_demo_universe.append(symbol)
                break

    # active_portfolio = approved_demo AND profile is enabled (enabled_state == "enabled")
    active_portfolio: list[str] = []
    for symbol in approved_demo_universe:
        active_id = registry.get("active_profiles", {}).get(symbol, "")
        for entry in registry.get("profiles", {}).get(symbol, []):
            if entry.get("profile_id") == active_id:
                pp = entry.get("profile_payload", {})
                if pp.get("enabled_state") in ("enabled", "caution"):
                    active_portfolio.append(symbol)
                break

    return PortfolioSeparation(
        full_universe=full_universe,
        eligible_universe=eligible_universe,
        approved_demo_universe=tuple(approved_demo_universe),
        active_portfolio=tuple(active_portfolio),
        deliberately_blocked=deliberately_blocked,
        deliberately_blocked_details=deliberately_blocked_details,
        registry_active_profiles=registry_active_profiles,
        policy_version=str(policy["policy_version"]),
        policy_source=str(policy["policy_source"]),
        generated_at=datetime.now(tz=UTC).isoformat(),
    )


def active_portfolio_status_report(
    settings: Settings,
    registry: dict[str, Any],
) -> dict[str, Any]:
    """
    Returns a focused status report for the active portfolio only.

    This report excludes deliberately blocked symbols (they are listed separately)
    and provides clear per-symbol status for operationally relevant symbols only.
    """
    separation = build_portfolio_separation(settings, registry)

    per_symbol: dict[str, Any] = {}
    for symbol in separation.eligible_universe:
        active_id = registry.get("active_profiles", {}).get(symbol, "")
        entry = None
        if active_id:
            for e in registry.get("profiles", {}).get(symbol, []):
                if e.get("profile_id") == active_id:
                    entry = e
                    break

        in_approved = symbol in separation.approved_demo_universe
        in_active = symbol in separation.active_portfolio

        per_symbol[symbol] = {
            "in_approved_demo_universe": in_approved,
            "in_active_portfolio": in_active,
            "active_profile_id": active_id or None,
            "promotion_state": (entry or {}).get("promotion_state", "none"),
            "enabled_state": ((entry or {}).get("profile_payload", {}) or {}).get("enabled_state", "unknown"),
            "status": (
                "active" if in_active else
                "approved_not_active" if in_approved else
                "not_approved"
            ),
        }

    return {
        "portfolio_separation": separation.to_dict(),
        "per_symbol_eligible": per_symbol,
        "deliberately_blocked": separation.deliberately_blocked,
        "deliberately_blocked_details": separation.deliberately_blocked_details,
        "policy_context": {
            "policy_version": separation.policy_version,
            "policy_source": separation.policy_source,
        },
        "portfolio_impact_of_blocked": separation.portfolio_impact_of_blocked(),
        "summary": {
            "active_portfolio_size": len(separation.active_portfolio),
            "approved_demo_size": len(separation.approved_demo_universe),
            "eligible_size": len(separation.eligible_universe),
            "full_universe_size": len(separation.full_universe),
            "blocked_by_design": len(separation.deliberately_blocked),
        },
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }


def active_universe_status_report(
    settings: Settings,
    registry: dict[str, Any],
) -> dict[str, Any]:
    """
    Returns a comprehensive status report covering the full universe.

    Unlike active_portfolio_status_report, this includes deliberately blocked
    symbols with explicit blocking reasons, making it clear to auditors which
    symbols exist but are intentionally excluded from the portfolio.
    """
    separation = build_portfolio_separation(settings, registry)
    portfolio_report = active_portfolio_status_report(settings, registry)

    all_symbols: dict[str, Any] = {}
    for symbol in separation.full_universe:
        if symbol in separation.deliberately_blocked:
            all_symbols[symbol] = {
                "universe_category": "deliberately_blocked",
                "blocking_reason": separation.deliberately_blocked[symbol],
                "policy_rule": separation.deliberately_blocked_details.get(symbol, {}),
                "would_affect_portfolio": symbol in separation.approved_demo_universe or symbol in separation.active_portfolio,
                "in_eligible_universe": False,
                "in_approved_demo": False,
                "in_active_portfolio": False,
            }
        else:
            eligible_status = portfolio_report["per_symbol_eligible"].get(symbol, {})
            all_symbols[symbol] = {
                "universe_category": eligible_status.get("status", "unknown"),
                "in_eligible_universe": True,
                "in_approved_demo": symbol in separation.approved_demo_universe,
                "in_active_portfolio": symbol in separation.active_portfolio,
                "active_profile_id": eligible_status.get("active_profile_id"),
                "promotion_state": eligible_status.get("promotion_state", "none"),
                "enabled_state": eligible_status.get("enabled_state", "unknown"),
            }

    return {
        "full_universe": list(separation.full_universe),
        "per_symbol": all_symbols,
        "summary": portfolio_report["summary"],
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "policy_context": {
            "policy_version": separation.policy_version,
            "policy_source": separation.policy_source,
        },
    }
