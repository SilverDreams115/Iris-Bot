from __future__ import annotations

from iris_bot.config import Settings
from iris_bot.demo_readiness import demo_execution_readiness
from iris_bot.governance import (
    active_portfolio_status,
    active_strategy_status,
    approved_demo_gate_audit,
    audit_governance_locking,
    diagnose_profile_activation,
    evidence_store_status_command,
    list_strategy_profiles,
    materialize_active_profiles,
    promote_strategy_profile,
    review_approved_demo_readiness,
    rollback_strategy_profile,
    validate_strategy_profiles,
)


def list_strategy_profiles_command(settings: Settings) -> int:
    return list_strategy_profiles(settings)


def validate_strategy_profile_command(settings: Settings) -> int:
    return validate_strategy_profiles(settings)


def promote_strategy_profile_command(settings: Settings) -> int:
    return promote_strategy_profile(settings)


def review_approved_demo_readiness_command(settings: Settings) -> int:
    return review_approved_demo_readiness(settings)


def rollback_strategy_profile_command(settings: Settings) -> int:
    return rollback_strategy_profile(settings)


def active_strategy_status_command(settings: Settings) -> int:
    return active_strategy_status(settings)


def diagnose_profile_activation_command(settings: Settings) -> int:
    return diagnose_profile_activation(settings)


def audit_governance_consistency_command(settings: Settings) -> int:
    return diagnose_profile_activation(settings)


def symbol_reactivation_readiness_command(settings: Settings) -> int:
    return diagnose_profile_activation(settings)


# ---------------------------------------------------------------------------
# Phase 4 blindaje commands
# ---------------------------------------------------------------------------

def audit_governance_locking_command(settings: Settings) -> int:
    """Audits registry lock state and integrity."""
    return audit_governance_locking(settings)


def materialize_active_profiles_command(settings: Settings) -> int:
    """Materializes active_strategy_profiles.json from approved_demo entries only."""
    return materialize_active_profiles(settings)


def evidence_store_status_cmd(settings: Settings) -> int:
    """Reports canonical evidence store status and integrity."""
    return evidence_store_status_command(settings)


def approved_demo_gate_audit_command(settings: Settings) -> int:
    """Detailed gate audit showing exactly which checks pass/fail per symbol."""
    return approved_demo_gate_audit(settings)


def active_portfolio_status_command(settings: Settings) -> int:
    """Reports active portfolio with explicit universe/portfolio separation."""
    return active_portfolio_status(settings)


def demo_execution_readiness_command(settings: Settings) -> int:
    """Conservative readiness assessment for broker-executing demo (no order_send)."""
    return demo_execution_readiness(settings)
