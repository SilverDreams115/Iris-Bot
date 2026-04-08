from __future__ import annotations

from iris_bot.config import Settings
from iris_bot.demo_operational_readiness import demo_operational_readiness
from iris_bot.demo_readiness import demo_execution_readiness
from iris_bot.prolonged_serious_demo import demo_forward_runbook_command, prolonged_serious_demo_gate, demo_serious_validated_gate
from iris_bot.serious_demo_gate import serious_demo_control_gate
from iris_bot.governance import (
    active_portfolio_status,
    active_strategy_status,
    approved_demo_gate_audit,
    audit_governance_locking,
    diagnose_profile_activation,
    evidence_store_status_command,
    ingest_governance_evidence,
    list_strategy_profiles,
    materialize_active_profiles,
    repair_strategy_profile_registry,
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


def repair_strategy_profile_registry_command(settings: Settings) -> int:
    """Recomputes stale registry checksums and refreshes active materialization."""
    return repair_strategy_profile_registry(settings)


def evidence_store_status_cmd(settings: Settings) -> int:
    """Reports canonical evidence store status and integrity."""
    return evidence_store_status_command(settings)


def ingest_governance_evidence_command(settings: Settings) -> int:
    """Ingests latest lifecycle/endurance governance evidence into the canonical evidence store."""
    return ingest_governance_evidence(settings)


def approved_demo_gate_audit_command(settings: Settings) -> int:
    """Detailed gate audit showing exactly which checks pass/fail per symbol."""
    return approved_demo_gate_audit(settings)


def active_portfolio_status_command(settings: Settings) -> int:
    """Reports active portfolio with explicit universe/portfolio separation."""
    return active_portfolio_status(settings)


def demo_execution_readiness_command(settings: Settings) -> int:
    """Conservative readiness assessment for broker-executing demo (no order_send)."""
    return demo_execution_readiness(settings)


def demo_operational_readiness_command(settings: Settings) -> int:
    """Operational credibility gate: validates all 6 operational resilience blocks."""
    return demo_operational_readiness(settings)


def serious_demo_control_gate_command(settings: Settings) -> int:
    """Serious demo control gate: validates readiness for controlled serious demo execution."""
    return serious_demo_control_gate(settings)


def prolonged_serious_demo_gate_command(settings: Settings) -> int:
    """Prolonged serious demo gate: validates cumulative forward demo evidence."""
    return prolonged_serious_demo_gate(settings)


def demo_serious_validated_gate_command(settings: Settings) -> int:
    """Demo serious validated gate: cumulative multi-series forward validation."""
    return demo_serious_validated_gate(settings)


def demo_forward_runbook_runtime_command(settings: Settings) -> int:
    """Writes the short controlled execution runbook for prolonged demo validation."""
    return demo_forward_runbook_command(settings)
