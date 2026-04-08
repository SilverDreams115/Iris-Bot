# ADR 0005: Artifact provenance and demo readiness

## Status
Accepted

## Decision
Critical artifacts expose explicit provenance and demo readiness is assessed by a guarded, auditable gate rather than implicit operator judgment.

## Rationale
Operational review should not depend on ad hoc matching of run directories or memory of which experiment generated which artifact.

## Consequences

- Experiment, walk-forward, backtest, paper/demo-dry, governance, and evidence artifacts carry lineage metadata where relevant.
- Demo readiness can be evaluated without activating order routing.
- The system remains explicitly not approved for live real trading.
