# ADR 0004: Governance policy layer

## Status
Accepted

## Decision
Business rules that affect promotion/portfolio eligibility live in a versioned policy document:

- `config/governance_policy.json`

## Rationale
Business exclusions hardcoded inside governance logic are opaque and difficult to audit or evolve safely.

## Consequences

- Policy provenance is exposed in governance and portfolio artifacts.
- The current symbol restriction for `USDJPY` is declarative instead of embedded logic.
