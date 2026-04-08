# ADR 0001: MT5 ownership policy

## Status
Accepted

## Decision
Broker records are classified with explicit ownership semantics:

- `owned_by_bot`
- `in_symbol_scope`
- `visible_for_audit`
- `ownership_reason`

Policy modes:

- `strict`
- `compatibility`
- `audit_only`

## Rationale
Symbol coincidence is not a safe operational ownership signal on mixed accounts.

## Consequences

- `strict` is the hardened operational mode.
- `compatibility` preserves legacy behavior for rollout safety.
- Audit reports can explain why a record was visible or treated as owned.
