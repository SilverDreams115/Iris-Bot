# IRIS-Bot Architecture

This document describes the implemented architecture, not an aspirational design.

## System Boundaries

- Research: dataset construction, labels, XGBoost training, walk-forward economic replay.
- Controlled execution: paper trading, demo-dry, resilient operational state handling.
- Governance: strategy profile registry, promotion gates, active profile materialization.
- Demo readiness: explicit readiness assessment for guarded demo execution.
- Not included: live real order routing for production capital.

## Contracted Layers

### MT5 ownership

- Ownership is explicit and policy-driven: `strict`, `compatibility`, `audit_only`.
- Operational ownership requires `magic_match` or `comment_match` under `strict`.
- Symbol-only scope is audit scope, not ownership.

### Training contract

- Source of truth: `training_contract.py`
- Covers feature ordering, economic sample weighting, class weighting, probability calibration, and training provenance.
- Persisted in experiment and consumed by downstream evaluation artifacts.

### Evaluation contract

- Source of truth: `evaluation_contract.py`
- Covers threshold selection/application, threshold-by-symbol policy, profile gating, signal timing, execution semantics, and consistency depth.
- Shared by walk-forward, backtest, and paper/demo-dry comparable paths.

## Durable Persistence

- Critical JSON/manifests use durable commit semantics: write tmp, flush, `fsync(tmp)`, `os.replace`, `fsync(parent)`.
- Critical CSV operational artifacts use the same durable write path.
- The evidence store and governance materializations rely on this primitive instead of ad hoc `write_text`.

## Evidence Store

- Canonical location: `data/runtime/evidence_store/`
- Manifest is transactional, lock-protected, and durable.
- Re-ingest semantics:
  - same `entry_id` + same checksum: idempotent
  - same `entry_id` + different checksum: conflict
  - same logical key + different provenance: preserved with explicit contradiction reporting
- Expiration is tombstoned, not silent deletion.

## Artifact Provenance

Critical artifacts expose enough metadata to reconstruct lineage without path heuristics:

- `schema_version`
- `artifact_type`
- `artifact_version`
- `training_contract_version`
- `evaluation_contract_version`
- `contract_hashes`
- `artifact_provenance.run_id`
- `artifact_provenance.source_run_id`
- `artifact_provenance.lineage_id`
- `artifact_provenance.correlation_keys`
- `artifact_provenance.references`
- `policy_version` / `policy_source` where policy is part of the decision

## Governance Policy

- Business rules that block/allow symbols live in `config/governance_policy.json`.
- The current policy layer is explicit, versioned, auditable, and consumed by governance and portfolio separation.

## Official Quality Gate

The repo-standard quality gate is:

```bash
make check
```

Equivalent explicit commands:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
.venv/bin/python -m pytest
.venv/bin/python -m iris_bot.main --help
```

## Demo Readiness

`demo-execution-readiness` is a guarded assessment. It does not send orders and does not enable live real trading. Its purpose is to decide whether the repository and its current evidence state are ready for controlled demo execution review.

Decision values:

- `ready_for_demo_guarded`
- `ready_for_demo_with_reservations`
- `not_ready_for_demo`
