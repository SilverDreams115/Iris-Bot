# ADR 0002: Training and evaluation contracts

## Status
Accepted

## Decision
Training and evaluation semantics are versioned explicitly and persisted in artifacts.

- `training_contract.py`
- `evaluation_contract.py`

## Rationale
Research and operational replay cannot be compared safely if weighting, calibration, thresholding, or signal semantics drift by route.

## Consequences

- Experiment and walk-forward share training semantics.
- Walk-forward, backtest, and paper/demo-dry share evaluation semantics for comparable paths.
- Artifacts persist contract versions and hashes for post-mortem reconstruction.
