# IRIS-Bot

IRIS-Bot is a research, validation, governance, and controlled-execution framework for FX trading on MetaTrader 5.

At this point the project is no longer just "an ML bot". The codebase currently implements:

- a data and feature pipeline built from MT5 history
- XGBoost multiclass training
- economic backtesting and walk-forward evaluation
- paper trading and resilient execution modes with persisted state
- evidence-gated strategy profile lifecycle management
- controlled demo execution with preflight checks and explicit activation

It is not a production live-trading system for real capital.

## Current State

State observed in this workspace on **April 8, 2026**:

- The official quality suite completed successfully and is recorded in `runs/20260408T055455Z_demo_execution_readiness/`:
  `ruff`, `mypy`, `pytest`, and `smoke`.
- That report records `587 passed`.
- The materialized runtime currently has **3 active `approved_demo` profiles**:
  `EURUSD`, `GBPUSD`, and `AUDUSD`.
- Governance policy blocks `USDJPY` from promotion due to insufficient detected edge.
- The demo execution registry currently leaves **only `EURUSD` approved and active** for that phase.
- There are recent artifacts for `demo_execution_preflight`, `activate-demo-execution`,
  `run-demo-execution`, `lifecycle_reconciliation`, and serious-demo gating.

That places the project in a **strong research + controlled demo operations** phase,
with explicit restrictions and evidence requirements, not an unrestricted automation phase.

## What The Project Is

### 1. Research layer

Implemented flow:

```text
MT5 history
  -> fetch / fetch-historical
  -> validate-data
  -> build-dataset
  -> run-experiment
  -> run-backtest / run-backtest --walk-forward
```

This includes:

- versioned processed datasets with schema and manifest
- triple-barrier-style labeling and related variants
- native XGBoost training
- configurable threshold selection
- comparison of global and per-symbol variants
- additional evaluation work around regime logic, exits, labeling, and signal quality

### 2. Operational execution layer

The repository implements several execution levels:

- `run-paper`
- `run-paper-resilient`
- `run-demo-dry`
- `run-demo-dry-resilient`
- `run-demo-execution`
- `run-demo-live-probe`

The important distinction is:

- `paper` and `demo-dry` use the operational engine without implying real-capital approval.
- `resilient` adds restore, persistence, reconciliation, and idempotency handling.
- `demo_execution` is a **guarded and narrow** demo-account phase.
- `demo_live_probe` performs a real open/close check on demo as an operational validation step.

### 3. Governance layer

The project already has a formal governance domain:

- strategy profile registry
- promotion and rollback under locking
- active profile materialization
- transactional evidence store
- endurance, lifecycle, and approved-demo gates
- readiness and control gates for serious demo operation

Observable lifecycle states include:

- `validated`
- `approved_demo`
- `active`
- `caution`
- `blocked`
- `deprecated`

## Current Architecture

The current code organization revolves around these layers:

- `src/iris_bot/main.py`
  CLI entrypoint
- `src/iris_bot/cli.py`
  command registration and dispatch
- `src/iris_bot/commands/`
  CLI adapters by domain
- `src/iris_bot/config_runtime.py` and `src/iris_bot/config_types.py`
  typed configuration loading from environment
- `src/iris_bot/processed_dataset.py`
  processed dataset construction and feature-space definition
- `src/iris_bot/quant_experiments.py`
  training, thresholding, variant comparison, and experiment reporting
- `src/iris_bot/backtest.py`
  economic backtest engine
- `src/iris_bot/paper.py`
  paper trading engine
- `src/iris_bot/resilient*.py`
  restore, reconciliation, persistence, and idempotency
- `src/iris_bot/governance*.py`
  validation, promotion, active profile resolution, and policy enforcement
- `src/iris_bot/demo_execution.py`
  preflight checks, runtime inference, and controlled demo execution

Existing technical documentation:

- `docs/ARCHITECTURE.md`
- `docs/architecture.md`
- `docs/adr/`

## Data, Features, and Model

### Current feature space

The real feature space no longer matches older README versions.

Recent runtime inference for `EURUSD` records **36 features**.
They include:

- returns and momentum
- rolling and Parkinson volatility
- ATR and regime percentiles
- candle structure
- distance-to-average features
- efficiency and autocorrelation
- normalized volume features
- ADX and regime flags
- Asia/London/New York session flags
- cross-symbol features:
  `cross_momentum_agreement`, `usd_strength_index`, `currency_strength_rank`

The real source of truth for feature ordering is:

- `src/iris_bot/processed_dataset.py`

### Model

What is implemented today:

- multiclass XGBoost
- classes `-1`, `0`, `1`
- ATR-based economic weighting
- configurable class weighting
- probability calibration
- grid-based threshold search
- global, contextual, and per-symbol model variants

## Current Operational State

### Governance and portfolio state

Current runtime artifacts:

- `data/runtime/strategy_profile_registry.json`
- `data/runtime/active_strategy_profiles.json`
- `data/runtime/demo_execution_registry.json`
- `data/runtime/evidence_store/`

Current reading of those artifacts:

- `AUDUSD`, `EURUSD`, and `GBPUSD` are materialized as `approved_demo`
- `USDJPY` is blocked by policy
- only `EURUSD` appears as `APPROVED_FOR_DEMO_EXECUTION`
- the demo execution `gate_open` flag is currently `true`

### Quality and validation

The official quality gate remains:

```bash
make check
```

Equivalent commands:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
.venv/bin/python -m pytest
.venv/bin/python -m iris_bot.main --help
```

In the audited state today, that gate is already recorded as passing in repository artifacts.

## MT5 And WSL Integration

The project explicitly supports an important operational case:

- if it runs under WSL and `MetaTrader5` is not importable, some commands are delegated to Windows
- that logic lives in `src/iris_bot/windows_mt5_bridge.py`
- the operational wrapper is `scripts/run_mt5_research_windows.sh`

This is not incidental; it is part of the actual design in use today.

## Repository Structure

```text
src/iris_bot/          main source code
src/iris_bot/commands/ CLI commands by domain
tests/                 test suite
config/                policy and business configuration
data/raw/              raw datasets
data/processed/        processed dataset, schema, and manifest
data/runtime/          runtime state, registry, evidence, demo models
runs/                  execution and audit artifacts
docs/                  architecture notes and ADRs
scripts/               operational wrappers
```

## Most Important Commands

### Bootstrap and checks

```bash
make bootstrap
make check
python -m iris_bot.main --help
```

### Data

```bash
python -m iris_bot.main fetch
python -m iris_bot.main validate-data
python -m iris_bot.main build-dataset
python -m iris_bot.main inspect-dataset
```

### Research

```bash
python -m iris_bot.main run-experiment
python -m iris_bot.main run-backtest
python -m iris_bot.main run-backtest --walk-forward
python -m iris_bot.main run-experiment-matrix
python -m iris_bot.main run-symbol-research
python -m iris_bot.main run-strategy-validation
```

### Operations and resilience

```bash
python -m iris_bot.main run-paper
python -m iris_bot.main run-paper-resilient
python -m iris_bot.main run-demo-dry
python -m iris_bot.main run-demo-dry-resilient
python -m iris_bot.main reconcile-state
python -m iris_bot.main restore-state-check
python -m iris_bot.main operational-status
```

### Governance

```bash
python -m iris_bot.main list-strategy-profiles
python -m iris_bot.main validate-strategy-profile
python -m iris_bot.main promote-strategy-profile
python -m iris_bot.main rollback-strategy-profile
python -m iris_bot.main active-strategy-status
python -m iris_bot.main demo-execution-readiness
```

### Controlled demo execution

```bash
python -m iris_bot.main validate-model-artifact
python -m iris_bot.main activate-demo-execution
python -m iris_bot.main demo-execution-preflight
python -m iris_bot.main run-demo-execution
python -m iris_bot.main demo-execution-status
python -m iris_bot.main run-demo-live-probe
```

For the full command inventory:

```bash
python -m iris_bot.main --help
```

## What This Project Is Not

To avoid incorrect expectations:

- it is not a plug-and-play bot for real money
- it is not a high-frequency trading system
- it is not a single simple strategy wrapped in one script
- it does not rely on backtests alone; it relies on operational evidence and gates
- the README is not the technical source of truth; the code and runtime artifacts are

## Useful References

- `docs/ARCHITECTURE.md`
- `docs/architecture.md`
- `config/governance_policy.json`
- `data/runtime/active_strategy_profiles.json`
- `data/runtime/demo_execution_registry.json`
- `runs/20260408T055455Z_demo_execution_readiness/demo_execution_readiness_report.json`
