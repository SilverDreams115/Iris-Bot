# IRIS-Bot

IRIS-Bot is a research and controlled-execution framework for FX trading with MetaTrader 5.  It covers the full pipeline from raw MT5 history to statistical validation, paper trading, demo dry-run, and a governance lifecycle that gates each strategy profile through evidence-backed promotions before it can ever touch real capital.

This README describes what actually exists in the code and validated artifacts today.  It makes no aspirational claims.

---

## What this project is

- A **research framework**: fetch MT5 history → build a processed dataset → train XGBoost → walk-forward economic backtest → permutation significance testing.
- A **controlled-execution harness**: paper trading, demo dry-run, and a single validated demo-live probe (open + close a real order on a demo account).
- A **governance engine**: strategy profiles move through `validated → approved_demo → active` via evidence gates, checksums, atomic writes, and registry locking.
- **Not** a general-purpose live-trading bot for real capital in its current state.

---

## ML pipeline

### Features — 32 inputs (M15, OHLCV)

All features are computed without look-ahead.  The processed dataset is built by `processed_dataset.py` and validated against a schema manifest on load.

**Returns and momentum** (cross-pair comparable, normalized fractional returns)
- `return_1`, `return_3`, `return_5` — bar-close fractional returns over 1/3/5 bars
- `log_return_1` — log return for the last bar
- `momentum_3` — `(close - close_3) / close_3` (normalized, not raw price diff)
- `momentum_5` — `(close - close_5) / close_5` (normalized)

**Volatility**
- `rolling_volatility_5`, `rolling_volatility_10` — rolling std of returns
- `atr_5`, `atr_10` — Average True Range over 5 / 10 bars
- `parkinson_volatility_10` — Parkinson estimator using high/low (more efficient than close-to-close for intrabar volatility)

**Candle structure**
- `range_ratio` — `(high - low) / close`
- `body_ratio` — `|open - close| / (high - low + ε)`
- `upper_wick_ratio`, `lower_wick_ratio` — wick fractions of total range

**Trend and mean-reversion**
- `distance_to_sma_5`, `distance_to_sma_10` — normalized distance from close to SMA
- `efficiency_ratio_10` — Kaufman ER over 10 bars (net move / path length)
- `efficiency_ratio_50` — Kaufman ER over up to 50 bars (regime-scale trend strength)
- `return_autocorr_10`, `return_autocorr_3`, `return_autocorr_5` — Pearson autocorrelation of returns at lag 1, 3, 5 (positive = momentum, negative = mean-reversion)
- `variance_ratio_hurst_proxy` — VR(5) over up to 50 bars; VR > 1 trending, VR < 1 mean-reverting (Hurst proxy)

**Volume**
- `volume_zscore_20` — volume z-score vs. 20-bar rolling mean/std (wider context than 5-bar; detects institutional flow)
- `volume_percentile_20` — volume percentile within last 20 bars

**Regime**
- `atr_regime_percentile` — ATR₁₀ percentile within last 50 bars (0 = compressed, 1 = expansion)

**Session**
- `session_asia`, `session_london`, `session_new_york` — binary flags (M15 UTC hour bands)

**Cross-symbol / CurrencyStrengthMatrix** (new)
- `cross_momentum_agreement` — fraction of correlated pairs moving in the expected direction (0.5 = neutral/single-symbol)
- `usd_strength_index` — DXY proxy computed from EURUSD / GBPUSD / AUDUSD / USDJPY returns with directional weights
- `currency_strength_rank` — relative rank of this symbol's currency vs. cross-pair universe (0.5 = neutral/single-symbol)

When only one symbol is present (e.g., in tests), cross-symbol features default to neutral values (`0.0`, `0.0`, `0.5`).

### Model

- **Algorithm**: XGBoost native (no sklearn wrapper), multiclass softprob
- **Labels**: -1 (Short) / 0 (Neutral) / 1 (Long)
- **Labeling strategies**: triple barrier (configurable barrier width, hold period, `allow_no_trade`) and next-bar-direction
- **Class weights**: frequency-inverse weighting with configurable cap, combined multiplicatively with economic sample weights
- **Economic sample weights**: per-bar weights proportional to `atr_5` relative to the training-set median ATR (capped at 3×). Bars with larger potential moves are weighted more — the model prioritizes getting high-stakes bars right.
- **Probability calibration** (auto-selects best on validation log-loss):
  - `uncalibrated` — raw softmax probabilities
  - `global_temperature` — single temperature scale applied to all classes
  - `classwise_temperature` — per-class temperature calibration
  - `auto` (default) — evaluates all three, applies the winner (including `uncalibrated` if it genuinely wins)
- **Feature importance**: gain-based, reported in experiment artifacts

### Threshold selection

- Grid search over configurable probability thresholds; optional fine-grained refinement around the coarse best
- Objective metrics: `macro_f1`, `balanced_accuracy`, `accuracy`, `directional_precision`
- **`directional_precision`**: fraction of non-neutral predictions that are correct — the economically meaningful metric because wrong directional calls lose money, missed opportunities (neutral) do not

### Baseline

`WeightedMomentumBaseline` — weighted combination of `return_1` (50%), `momentum_3` (30%), `momentum_5` (20%).  All inputs are normalized fractional returns, making the score cross-pair comparable and a more meaningful directional signal than a single raw momentum value.

### Statistical significance

- Permutation testing over walk-forward windows (labels shuffled, full economic replay repeated per trial)
- Deflated Sharpe Ratio (accounts for multiple-testing bias)
- Reported in `experiment_report.json`: p-value, null distribution percentile, per-trial results CSV

---

## Pipeline overview

```
MT5 history
    └── fetch / fetch-historical
          └── validate-data
                └── build-dataset  (processed_dataset.py, schema + manifest)
                      └── run-experiment  (train XGBoost, calibrate, threshold, significance)
                            └── run-backtest [--walk-forward]  (economic P&L replay)
                                  └── run-paper / run-demo-dry  (paper or dry-run on MT5)
                                        └── governance lifecycle
                                              (validated → approved_demo → active)
```

---

## Governance lifecycle

Each strategy profile (symbol + hyperparameters + thresholds) follows a state machine enforced by the registry:

| State | Meaning |
|---|---|
| `validated` | Profile registered; backtest evidence acceptable |
| `approved_demo` | Passed endurance + lifecycle gates; demo audit checked |
| `active` | In-use by the paper/demo engine for this symbol |
| `caution` | Borderline gates; still in use but flagged |
| `blocked` | Failed gates; removed from active |
| `deprecated` | Superseded by a newer profile or manual rollback |

**Gate matrix** (promotion review):
- Endurance gates: `min_sharpe`, `max_drawdown`, `min_win_rate`, `min_trade_count`
- Lifecycle gates: `min_age_hours`, `sufficient_evidence`
- Demo audit gate: broker fills vs. internal fills reconciliation; P&L and slippage tolerance check

**Registry integrity**: atomic JSON writes, file-level exclusive lock with stale-lock detection, ETag-based mutation conflict detection, SHA-256 checksums per profile payload.

---

## Demo execution validation

`run-demo-live-probe` performs a real, isolated validation:
1. Confirms the connected account is a demo account
2. Opens a minimal real order on the first configured symbol
3. Locates the open position
4. Closes it
5. Writes an auditable report to `runs/*_demo_live_probe/demo_live_probe_report.json`

This probe was validated on a `MetaQuotes-Demo` account (EURUSD, no residual positions).

**This is not** the normal paper/dry-run pipeline, and it does not make the system a general-purpose live-trading bot.

---

## Commands reference

### Data

```bash
python -m iris_bot.main fetch
python -m iris_bot.main fetch-historical
python -m iris_bot.main validate-data
python -m iris_bot.main build-dataset
python -m iris_bot.main inspect-dataset
```

### Research and training

```bash
python -m iris_bot.main run-experiment
python -m iris_bot.main run-backtest
python -m iris_bot.main run-backtest --walk-forward
python -m iris_bot.main build-symbol-profiles
python -m iris_bot.main run-symbol-research
python -m iris_bot.main run-strategy-validation
python -m iris_bot.main audit-strategy-block-causes
python -m iris_bot.main compare-symbol-models
python -m iris_bot.main evaluate-dynamic-exits
python -m iris_bot.main symbol-go-no-go
```

### Paper trading and demo dry-run

```bash
python -m iris_bot.main run-paper
python -m iris_bot.main run-paper-resilient
python -m iris_bot.main run-demo-dry
python -m iris_bot.main run-demo-dry-resilient
python -m iris_bot.main mt5-check
python -m iris_bot.main operational-status
python -m iris_bot.main reconcile-state
python -m iris_bot.main restore-state-check
```

### Governance

```bash
python -m iris_bot.main list-strategy-profiles
python -m iris_bot.main validate-strategy-profile
python -m iris_bot.main review-approved-demo-readiness
python -m iris_bot.main promote-strategy-profile
python -m iris_bot.main rollback-strategy-profile
python -m iris_bot.main active-strategy-status
python -m iris_bot.main diagnose-profile-activation
python -m iris_bot.main audit-governance-consistency
python -m iris_bot.main symbol-reactivation-readiness
python -m iris_bot.main reconcile-lifecycle
python -m iris_bot.main lifecycle-audit-report
python -m iris_bot.main audit-governance-locking
python -m iris_bot.main materialize-active-profiles
python -m iris_bot.main repair-strategy-profile-registry
python -m iris_bot.main evidence-store-status
python -m iris_bot.main approved-demo-gate-audit
python -m iris_bot.main active-portfolio-status
python -m iris_bot.main demo-execution-readiness
```

### Soak, endurance, chaos

```bash
python -m iris_bot.main run-paper-soak
python -m iris_bot.main run-demo-dry-soak
python -m iris_bot.main run-symbol-endurance
python -m iris_bot.main run-enabled-symbols-soak
python -m iris_bot.main symbol-stability-report
python -m iris_bot.main audit-endurance-reporting
python -m iris_bot.main run-chaos-scenario
python -m iris_bot.main go-no-go-report
```

### Demo live probe

```bash
python -m iris_bot.main run-demo-live-checklist
python -m iris_bot.main run-demo-live-probe
```

---

## Environment

### Setup

```bash
make bootstrap          # creates .venv and installs all dependencies
make test               # runs the full test suite
```

Manual:

```bash
./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/python -m pytest
./.venv/bin/python -m iris_bot.main --help
```

### MetaTrader 5

- The `MetaTrader5` package is a Windows-only dependency.
- Research, backtesting, governance, and most tests run on Linux/WSL without MT5.
- Live MT5 connectivity and the demo-live probe require Python on Windows.
- The project auto-loads a `.env` file if present.

---

## Key modules

| Area | Modules |
|---|---|
| Data & features | `data.py`, `processed_dataset.py`, `labels.py`, `preprocessing.py` |
| ML | `xgb_model.py`, `baselines.py`, `thresholds.py`, `metrics.py` |
| Experiment & backtest | `experiments.py`, `wf_backtest.py`, `backtest.py`, `significance.py` |
| Paper & demo | `paper.py`, `resilient.py`, `operational.py`, `mt5.py` |
| Governance | `governance.py`, `governance_promotion.py`, `governance_active.py`, `governance_validation.py` |
| Registry | `profile_registry.py`, `registry_lock.py`, `profile_evidence.py`, `lifecycle.py` |
| Demo live | `demo_readiness.py`, `demo_live_checklist.py`, `demo_live_probe.py` |

---

## Known limitations

- No documented or validated live-trading flow for real capital.
- The demo-live probe is isolated from the normal paper/dry-run pipeline.
- Full MT5 connectivity requires Windows Python (`MetaTrader5` library).
- `demo_execution_readiness` remains conservative by design and may return `caution` even when the separate demo-live checklist passes.
- Statistical edge and profitability are not guaranteed by the infrastructure.

---

## What this README does not claim

- Future profitability or real alpha
- Readiness for real-capital deployment
- Uniform validation across Windows and Linux for all execution paths
- An integrated economic-calendar gate in the main pipeline
- Meta-labeling in production
- Trailing stop as an integrated operational policy
