# ADR-006: Walk-Forward Profit Factor Gate Metric

**Status:** Accepted  
**Date:** 2026-04-06  
**Deciders:** Principal Engineer (quantitative governance review)

---

## Context

The `compare_symbol_variants` pipeline in `symbol_focused_rework.py` gates demo-execution approval using a walk-forward profit factor floor of 1.10 (`ApprovedDemoGateConfig.min_profit_factor`).

Prior to this ADR, the gate metric was the **arithmetic mean of per-fold profit factors** (`mean_profit_factor`). A cross-symbol audit conducted 2026-04-06 identified two structural defects:

### Defect 1: Zero-trade folds drag the mean down

When a fold produces zero trades (model probability never exceeds the threshold), the backtest engine returns `profit_factor = 0.0`. Including these zero-trade folds in the arithmetic mean artificially deflates it.

**Observed case — EURUSD V1_baseline:**
- 3 zero-trade folds (indices 0, 6, 13) → PF = 0.0
- Trading folds (non-zero-trade) had weighted PF = 1.1415 (above floor)
- Arithmetic mean = 0.9769 → FAIL (below 1.10 floor)
- Real economic question: "Do the trades that were taken make money?" → Yes (1.14)
- Zero-trade folds are already captured by `max_no_trade_ratio` gate; double-penalising via PF=0 in the mean is redundant and misleading.

### Defect 2: PF=999 outlier folds inflate the mean to a pathological level

When a fold has winning trades but zero losing trades, the engine caps `profit_factor = 999.0`. Including these in the arithmetic mean makes the mean meaningless as a signal.

**Observed cases — GBPUSD V4 and V5:**
- Several folds with PF=999 (single win, zero losses)
- GBPUSD V4: arithmetic mean = 71.95 → FALSE PASS despite weighted PF = 0.9826 (money-losing)
- GBPUSD V5: arithmetic mean = 72.06 → FALSE PASS despite weighted PF = 0.8960 (money-losing)

The arithmetic mean produced both false negatives (EURUSD V1) and false positives (GBPUSD V4/V5).

---

## Decision

Replace the walk-forward gate metric with **trade-weighted profit factor**:

```
trade_weighted_profit_factor = sum(gross_profit_usd across all folds) /
                               sum(gross_loss_usd across all folds)
```

- If `total_gross_loss = 0` and `total_gross_profit > 0`: result = 999.0 (no-loss strategy)
- If both = 0: result = 0.0 (no trades executed anywhere)

### What stays unchanged

- Floor value: `min_profit_factor = 1.10` (unchanged)
- All other gate conditions: test PF, trade count, drawdown, no-trade ratio, WF expectancy, WF total PnL, positive fold ratio (unchanged)
- The `mean_profit_factor` (arithmetic mean) is retained in the aggregate output for audit/reporting but no longer gates approval

### Implementation

**`_evaluate_rows`** (symbol_focused_rework.py): now returns `gross_profit_usd` and `gross_loss_usd` per fold directly from `run_backtest_engine` metrics.

**`_walk_forward_variant`**: computes `trade_weighted_profit_factor` by summing gross_profit and gross_loss across all non-skipped folds.

**`_apply_variant_gates`**: gate check on line 619 now reads `agg["trade_weighted_profit_factor"]` instead of `agg["mean_profit_factor"]`.

---

## Correctness Verification (not a backdoor)

This change was validated against all symbols with known outcomes:

| Symbol | Variant | Arith mean PF | Weighted PF | Gate before | Gate after | Correct? |
|--------|---------|--------------|-------------|------------|-----------|----------|
| EURUSD | V1_baseline | 0.9769 | 1.1415 | FAIL | PASS | ✓ Yes — trades are genuinely profitable |
| GBPUSD | V4 | 71.95 | 0.9826 | FALSE PASS | FAIL | ✓ Yes — variant loses money |
| GBPUSD | V5 | 72.06 | 0.8960 | FALSE PASS | FAIL | ✓ Yes — variant loses money |
| GBPUSD | V3 | 1.14 | 1.28 | PASS | PASS | ✓ No regression (already blocked by other gates: trade count, no-trade ratio, WF positive ratio) |

The change corrects two real defects (false negative + false positive). It is not symbol-specific. EURUSD V1 was the concrete case that triggered the audit, but the rule change is general.

---

## Rejected Alternatives

**Keep arithmetic mean:** Produces documented false positives (GBPUSD V4/V5) and false negatives (EURUSD V1). Not defensible.

**Exclude zero-trade folds from arithmetic mean ("trading-only mean"):** Fixes Defect 1 but not Defect 2. GBPUSD V4 still gets PF=999 folds inflating the mean. Rejected.

**Relax the floor:** Not permitted under project governance rules. Floor stays at 1.10.

---

## Consequences

- EURUSD V1_baseline now passes the walk-forward PF gate (1.1415 ≥ 1.10).
- GBPUSD V4 and V5 now correctly fail the gate (0.98, 0.90 < 1.10).
- All existing tests updated to include `trade_weighted_profit_factor` in mock WF aggregate dicts.
- No changes to any gate floor values.
- No changes to other gate conditions.
