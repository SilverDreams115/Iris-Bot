# IRIS-Bot

Base de trabajo para un bot de trading forex con machine learning, backtesting, paper trading operativo y validación MT5 demo.

## Estado actual (Fase Correctiva Premium pre-demo seria)

- Dataset procesado, labels, splits temporales y walk-forward base: ✅
- Modelo principal XGBoost nativo (sin sklearn): ✅
- Backtest económico con replay temporal, costos y sizing: ✅
- **Política intrabar configurable** (SL/TP misma vela): ✅
- **PnL y sizing generalizados para cruces no USD**: ✅
- **Replay económico walk-forward fold por fold**: ✅
- **Validaciones de consistencia interna del engine**: ✅
- **Paper trading separado del backtest**: ✅ — Fase 4
- **Estado operativo explícito y serializable**: ✅ — Fase 4
- **MT5 demo adapter con validación fuerte y dry-run**: ✅ — Fase 4
- **Journaling operativo reproducible**: ✅ — Fase 4
- **Reconciliación broker ↔ estado interno**: ✅ — Fase 5
- **Persistencia y restore operativo**: ✅ — Fase 5
- **Reconnect y recovery controlados**: ✅ — Fase 5
- **Alertas y estado operativo explícito**: ✅ — Fase 5
- **Control de sesiones y loop resiliente**: ✅ — Fase 5
- **Soak testing multi-cycle**: ✅ — Fase 5.5
- **Chaos testing controlado**: ✅ — Fase 5.5
- **Go / Caution / No-Go explícito**: ✅ — Fase 5.5
- **Research y profiling por símbolo desde histórico MT5**: ✅ — Fase 6
- **Framework de estrategias por símbolo**: ✅ — Fase 6
- **SL/TP dinámicos ATR auditables**: ✅ — Fase 6
- **Comparación ML global vs por símbolo**: ✅ — Fase 6
- **Thresholds y gates `enabled/caution/disabled` por símbolo**: ✅ — Fase 6
- **Corrección de leakage en validación por símbolo**: ✅ — Fase Correctiva Premium
- **Reconciliación MT5 filtrada por ownership/scope del bot**: ✅ — Fase Correctiva Premium
- **Idempotencia real por event id con fallback explícito**: ✅ — Fase Correctiva Premium
- **Taxonomía de sesiones unificada end-to-end**: ✅ — Fase Correctiva Premium
- **CLI modular por dominios**: ✅ — Fase Correctiva Premium
- **Versionado y checks de compatibilidad de artefactos**: ✅ — Fase Correctiva Premium
- Live trading: ❌ — Explícitamente fuera de alcance

> **AVISO**: Las métricas de clasificación no prueban rentabilidad. El backtest es
> econónicamente serio pero no sustituye validación demo ni ejecución real.
> **Esta fase sigue sin habilitar live real.** Paper trading y demo dry-run siguen
> siendo capas de validación previas. La fase correctiva premium endurece la
> confiabilidad cuantitativa y operativa: elimina leakage en selección por símbolo,
> acota la reconciliación MT5 al ownership real del bot, hace explícita la
> idempotencia por evento y reduce acoplamiento del CLI. No convierte al sistema en
> "production ready" ni sustituye validación demo prolongada.

---

## Política intrabar (Fase 3.5)

Cuando en una misma vela el precio toca tanto el Stop Loss como el Take Profit,
el engine aplica la política configurada:

| Política | Comportamiento | exit_reason |
|---|---|---|
| `conservative` (default) | Gana el Stop Loss — peor resultado para el trader | `stop_loss_same_bar` |
| `optimistic` | Gana el Take Profit — mejor resultado para el trader | `take_profit_same_bar` |

El campo `is_intrabar_ambiguous = True` en el trade log identifica estos eventos.

**Configurar via env var:**
```bash
export IRIS_BACKTEST_INTRABAR_POLICY=conservative   # o optimistic
```

**Configurar via CLI:**
```bash
python3 -m iris_bot.main run-backtest --intrabar-policy optimistic
```

---

## Modelo económico de PnL (Fase 3.5)

### Conversión quote→USD

| Tipo de par | Ejemplo | Fórmula | Necesita aux_rate |
|---|---|---|---|
| XXXUSD | EURUSD, GBPUSD | rate = 1.0 | No |
| USDXXX | USDJPY, USDCAD | rate = 1/price | No |
| Cross | EURGBP, EURJPY, GBPJPY | rate = aux_rates[quote] | **Sí** |

Para cruces no USD, el engine devuelve `volume_lots=0.0` (trade bloqueado) si no se
provee `aux_rates`. Esto falla de forma explícita y trazable — no silenciosamente con
matemática incorrecta.

### Fórmula PnL

```
price_delta     = (exit_price - entry_price) × direction
quote_pnl       = price_delta × contract_size × volume_lots
pnl_account     = quote_pnl × quote_to_account_rate(exit_price)
net_pnl_usd     = pnl_account_entry_cost - pnl_account_exit_cost - commission_entry - commission_exit
```

### Límites del modelo económico actual

- Comisión fija por lote, no por símbolo.
- Spread y slippage uniformes para todos los símbolos.
- `aux_rates` son estáticos (no se actualizan con el mercado).
- Sin modelado de swap/rollover overnight.
- Sin costos de financiamiento.

---

## Fase 4

### Qué agrega

- `run-paper`: engine operativo separado del backtest, con estado vivo, eventos y journals
- `mt5-check`: validación fuerte de entorno MT5 y de símbolos configurados
- `run-demo-dry`: flujo demo que valida requests MT5 y registra qué se habría enviado, sin enviar órdenes reales

### Estado operativo explícito

El engine paper mantiene y serializa:

- `account_state`
- `open_positions`
- `closed_positions`
- `daily_loss_tracker`
- `cooldown_tracker`
- `exposure`
- `last_signal_per_symbol`
- `current_session_status`
- `blocked_trades_summary`

### Eventos operativos mínimos

- `signal_generated`
- `signal_rejected`
- `order_simulated`
- `position_opened`
- `position_closed`
- `stop_loss_hit`
- `take_profit_hit`
- `cooldown_active`
- `max_daily_loss_block`
- `symbol_blocked`
- `volume_rejected`

### Estado actual de SL/TP dinámicos

La arquitectura ya soporta:

```bash
IRIS_STOP_POLICY=static
IRIS_TARGET_POLICY=static
IRIS_STOP_POLICY=atr_dynamic
IRIS_TARGET_POLICY=atr_dynamic
```

La implementación actual de `atr_dynamic`:

- usa ATR reciente y volatilidad reciente
- aplica multiplicadores por símbolo
- respeta límites mínimos y máximos
- deja trazabilidad explícita por trade en logs y journals

Todavía siguen fuera de alcance:

- trailing stop
- break-even
- volatility-adjusted stop
- take profit adaptativo

Esas políticas siguen pendientes para una fase posterior.

## Fase 5

### Qué endurece

- reconciliación explícita entre estado local y broker
- restore seguro tras reinicio
- reconnect controlado con estado operativo
- alertas accionables
- persistencia atómica del estado runtime
- control de sesiones y bloqueo fuera de horario
- loop resiliente con prevención de procesamiento duplicado

### Políticas de reconciliación

Configurable por:

```bash
IRIS_RECONCILIATION_POLICY=hard_fail
```

Valores soportados:

- `log_only`
- `soft_resync`
- `block`
- `hard_fail`

Default conservador: `hard_fail`.
Si hay discrepancia crítica, el sistema no sigue operando como si nada.

### Persistencia runtime

El sistema guarda un snapshot operativo runtime en:

```text
data/runtime/runtime_state.json
```

Contiene:

- account snapshot
- open positions
- pending intents
- cooldown state
- daily loss tracker
- blocked reasons
- processing state e idempotencia
- latest broker snapshot

Si el restore no es confiable y `IRIS_REQUIRE_STATE_RESTORE_CLEAN=true`,
el sistema se bloquea.

## Fase 5.5

### Qué añade

- soak runner multi-cycle para paper y demo-dry
- chaos scenarios controlados y reproducibles
- health checks por ciclo
- clasificación formal `go`, `caution`, `no_go`
- agregación de artefactos por ciclo y globales

### Comandos

```bash
PYTHONPATH=src python3 -m iris_bot.main run-paper-soak
PYTHONPATH=src python3 -m iris_bot.main run-demo-dry-soak
PYTHONPATH=src python3 -m iris_bot.main run-chaos-scenario
PYTHONPATH=src python3 -m iris_bot.main go-no-go-report
```

### Chaos scenarios soportados

- `disconnect_once`
- `reconnect_fail_once`
- `corrupt_restore_once`
- `duplicate_event_once`
- `broker_mismatch_once`
- `repeated_rejections`
- `communication_error_once`
- `market_session_blocked`

### Lectura de go / caution / no_go

- `go`: ciclos estables, sin restore corrupto ni discrepancias críticas
- `caution`: warnings recuperables o ruido operativo relevante
- `no_go`: restore fallido, reconnect no recuperado, mismatch crítico, validación fallida o alerta crítica

## Fase 6

### Qué añade

- profiling por símbolo y timeframe desde histórico MT5 ya descargado
- framework común de estrategias por símbolo con overrides auditables
- `stop_policy=atr_dynamic` y `target_policy=atr_dynamic`
- comparación de modelo global vs modelo por símbolo
- pruning conservador de features dominantes/inestables
- selección de threshold por símbolo basada en métrica económica
- gates explícitos `enabled`, `caution`, `disabled`

## Fase Correctiva Premium

### Qué corrige de forma prioritaria

- elimina leakage entre selección y test final en validación por símbolo
- corrige reconciliación MT5 para no bloquear por posiciones ajenas al scope del bot
- corrige la semántica interna de conexión MT5 cuando `initialize()` pasa y `login()` falla
- implementa uso real de `processed_event_ids` y deja explícito cuándo hay fallback por timestamp
- unifica la taxonomía de sesiones en dataset, research, reporting y gates
- modulariza la CLI por dominios
- añade versionado básico y checks de compatibilidad para artefactos críticos

### Validación por símbolo sin leakage

La selección ahora separa explícitamente:

- `fit_train`
- `fit_validation`
- `selection`
- `final_test`

Las decisiones de:

- `global_model` vs `symbol_model`
- threshold por símbolo
- `stop_policy` / `target_policy` preferida
- gates `enabled/caution/disabled`

se toman usando el tramo de `selection`, no el `final_test`.

El `final_test` queda reservado solo para evaluación final y reporting.

Artefactos relevantes:

- `leakage_fix_report.json`
- `strategy_validation_report.json`

### Reconciliación MT5 filtrada por ownership

La reconciliación ya no trata como discrepancia crítica cualquier posición de la cuenta.
El snapshot del broker filtra por ownership del bot usando:

- `magic_number`
- `comment_tag`
- universo de símbolos controlado cuando aplica

Las posiciones ajenas quedan fuera del scope y se registran en:

- `reconciliation_scope_report.json`

Si la discrepancia sí corresponde al scope del bot, el comportamiento sigue siendo conservador.

### Idempotencia real

El sistema usa:

- `processed_event_ids` si existe un `event_id` real
- fallback explícito por `symbol + timestamp` si no existe

Ese comportamiento queda trazado en:

- `idempotency_report.json`

Nunca se reporta idempotencia por `event_id` si realmente se usó fallback.

### CLI modular

La superficie de comandos quedó separada por dominios:

- `iris_bot.commands.data`
- `iris_bot.commands.research`
- `iris_bot.commands.backtest`
- `iris_bot.commands.operations`
- `iris_bot.commands.soak`
- `iris_bot.commands.audit`

La entrada sigue siendo:

```bash
PYTHONPATH=src python3 -m iris_bot.main <comando>
```

### Auditoría correctiva

```bash
PYTHONPATH=src python3 -m iris_bot.main run-corrective-audit
```

Artefactos:

- `corrective_audit_report.json`
- `leakage_fix_report.json`
- `reconciliation_scope_report.json`
- `idempotency_report.json`
- `session_consistency_report.json`
- `performance_benchmark_report.json`
- `artifact_schema_report.json`

### Qué sigue faltando para un estándar más premium

- validación demo prolongada por símbolo habilitado
- reconciliación más profunda por deals/tickets reales del broker
- observabilidad externa fuera del proceso
- orchestration multi-symbol más estricta
- gates previos a live todavía más duros

Live real sigue explícitamente deshabilitado.

### Filosofía operativa

- si un símbolo no demuestra edge suficiente, queda `disabled`
- `caution` no es aprobación automática; exige revisión humana
- los exits dinámicos implementados son simples y auditables, no “inteligentes”
- live real sigue explícitamente deshabilitado

## Comandos

### Descargar histórico desde MT5
```bash
export IRIS_MT5_ENABLED=true
export IRIS_MT5_LOGIN=123456
export IRIS_MT5_PASSWORD='tu_password'
export IRIS_MT5_SERVER='TuBroker-Demo'
PYTHONPATH=src python3 -m iris_bot.main fetch
PYTHONPATH=src python3 -m iris_bot.main fetch-historical
```

### Validar integridad del dataset
```bash
PYTHONPATH=src python3 -m iris_bot.main validate-data
```

### Construir dataset procesado
```bash
PYTHONPATH=src python3 -m iris_bot.main build-dataset
```

### Ejecutar experimento ML
```bash
PYTHONPATH=src python3 -m iris_bot.main run-experiment
```

### Backtest económico (test split)
```bash
PYTHONPATH=src python3 -m iris_bot.main run-backtest
# Con política explícita:
PYTHONPATH=src python3 -m iris_bot.main run-backtest --intrabar-policy conservative
PYTHONPATH=src python3 -m iris_bot.main run-backtest --intrabar-policy optimistic
```

### Backtest económico walk-forward fold por fold
```bash
PYTHONPATH=src python3 -m iris_bot.main run-backtest --walk-forward
# Con política explícita:
PYTHONPATH=src python3 -m iris_bot.main run-backtest --walk-forward --intrabar-policy optimistic
```

### Paper trading operativo
```bash
PYTHONPATH=src python3 -m iris_bot.main run-paper
```

Requisitos mínimos:

- dataset procesado presente
- `experiment_report.json` disponible en el último run de experimento o en `IRIS_BACKTEST_EXPERIMENT_RUN_DIR`
- `models/xgboost_model.json` presente

### Validación de entorno MT5 demo
```bash
export IRIS_MT5_ENABLED=true
export IRIS_MT5_LOGIN=123456
export IRIS_MT5_PASSWORD='tu_password'
export IRIS_MT5_SERVER='TuBroker-Demo'
PYTHONPATH=src python3 -m iris_bot.main mt5-check
```

### Demo dry-run MT5
```bash
export IRIS_MT5_ENABLED=true
export IRIS_MT5_LOGIN=123456
export IRIS_MT5_PASSWORD='tu_password'
export IRIS_MT5_SERVER='TuBroker-Demo'
PYTHONPATH=src python3 -m iris_bot.main run-demo-dry
```

`run-demo-dry`:

- usa el engine paper, no el backtest
- valida símbolo, visibilidad, trading permitido, min/max/step, filling mode y request
- registra la request simulada
- **no envía órdenes reales**

### Reconciliar estado broker/local
```bash
PYTHONPATH=src python3 -m iris_bot.main reconcile-state
```

### Verificar restore de estado
```bash
PYTHONPATH=src python3 -m iris_bot.main restore-state-check
```

### Paper resilient
```bash
PYTHONPATH=src python3 -m iris_bot.main run-paper-resilient
```

### Demo dry-run resilient
```bash
PYTHONPATH=src python3 -m iris_bot.main run-demo-dry-resilient
```

### Estado operativo actual
```bash
PYTHONPATH=src python3 -m iris_bot.main operational-status
```

### Soak paper
```bash
export IRIS_SOAK_CYCLES=5
PYTHONPATH=src python3 -m iris_bot.main run-paper-soak
```

### Soak demo dry con restore entre ciclos
```bash
export IRIS_SOAK_CYCLES=5
export IRIS_SOAK_RESTORE_BETWEEN_CYCLES=true
PYTHONPATH=src python3 -m iris_bot.main run-demo-dry-soak
```

### Chaos scenario reproducible
```bash
export IRIS_CHAOS_ENABLED=true
export IRIS_CHAOS_SCENARIOS=disconnect_once,repeated_rejections
PYTHONPATH=src python3 -m iris_bot.main run-chaos-scenario
```

### Research por símbolo desde histórico MT5
```bash
PYTHONPATH=src python3 -m iris_bot.main run-symbol-research
```

### Construir snapshot de strategy profiles
```bash
PYTHONPATH=src python3 -m iris_bot.main build-symbol-profiles
```

### Validación completa por símbolo
```bash
export IRIS_STOP_POLICY=atr_dynamic
export IRIS_TARGET_POLICY=atr_dynamic
PYTHONPATH=src python3 -m iris_bot.main run-strategy-validation
```

### Comparar modelo global vs por símbolo
```bash
PYTHONPATH=src python3 -m iris_bot.main compare-symbol-models
```

### Evaluar exits dinámicos
```bash
export IRIS_STOP_POLICY=atr_dynamic
export IRIS_TARGET_POLICY=atr_dynamic
PYTHONPATH=src python3 -m iris_bot.main evaluate-dynamic-exits
```

### Decisión final por símbolo
```bash
PYTHONPATH=src python3 -m iris_bot.main symbol-go-no-go
```

### Ejecutar tests
```bash
PYTHONPATH=src pytest tests/
# Bloque operativo principal de Fase 4:
PYTHONPATH=src pytest tests/test_paper_engine.py tests/test_mt5.py tests/test_engine_constraints.py -v
# Bloque resiliente de Fase 5:
PYTHONPATH=src pytest tests/test_resilient_operations.py -v
# Bloque endurance / chaos de Fase 5.5:
PYTHONPATH=src pytest tests/test_soak_operations.py -v
# Bloque research / symbol strategies / dynamic exits:
PYTHONPATH=src pytest tests/test_symbol_strategy_phase.py -v
```

---

## Artefactos generados

### Backtest (`runs/<ts>_backtest/`)
- `trade_log.csv` — un trade por fila, incluye `is_intrabar_ambiguous`
- `equity_curve.csv` — balance y equity por barra
- `backtest_report.json` — métricas, config, consistency check, policy usada

### Walk-forward económico (`runs/<ts>_wf_backtest/`)
- `wf_backtest_summary.json` — resumen global y por fold
- `fold_NN/trade_log.csv` — trades del fold
- `fold_NN/equity_curve.csv` — equity del fold
- `fold_NN/fold_report.json` — métricas, threshold, consistency por fold

### Paper trading (`runs/<ts>_paper/`)
- `config_used.json` — config exacta usada
- `signal_log.csv` — señales generadas y rechazadas
- `execution_journal.csv` — eventos operativos y órdenes simuladas
- `open_positions_snapshot.json` — snapshot serializable del estado final
- `closed_trades.csv` — cierres simulados
- `daily_summary.json` — resumen diario reproducible
- `run_report.json` — estado final, exposición y bloqueos
- `validation_report.json` — validaciones y checks operativos

### Demo dry-run (`runs/<ts>_demo_dry/`)
- mismos artefactos del paper engine
- `execution_journal.csv` incluye el payload validado del request MT5 simulado
- no se ejecutan órdenes reales

### Resilient runs (`runs/<ts>_paper_resilient/`, `runs/<ts>_demo_dry_resilient/`)
- `config_used.json`
- `reconciliation_report.json`
- `restore_state_report.json`
- `operational_status.json`
- `alerts_log.jsonl`
- `execution_journal.csv`
- `signal_log.csv`
- `closed_trades.csv`
- `open_positions_snapshot.json`
- `run_report.json`
- `validation_report.json`

### Soak / chaos (`runs/<ts>_paper_soak/`, `runs/<ts>_demo_dry_soak/`)
- `soak_report.json`
- `health_report.json`
- `go_no_go_report.json`
- `chaos_scenarios_applied.json`
- `cycle_summaries/`
- `alerts_log.jsonl`
- `incident_log.jsonl`
- `reconciliation_report.json`
- `restore_state_report.json`
- `operational_status.json`
- `execution_journal.csv`
- `signal_log.csv`
- `closed_trades.csv`

### Symbol research / validation (`runs/<ts>_symbol_research/`, `runs/<ts>_strategy_validation/`)
- `symbol_profile.json` por símbolo/timeframe
- `symbol_research_report.json`
- `strategy_validation_report.json`
- `dynamic_exit_report.json`
- `symbol_enablement_report.json`
- `model_comparison_report.json`
- `threshold_report.json`
- `aggregated_portfolio_report.json`
- `data/runtime/strategy_profiles.json`

---

## Estructura
```text
src/iris_bot/
  backtest.py          Engine principal + helpers de I/O
  consistency.py       Validaciones de consistencia interna
  wf_backtest.py       Replay económico walk-forward fold por fold
  paper.py             Engine operativo paper/demo separado del backtest
  operational.py       Estado serializable y writers de journaling
  resilient.py         Reconciliación, restore, reconnect, alertas, sesiones
  soak.py              Soak runner, chaos controlado y go/no-go
  exits.py             Policies `static` y `atr_dynamic` compartidas
  symbols.py           Framework de perfiles y overrides por símbolo
  symbol_research.py   Profiling y research por símbolo/timeframe
  symbol_validation.py Comparación global vs símbolo, thresholds y gates
  risk.py              PnL, sizing, conversión de monedas
  mt5.py               Histórico MT5 + validación demo dry-run
  config.py            Configuración por env vars
  experiments.py       Pipeline ML (clasificación)
  walk_forward.py      Generación de ventanas walk-forward
  ...

tests/
  test_intrabar_policy.py    SL/TP misma vela
  test_pnl_forex.py          PnL para EURUSD/USDJPY/crosses
  test_consistency.py        Validaciones de consistencia
  test_engine_constraints.py max_daily_loss, cooldown, etc.
  test_paper_engine.py       Estado paper, journals y demo dry-run
  test_mt5.py                Validaciones MT5 demo y request formation
  test_resilient_operations.py Reconciliación, restore, reconnect, alertas
  test_soak_operations.py    Soak multi-cycle, chaos y go/no-go
  test_wf_backtest.py        Walk-forward económico fold por fold
  ...
```

## Variables de entorno relevantes (Fase 6)

```bash
IRIS_BACKTEST_INTRABAR_POLICY=conservative   # o optimistic
IRIS_BACKTEST_STARTING_BALANCE_USD=1000
IRIS_BACKTEST_SPREAD_PIPS=1.2
IRIS_BACKTEST_SLIPPAGE_PIPS=0.2
IRIS_BACKTEST_COMMISSION_PER_LOT_PER_SIDE_USD=3.5
IRIS_BACKTEST_USE_ATR_STOPS=true
IRIS_BACKTEST_FIXED_STOP_LOSS_PCT=0.0020
IRIS_BACKTEST_FIXED_TAKE_PROFIT_PCT=0.0030
IRIS_BACKTEST_MAX_HOLDING_BARS=8
IRIS_WF_TRAIN_WINDOW=240
IRIS_WF_VALIDATION_WINDOW=80
IRIS_WF_TEST_WINDOW=80
IRIS_WF_STEP=80
IRIS_STOP_POLICY=static          # o atr_dynamic
IRIS_TARGET_POLICY=static        # o atr_dynamic
IRIS_DYNAMIC_EXIT_VOL_ADJUST_SCALE=8.0
IRIS_DYNAMIC_EXIT_MIN_STOP_PCT=0.0010
IRIS_DYNAMIC_EXIT_MAX_STOP_PCT=0.0100
IRIS_DYNAMIC_EXIT_MIN_TAKE_PROFIT_PCT=0.0015
IRIS_DYNAMIC_EXIT_MAX_TAKE_PROFIT_PCT=0.0200
IRIS_STRATEGY_PROFILES_FILENAME=strategy_profiles.json
IRIS_STRATEGY_MIN_SYMBOL_ROWS=180
IRIS_STRATEGY_MIN_VALIDATION_TRADES=5
IRIS_STRATEGY_MIN_EXPECTANCY_USD=0.0
IRIS_STRATEGY_MAX_DRAWDOWN_USD=125.0
IRIS_STRATEGY_MIN_PROFIT_FACTOR=1.05
IRIS_STRATEGY_MIN_POSITIVE_WF_RATIO=0.50
IRIS_RECONCILIATION_POLICY=hard_fail
IRIS_RECONNECT_RETRIES=3
IRIS_RECONNECT_BACKOFF_SECONDS=0
IRIS_REQUIRE_STATE_RESTORE_CLEAN=true
IRIS_SESSION_ENABLED=true
IRIS_ALLOWED_WEEKDAYS=0,1,2,3,4
IRIS_ALLOWED_START_HOUR_UTC=0
IRIS_ALLOWED_END_HOUR_UTC=23
IRIS_SOAK_CYCLES=3
IRIS_SOAK_PAUSE_SECONDS=0
IRIS_SOAK_RESTORE_BETWEEN_CYCLES=true
IRIS_CHAOS_ENABLED=false
IRIS_CHAOS_SCENARIOS=
IRIS_CHAOS_EVERY_N_CYCLES=0
```

---

## Supuestos operativos del engine (auditables)

1. **Entrada**: al open de la barra N+1 (una barra después de la señal N)
2. **Salida**: por `stop_loss`, `take_profit`, `time_exit` o `end_of_data`
3. **Intrabar**: política configurable (`conservative` por defecto)
4. **Costos**: aplicados una vez por lado — entry y exit son independientes
5. **Cooldown**: bloqueo de entradas para el símbolo afectado por N barras tras pérdida
6. **Max daily loss**: se computa por día calendario sobre PnL realizado
7. **One position per symbol**: no se abre segunda posición si ya hay una abierta
8. **Max open positions**: máximo de posiciones simultáneas entre todos los símbolos

---

## Qué resolvió la Fase 4

| Riesgo operativo | Estado antes | Estado después |
|---|---|---|
| Paper mezclado con backtest | No existía capa operativa separada | `paper.py` separado del replay económico |
| Estado implícito | Estado repartido y poco auditable | `PaperEngineState` serializable y explícito |
| Demo MT5 débil | Adapter mínimo sin validaciones fuertes | `mt5-check` y `run-demo-dry` con bloqueo explícito |
| Journaling operativo incompleto | Solo artefactos de backtest | logs operativos, snapshots y reportes reproducibles |
| Preparación para SL/TP dinámicos | No había hook claro | hooks listos para crecer sin reescribir el engine |

## Qué resolvió la Fase 5

| Riesgo operativo | Estado antes | Estado después |
|---|---|---|
| Divergencia broker/local | No había reconciliación explícita | `reconcile-state` + `reconciliation_report.json` |
| Reinicio inseguro | No había restore validado | restore bloqueante y snapshot runtime |
| Pérdida de conexión | Sin reconnect controlado | reconnect con attempts y estado operativo |
| Alertas débiles | Logs sin sink útil | `alerts_log.jsonl` + alertas accionables |
| Fuera de sesión | No había gate operativo explícito | session blocking con motivo auditable |
| Duplicados tras restart | Riesgo de reprocesar barras | `processing_state` e idempotencia básica |

## Qué resolvió la Fase 5.5

| Riesgo operativo | Estado antes | Estado después |
|---|---|---|
| Resistencia prolongada no medida | Solo validación por corrida | soak multi-cycle con health por ciclo |
| Fallos hostiles no reproducibles | Diagnóstico manual | chaos scenarios explícitos y repetibles |
| Gate operativo difuso | No había decisión consolidada | `go_no_go_report.json` con reglas |
| Reconstrucción de incidentes difícil | Artefactos dispersos | `cycle_summaries/` + `incident_log.jsonl` |

## Qué resolvió la Fase 6

| Riesgo cuantitativo | Estado antes | Estado después |
|---|---|---|
| Símbolos tratados casi igual | Sin perfilado serio por símbolo | `symbol_research.py` + `symbol_profile.json` |
| Threshold global único | Sin hardening económico por símbolo | `threshold_report.json` por símbolo |
| ML sin comparación rigurosa | Global vs símbolo no medido | `model_comparison_report.json` |
| Exits solo estáticos | No había SL/TP dinámicos implementados | `atr_dynamic` auditable en backtest y paper/demo |
| Gate de activación difuso por símbolo | No había `enabled/caution/disabled` | `symbol_enablement_report.json` + `strategy_profiles.json` |

## Qué sigue faltando antes de live real

- reconciliación con estado real del broker
- manejo de reconexión, latencia y ticks en tiempo real
- control de órdenes parcialmente llenadas y rechazos de servidor
- gestión de sesiones/mercados y horarios operativos
- persistencia transaccional más dura
- monitoreo, alertas y recuperación tras reinicio
- fills reales y partial fills sobre órdenes enviadas de verdad
- reconciliación más profunda por ticket/deal en operativa real
- alertas externas y supervisión fuera del proceso
- trailing stops, break-even y exits adaptativos más avanzados
- validación prolongada en demo seria por símbolo habilitado
- coordinación multi-symbol en condiciones demo más exigentes

## Condiciones mínimas antes de una futura fase 7

- varias corridas soak con `go`
- sin `no_go` en escenarios de reconnect controlado
- restore limpio bajo reinicios repetidos
- sin corrupción de journals ni discrepancias críticas
- estabilidad operativa sostenida, no solo una corrida feliz
- símbolos `enabled` con expectancy positiva y drawdown aceptable
- `caution` revisados manualmente antes de cualquier demo más seria

### Estado explícito de SL/TP dinámicos

Existe una primera capa **implementada** y auditable:

- `stop_policy=atr_dynamic`
- `target_policy=atr_dynamic`

Todavía **NO** existen trailing stops, break-even ni salidas adaptativas más complejas.

### Estado explícito de live real

IRIS-Bot sigue **NO habilitado para live real** tras esta fase.
