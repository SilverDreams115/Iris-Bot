# IRIS-Bot

IRIS-Bot es un framework de investigacion y operacion controlada para trading FX con MetaTrader 5. El repositorio ya cubre adquisicion de datos, construccion de datasets, entrenamiento y evaluacion economica de modelos, paper trading, demo dry-run, governance por perfil de estrategia, reconciliacion operativa y pruebas de resistencia. No es, en su estado actual, un bot de live trading para capital real.

Este README describe el estado real del proyecto tal como existe hoy en el codigo y en los artefactos de validacion del repositorio. No documenta features aspiracionales.

## Estado real del proyecto

- El proyecto funciona hoy como framework de research y validacion operativa para FX sobre MT5.
- El flujo normal soportado es: historico MT5 -> dataset procesado -> experimento/backtest -> walk-forward -> paper o demo dry-run -> governance/readiness.
- Existe una prueba separada de ejecucion real en cuenta demo: `run-demo-live-probe`. Esa prueba abre una orden real minima, la audita y la cierra.
- No existe un flujo general de live trading sobre cuenta real documentado ni validado en este repositorio.
- El gate `demo_execution_readiness` sigue siendo deliberadamente conservador y no declara el sistema como listo para live. El `demo_live_checklist` es un chequeo distinto para la prueba demo-live separada.

## Lo que el repositorio si puede hacer

### 1. Datos y dataset

Comandos disponibles:

```bash
python -m iris_bot.main fetch
python -m iris_bot.main fetch-historical
python -m iris_bot.main validate-data
python -m iris_bot.main build-dataset
python -m iris_bot.main inspect-dataset
```

Capacidades reales:

- descarga historico desde MT5
- valida integridad basica de los datos descargados
- construye dataset procesado con schema y manifest
- inspecciona el dataset procesado para uso de research y backtesting

### 2. Entrenamiento, backtest y research

Comandos disponibles:

```bash
python -m iris_bot.main run-experiment
python -m iris_bot.main run-backtest
python -m iris_bot.main run-backtest --walk-forward
python -m iris_bot.main backtest
```

Capacidades reales:

- entrenamiento del modelo principal con XGBoost
- evaluacion economica sobre test split
- walk-forward economico fold por fold
- politica intrabar configurable (`conservative` u `optimistic`)
- perfiles por simbolo y filtrado de filas permitidas por perfil
- comparacion de modelos por simbolo y validacion por estrategia
- evaluacion de salidas dinamicas

Comandos de research por simbolo y governance cuantitativa:

```bash
python -m iris_bot.main build-symbol-profiles
python -m iris_bot.main run-symbol-research
python -m iris_bot.main run-strategy-validation
python -m iris_bot.main audit-strategy-block-causes
python -m iris_bot.main compare-symbol-models
python -m iris_bot.main evaluate-dynamic-exits
python -m iris_bot.main symbol-go-no-go
```

### 3. Significancia estadistica

El pipeline de experimentos ya integra permutation testing y Deflated Sharpe Ratio.

Estado real:

- existe permutation testing sobre labels permutadas para el flujo walk-forward
- el reporte de experimento incluye `p_value`, percentil dentro de la distribucion nula y resumen por trial
- existe calculo de `deflated_sharpe_ratio`
- el evaluador reutiliza el replay economico walk-forward existente en lugar de inventar otro scorer

Esto vive en el pipeline de experimentos; no es un script aislado fuera del flujo principal.

### 4. Paper trading y demo dry-run

Comandos disponibles:

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

Capacidades reales:

- paper trading con estado operativo persistible
- demo dry-run que valida requests contra MT5 sin enviar ordenes reales como flujo normal
- reconciliacion broker <-> estado interno
- chequeo de restauracion y estado operativo
- sesion resiliente con persistencia y recuperacion controlada

Importante:

- el flujo operativo normal del bot sigue tratando la ejecucion real como fase separada
- el modulo de readiness `demo_execution_readiness` sigue declarando `order_send_integrated = false` para ese pipeline normal

### 5. Governance, promotion y evidencia

Comandos disponibles:

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

Capacidades reales:

- registro de perfiles de estrategia por simbolo
- promotion/rollback de perfiles
- materializacion de perfiles activos
- auditoria de locks y checksums del registry
- evidencia de lifecycle y status del evidence store
- gates para approved-demo y separacion explicita del portfolio activo

### 6. Soak, endurance y chaos

Comandos disponibles:

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

Capacidades reales:

- pruebas multi-ciclo de estabilidad operacional
- pruebas por simbolo y por universo habilitado
- reporting de estabilidad y decision go/caution/no-go
- escenarios de chaos controlado

## Ejecucion real en cuenta demo: que existe y que no existe

### Lo que si existe

Comandos disponibles:

```bash
python -m iris_bot.main run-demo-live-checklist
python -m iris_bot.main run-demo-live-probe
```

`run-demo-live-probe` hace una validacion real y separada:

- confirma que la cuenta conectada sea demo
- toma el primer simbolo configurado
- envia una orden real minima al broker demo
- localiza la posicion abierta
- envia la orden de cierre
- escribe un reporte auditable en `runs/*_demo_live_probe/demo_live_probe_report.json`

Este flujo ya fue validado en una cuenta `MetaQuotes-Demo` durante el trabajo actual del repositorio. El probe abrio y cerro una posicion real de `EURUSD` sin dejar posiciones remanentes.

### Lo que no debe confundirse con eso

- `run-demo-live-probe` no convierte al sistema en un bot live de proposito general
- `demo_execution_readiness` no afirma que el pipeline normal ya enrute ordenes reales
- la ejecucion real validada hasta ahora es una prueba controlada de apertura/cierre sobre cuenta demo, no una fase de operacion automatica continua sobre cuenta real

## Requisitos de entorno reales

### Python y entorno local

Bootstrap recomendado:

```bash
make bootstrap
```

Uso manual:

```bash
./.venv/bin/python -m pip install -e ".[dev]"
./.venv/bin/python -m pytest
./.venv/bin/python -m iris_bot.main --help
```

### MetaTrader 5

Realidad actual del adaptador MT5:

- el paquete `MetaTrader5` esta declarado como dependencia solo en Windows
- las operaciones que dependen de `MetaTrader5` no son portables a Linux/WSL de la misma manera
- el trabajo de research, tests y mucha logica del proyecto puede correrse desde Linux/WSL
- la conexion real a MT5 y la prueba de orden demo se validaron desde Python de Windows

Variables de entorno soportadas por el proyecto incluyen las de MT5 y otras de configuracion. El repositorio ya carga automaticamente un archivo `.env` si existe.

## Limitaciones reales actuales

Estas limitaciones existen hoy en el codigo o en el estado operativo validado:

- no hay soporte documentado y validado para live trading en cuenta real
- la prueba demo-live existe, pero esta separada del pipeline normal del bot
- la compatibilidad completa con MT5 depende de Python de Windows por la libreria `MetaTrader5`
- el gate `demo_execution_readiness` sigue siendo conservador y puede devolver `caution` aunque el `demo_live_checklist` para el probe separado pase
- el repositorio contiene bastante infraestructura de governance y operacion, pero eso no implica por si solo edge estadistico ni rentabilidad

## Cosas que este README no afirma porque hoy no estan demostradas aqui

- no afirma rentabilidad futura
- no afirma alpha real
- no afirma readiness para capital real
- no afirma que todas las rutas operativas esten igualmente validadas en Windows y Linux
- no afirma que exista un economic calendar gate integrado en el flujo principal
- no afirma meta-labeling en produccion
- no afirma trailing stop como politica operativa ya integrada en el flujo principal

## Estructura practica del proyecto

Modulos relevantes hoy:

- `src/iris_bot/data.py`, `processed_dataset.py`, `labels.py`, `preprocessing.py`
- `src/iris_bot/xgb_model.py`, `experiments.py`, `wf_backtest.py`, `significance.py`
- `src/iris_bot/paper.py`, `resilient.py`, `operational.py`, `mt5.py`
- `src/iris_bot/governance.py`, `profile_registry.py`, `profile_evidence.py`, `lifecycle.py`
- `src/iris_bot/demo_readiness.py`, `demo_live_checklist.py`, `demo_live_probe.py`

## Comandos mas utiles para empezar

```bash
python -m iris_bot.main fetch
python -m iris_bot.main build-dataset
python -m iris_bot.main run-experiment
python -m iris_bot.main run-backtest --walk-forward
python -m iris_bot.main run-paper
python -m iris_bot.main run-demo-dry
python -m iris_bot.main mt5-check
python -m iris_bot.main demo-execution-readiness
python -m iris_bot.main run-demo-live-checklist
python -m iris_bot.main run-demo-live-probe
```

## Validacion conocida en este estado del repositorio

Validaciones reales ya ejecutadas sobre este estado de trabajo:

- la suite de tests del repositorio pasa completa
- el probe demo-live abrio y cerro una orden real en cuenta demo
- el checklist demo-live para ese flujo separado ya corre

Lo correcto es seguir tratando el proyecto como un framework serio de research y validacion operativa, no como un sistema ya listo para operar capital real sin mas trabajo.
