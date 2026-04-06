# IRIS-Bot Architecture

## Objetivo

Documento corto de referencia para la organización interna actual del proyecto.
No describe estrategia de trading; describe límites de módulos y ownership técnico.

## Capas principales

- `config.py`
  Fachada pública de configuración.
- `config_types.py`
  Dataclasses y shape de `Settings`.
- `config_runtime.py`
  Carga desde entorno y validación de configuración.

- `main.py`
  Entry point del CLI.
- `cli.py`
  Registro y dispatch de comandos.
- `commands/`
  Adaptadores del CLI por dominio.

- `backtest.py`
  Motor principal de backtest e IO de reportes.
- `backtest_pricing.py`
  Pricing, costes e intrabar.
- `backtest_analysis.py`
  Probabilidades, mark-to-market y métricas agregadas.

- `paper.py`
  Orquestación de paper trading.
- `paper_types.py`
  Tipos públicos compartidos con ejecución resiliente.
- `paper_engine_support.py`
  Helpers de estado, gates y apertura de posiciones.

- `resilient.py`
  Orquestación resiliente de alto nivel.
- `resilient_models.py`
  Tipos compartidos y utilidades comunes.
- `resilient_state.py`
  Restore, persistencia, sesión e idempotencia.
- `resilient_reconcile.py`
  Reconciliación broker/local, reconnect y validator demo.

- `governance.py`
  Fachada pública del dominio de governance y reporting restante.
- `governance_active.py`
  Resolución de perfiles activos e inputs de validación.
- `governance_validation.py`
  Validación y escritura de perfiles validados al registry.
- `governance_promotion.py`
  Promoción y rollback bajo lock del registry.

## Reglas de diseño

- Mantener fachadas públicas estables para evitar romper tests y consumidores internos.
- Extraer primero helpers puros o flujos autocontenidos antes de dividir orquestadores.
- No mezclar refactor estructural con cambios funcionales.
- Validar cada corte con suite completa.

## Puntos aún grandes

- `governance.py`
  Sigue concentrando review/promotion gating y reporting de diagnóstico.
- `symbol_validation.py`
  Sigue siendo uno de los mayores focos de acoplamiento.
- `demo_readiness.py`
  Todavía mezcla diagnóstico, policy y reporting.

## Flujo de trabajo recomendado

```bash
make bootstrap
make smoke
make test
```
