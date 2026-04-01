# IRIS-Bot

Base inicial para un bot de trading forex con machine learning, backtesting y conexion a MetaTrader 5.

## Enfoque inicial

El proyecto arranca con un flujo conservador y extensible:

1. Cargar datos OHLCV desde CSV o MT5.
2. Construir features tecnicos por simbolo y timeframe.
3. Entrenar un clasificador base.
4. Generar señales con umbral de confianza.
5. Aplicar gestion de riesgo.
6. Validar en backtesting antes de operar en demo live.

## Estructura

```text
src/iris_bot/
tests/
data/
```

## Uso

```bash
PYTHONPATH=src python3 -m iris_bot.main
```

Para descargar historico desde MT5:

```bash
export IRIS_MT5_ENABLED=true
export IRIS_MT5_LOGIN=123456
export IRIS_MT5_PASSWORD='tu_password'
export IRIS_MT5_SERVER='TuBroker-Demo'
export IRIS_MT5_PATH='/ruta/al/terminal64.exe'
PYTHONPATH=src python3 -m iris_bot.main fetch
```

Luego ejecutar backtesting:

```bash
PYTHONPATH=src python3 -m iris_bot.main backtest
```

## Formato de datos esperado

`data/market.csv`

```text
timestamp,symbol,timeframe,open,high,low,close,volume
2026-01-01T00:00:00,EURUSD,M5,1.1000,1.1010,1.0990,1.1005,120
```

## Defaults iniciales

- Simbolos: `EURUSD`, `GBPUSD`, `USDJPY`, `AUDUSD`
- Timeframes: `M5`, `M15`, `H1`
- Operativa: `long/short`
- Riesgo por trade: `1%`
- Balance minimo objetivo: `25 USD`
- Plataforma live: `MetaTrader 5`

## Variables de entorno

- `IRIS_SYMBOLS=EURUSD,GBPUSD,USDJPY,AUDUSD`
- `IRIS_TIMEFRAMES=M5,M15,H1`
- `IRIS_MT5_ENABLED=true`
- `IRIS_MT5_LOGIN=123456`
- `IRIS_MT5_PASSWORD=...`
- `IRIS_MT5_SERVER=Broker-Demo`
- `IRIS_MT5_PATH=/ruta/al/terminal`
- `IRIS_MT5_HISTORY_BARS=1500`

## Siguientes pasos

- Cargar datos historicos reales desde MT5.
- Añadir walk-forward validation.
- Incorporar filtros de spread y sesiones.
- Conectar ordenes demo con credenciales MT5.
