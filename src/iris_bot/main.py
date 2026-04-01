from __future__ import annotations

import argparse

from iris_bot.backtest import run_backtest
from iris_bot.config import settings
from iris_bot.data import group_bars, load_bars, write_bars
from iris_bot.features import build_feature_vectors
from iris_bot.mt5 import MT5Client


def fetch_market_data() -> int:
    client = MT5Client(settings.mt5)
    if not client.connect():
        print("No se pudo conectar a MetaTrader 5. Revisa IRIS_MT5_* y la instalacion del terminal.")
        return 1

    try:
        all_bars = []
        for symbol in settings.trading.symbols:
            for timeframe in settings.trading.timeframes:
                bars = client.fetch_historical_bars(symbol, timeframe, settings.mt5.history_bars)
                all_bars.extend(bars)
                print(f"descargado {symbol} {timeframe} bars={len(bars)}")

        output_path = settings.data_dir / "market.csv"
        write_bars(output_path, all_bars)
        print(f"guardado {len(all_bars)} barras en {output_path}")
        return 0
    finally:
        client.shutdown()


def run_backtest_command() -> int:
    all_bars = load_bars(settings.data_dir / "market.csv")
    grouped = group_bars(all_bars)

    total_vectors = 0
    total_trades = 0
    total_symbols = 0
    for key, bars in grouped.items():
        symbol, timeframe = key
        if symbol not in settings.trading.symbols or timeframe not in settings.trading.timeframes:
            continue
        vectors = build_feature_vectors(bars)
        total_vectors += len(vectors)
        total_symbols += 1
        result = run_backtest(
            rows=vectors,
            starting_balance=max(settings.risk.min_balance_usd, 25.0),
            risk=settings.risk,
            trading=settings.trading,
        )
        total_trades += len(result.trades)
        print(
            f"{symbol} {timeframe} trades={len(result.trades)} "
            f"ending_balance={result.ending_balance:.2f} win_rate={result.win_rate:.2%}"
        )

    print(
        f"series={total_symbols} vectors={total_vectors} "
        f"symbols={','.join(settings.trading.symbols)} timeframes={','.join(settings.trading.timeframes)} "
        f"mt5_enabled={settings.mt5.enabled} trades={total_trades}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IRIS-Bot")
    parser.add_argument("command", nargs="?", default="backtest", choices=("backtest", "fetch"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "fetch":
        raise SystemExit(fetch_market_data())
    raise SystemExit(run_backtest_command())


if __name__ == "__main__":
    main()
