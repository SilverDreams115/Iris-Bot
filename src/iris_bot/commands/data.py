from __future__ import annotations

from typing import cast

from iris_bot.config import Settings
from iris_bot.datasets import write_dataset_bundle
from iris_bot.data import load_bars
from iris_bot.logging_utils import build_run_directory, configure_logging, write_json_report
from iris_bot.mt5 import MT5Client
from iris_bot.processed_dataset import build_processed_dataset, load_processed_dataset, write_processed_dataset
from iris_bot.validation import validate_bars


def fetch_market_data(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "fetch")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    client = MT5Client(settings.mt5)
    if not client.connect():
        logger.error("No se pudo conectar a MetaTrader 5. Revisa IRIS_MT5_* y la instalacion del terminal.")
        return 1
    try:
        all_bars = []
        for symbol in settings.trading.symbols:
            for timeframe in settings.trading.timeframes:
                bars = client.fetch_historical_bars(symbol, timeframe, settings.mt5.history_bars)
                all_bars.extend(bars)
                logger.info("descargado symbol=%s timeframe=%s bars=%s", symbol, timeframe, len(bars))
        report = validate_bars(all_bars)
        write_json_report(run_dir, "validation_report.json", report.to_dict())
        manifest = write_dataset_bundle(
            dataset_path=settings.data.raw_dataset_path,
            metadata_path=settings.data.raw_metadata_path,
            bars=all_bars,
            source="mt5",
            history_bars_requested=settings.mt5.history_bars,
            extra={"run_dir": str(run_dir)},
        )
        logger.info("guardado dataset=%s rows=%s", settings.data.raw_dataset_path, manifest.row_count)
        logger.info("validation_is_valid=%s issues=%s", report.is_valid, len(report.issues))
        return 0
    finally:
        client.shutdown()


def validate_market_data(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "validate")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    bars = load_bars(settings.data.raw_dataset_path)
    if not bars:
        logger.error("No hay dataset en %s", settings.data.raw_dataset_path)
        return 1
    report = validate_bars(bars)
    write_json_report(run_dir, "validation_report.json", report.to_dict())
    logger.info("dataset=%s rows=%s", settings.data.raw_dataset_path, len(bars))
    logger.info("validation_is_valid=%s issues=%s", report.is_valid, len(report.issues))
    return 0 if report.is_valid else 2


def build_dataset_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "build_dataset")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    bars = load_bars(settings.data.raw_dataset_path)
    if not bars:
        logger.error("No hay dataset crudo en %s", settings.data.raw_dataset_path)
        return 1
    validation_report = validate_bars(bars)
    write_json_report(run_dir, "raw_validation_report.json", validation_report.to_dict())
    if not validation_report.is_valid:
        logger.error("El dataset crudo no paso validacion. Revisa %s", run_dir / "raw_validation_report.json")
        return 2
    dataset = build_processed_dataset(bars, settings.labeling)
    if not dataset.rows:
        logger.error("No se pudieron construir filas procesadas")
        return 3
    write_processed_dataset(
        dataset=dataset,
        dataset_path=settings.experiment.processed_dataset_path,
        manifest_path=settings.experiment.processed_manifest_path,
        schema_path=settings.experiment.processed_schema_path,
    )
    write_json_report(run_dir, "processed_manifest.json", dataset.manifest)
    write_json_report(run_dir, "processed_schema.json", dataset.schema)
    logger.info("processed_dataset=%s rows=%s label_mode=%s", settings.experiment.processed_dataset_path, len(dataset.rows), dataset.label_mode)
    return 0


def inspect_dataset_command(settings: Settings) -> int:
    run_dir = build_run_directory(settings.data.runs_dir, "inspect_dataset")
    logger = configure_logging(run_dir, settings.logging.level, settings.logging.format)
    try:
        dataset = load_processed_dataset(
            settings.experiment.processed_dataset_path,
            settings.experiment.processed_schema_path,
            settings.experiment.processed_manifest_path,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    write_json_report(run_dir, "processed_manifest.json", dataset.manifest)
    write_json_report(run_dir, "processed_schema.json", dataset.schema)
    logger.info(
        "processed_dataset=%s rows=%s features=%s symbols=%s timeframes=%s",
        settings.experiment.processed_dataset_path,
        len(dataset.rows),
        len(dataset.feature_names),
        ",".join(cast(list[str], dataset.manifest["symbols"])),
        ",".join(cast(list[str], dataset.manifest["timeframes"])),
    )
    return 0
