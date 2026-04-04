from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import timedelta

from iris_bot.data import Bar, group_bars


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    symbol: str
    timeframe: str
    code: str
    message: str


@dataclass(frozen=True)
class SeriesValidationSummary:
    symbol: str
    timeframe: str
    bars: int
    duplicate_timestamps: int
    non_increasing_timestamps: int
    missing_ohlc: int
    invalid_ohlc: int
    gap_count: int


@dataclass(frozen=True)
class ValidationReport:
    summaries: list[SeriesValidationSummary]
    issues: list[ValidationIssue]

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "summaries": [asdict(item) for item in self.summaries],
            "issues": [asdict(item) for item in self.issues],
        }


TIMEFRAME_DELTAS = {
    "M1": timedelta(minutes=1),
    "M5": timedelta(minutes=5),
    "M15": timedelta(minutes=15),
    "M30": timedelta(minutes=30),
    "H1": timedelta(hours=1),
    "H4": timedelta(hours=4),
    "D1": timedelta(days=1),
}


def validate_bars(bars: list[Bar]) -> ValidationReport:
    grouped = group_bars(bars)
    summaries: list[SeriesValidationSummary] = []
    issues: list[ValidationIssue] = []

    for (symbol, timeframe), series in sorted(grouped.items()):
        duplicate_timestamps = 0
        non_increasing_timestamps = 0
        missing_ohlc = 0
        invalid_ohlc = 0
        gap_count = 0
        expected_delta = TIMEFRAME_DELTAS.get(timeframe)

        for index, bar in enumerate(series):
            if any(value is None for value in (bar.open, bar.high, bar.low, bar.close)):
                missing_ohlc += 1
                issues.append(
                    ValidationIssue("error", symbol, timeframe, "missing_ohlc", f"Barra con OHLC faltante en {bar.timestamp.isoformat()}")
                )
            if not (bar.low <= min(bar.open, bar.close) and bar.high >= max(bar.open, bar.close) and bar.low <= bar.high):
                invalid_ohlc += 1
                issues.append(
                    ValidationIssue("error", symbol, timeframe, "invalid_ohlc", f"OHLC inconsistente en {bar.timestamp.isoformat()}")
                )

            if index == 0:
                continue

            previous = series[index - 1]
            delta = bar.timestamp - previous.timestamp
            if bar.timestamp == previous.timestamp:
                duplicate_timestamps += 1
                issues.append(
                    ValidationIssue("error", symbol, timeframe, "duplicate_timestamp", f"Timestamp duplicado en {bar.timestamp.isoformat()}")
                )
            elif bar.timestamp < previous.timestamp:
                non_increasing_timestamps += 1
                issues.append(
                    ValidationIssue("error", symbol, timeframe, "non_increasing_timestamp", f"Serie desordenada en {bar.timestamp.isoformat()}")
                )
            elif expected_delta is not None and delta > expected_delta:
                gap_count += 1
                issues.append(
                    ValidationIssue(
                        "warning",
                        symbol,
                        timeframe,
                        "gap_detected",
                        f"Gap detectado entre {previous.timestamp.isoformat()} y {bar.timestamp.isoformat()}",
                    )
                )

        summaries.append(
            SeriesValidationSummary(
                symbol=symbol,
                timeframe=timeframe,
                bars=len(series),
                duplicate_timestamps=duplicate_timestamps,
                non_increasing_timestamps=non_increasing_timestamps,
                missing_ohlc=missing_ohlc,
                invalid_ohlc=invalid_ohlc,
                gap_count=gap_count,
            )
        )

    return ValidationReport(summaries=summaries, issues=issues)
