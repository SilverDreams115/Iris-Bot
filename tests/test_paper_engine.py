from datetime import datetime, timedelta
import json

from iris_bot.config import BacktestConfig, RiskConfig
from iris_bot.operational import ExitPolicyConfig, write_operational_artifacts
from iris_bot.paper import ExecutionDecision, OrderIntent, PaperSessionConfig, run_paper_engine
from iris_bot.processed_dataset import ProcessedRow


def _row(
    ts: datetime,
    price: float = 1.1000,
    symbol: str = "EURUSD",
    bar_low: float | None = None,
    bar_high: float | None = None,
) -> ProcessedRow:
    return ProcessedRow(
        timestamp=ts,
        symbol=symbol,
        timeframe="M15",
        open=price,
        high=bar_high if bar_high is not None else price + 0.0006,
        low=bar_low if bar_low is not None else price - 0.0006,
        close=price,
        volume=100.0,
        label=1,
        label_reason="test",
        horizon_end_timestamp=ts.isoformat(),
        features={"atr_10": 0.0005, "atr_5": 0.0005},
    )


def _prob_signal(direction: int = 1) -> dict[int, float]:
    if direction == 1:
        return {1: 0.9, 0: 0.05, -1: 0.05}
    return {-1: 0.9, 0: 0.05, 1: 0.05}


def _prob_neutral() -> dict[int, float]:
    return {1: 0.2, 0: 0.6, -1: 0.2}


TEST_BACKTEST = BacktestConfig(
    starting_balance_usd=1000.0,
    use_atr_stops=False,
    fixed_stop_loss_pct=0.002,
    fixed_take_profit_pct=0.004,
    spread_pips=0.0,
    slippage_pips=0.0,
    commission_per_lot_per_side_usd=1.0,
    max_holding_bars=5,
)


def _base_config(**overrides) -> PaperSessionConfig:
    defaults = dict(
        mode="paper",
        threshold=0.5,
        trading_symbols=("EURUSD",),
        one_position_per_symbol=True,
        allow_long=True,
        allow_short=True,
        backtest=TEST_BACKTEST,
        risk=RiskConfig(),
    )
    defaults.update(overrides)
    return PaperSessionConfig(**defaults)


def test_paper_engine_opens_and_closes_position() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [
        _row(start),
        _row(start + timedelta(minutes=15), bar_low=1.0990, bar_high=1.1050),
        _row(start + timedelta(minutes=30)),
    ]
    probabilities = [_prob_signal(), _prob_neutral(), _prob_neutral()]

    artifacts = run_paper_engine(
        _base_config(exit_policy_config=ExitPolicyConfig()),
        rows,
        probabilities,
    )

    assert len(artifacts.closed_trades) == 1
    assert any(event.event_type == "position_opened" for event in artifacts.events)
    assert any(event.event_type == "take_profit_hit" for event in artifacts.events)


def test_paper_engine_rejects_cooldown_and_daily_loss() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [
        _row(start),
        _row(start + timedelta(minutes=15), bar_low=1.0900, bar_high=1.1001),
        _row(start + timedelta(minutes=30)),
        _row(start + timedelta(minutes=45)),
        _row(start + timedelta(minutes=60)),
    ]
    probabilities = [
        _prob_signal(),
        _prob_neutral(),
        _prob_signal(),
        _prob_neutral(),
        _prob_signal(),
    ]

    artifacts = run_paper_engine(
        _base_config(risk=RiskConfig(max_daily_loss_usd=0.01, cooldown_bars_after_loss=2)),
        rows,
        probabilities,
    )

    reasons = {event.reason for event in artifacts.events if event.event_type == "signal_rejected"}
    assert "max_daily_loss" in reasons or "cooldown_active" in reasons


def test_paper_engine_rejects_invalid_symbol_and_duplicate_position() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [
        _row(start, symbol="INVALID"),
        _row(start + timedelta(minutes=15), symbol="INVALID"),
        _row(start + timedelta(minutes=30), symbol="EURUSD"),
        _row(start + timedelta(minutes=45), symbol="EURUSD"),
        _row(start + timedelta(minutes=60), symbol="EURUSD"),
    ]
    probabilities = [
        _prob_signal(),
        _prob_neutral(),
        _prob_signal(),
        _prob_signal(),
        _prob_neutral(),
    ]

    artifacts = run_paper_engine(
        _base_config(),
        rows,
        probabilities,
    )

    assert artifacts.state.blocked_trades_summary["symbol_not_configured"] >= 1


def test_paper_engine_enforces_max_open_positions() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [
        _row(start, symbol="EURUSD"),
        _row(start, symbol="GBPUSD"),
        _row(start + timedelta(minutes=15), symbol="EURUSD"),
        _row(start + timedelta(minutes=15), symbol="GBPUSD"),
    ]
    probabilities = [_prob_signal(), _prob_signal(), _prob_neutral(), _prob_neutral()]

    artifacts = run_paper_engine(
        _base_config(
            trading_symbols=("EURUSD", "GBPUSD"),
            risk=RiskConfig(max_open_positions=1),
        ),
        rows,
        probabilities,
    )

    assert artifacts.state.blocked_trades_summary["max_open_positions"] >= 1


def test_demo_dry_validator_records_order_simulation() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [_row(start), _row(start + timedelta(minutes=15)), _row(start + timedelta(minutes=30))]
    probabilities = [_prob_signal(), _prob_neutral(), _prob_neutral()]

    def validator(intent: OrderIntent) -> ExecutionDecision:
        return ExecutionDecision(
            accepted=True,
            reason="dry_run_only",
            details={"request": {"symbol": intent.symbol, "volume": intent.volume}},
        )

    artifacts = run_paper_engine(
        _base_config(mode="demo_dry", execution_validator=validator),
        rows,
        probabilities,
    )

    assert any(event.event_type == "order_simulated" for event in artifacts.events)


def test_journals_and_reports_are_written(tmp_path) -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    rows = [_row(start), _row(start + timedelta(minutes=15)), _row(start + timedelta(minutes=30))]
    probabilities = [_prob_signal(), _prob_neutral(), _prob_neutral()]
    artifacts = run_paper_engine(
        _base_config(),
        rows,
        probabilities,
    )

    write_operational_artifacts(
        tmp_path,
        artifacts,
        {"exit_policy": {"stop_policy": "static", "target_policy": "static"}},
    )

    assert (tmp_path / "signal_log.csv").exists()
    assert (tmp_path / "execution_journal.csv").exists()
    assert (tmp_path / "open_positions_snapshot.json").exists()
    assert (tmp_path / "closed_trades.csv").exists()
    assert (tmp_path / "daily_summary.json").exists()
    assert (tmp_path / "run_report.json").exists()
    validation = json.loads((tmp_path / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["state_checks"]["dynamic_stop_policy_implemented"] is False
