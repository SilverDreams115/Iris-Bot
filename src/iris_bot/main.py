from __future__ import annotations

import argparse

from iris_bot.cli import (
    build_command_handlers,
    command_choices,
    run_cli_command,
)
from iris_bot.commands.data import build_dataset_command
from iris_bot.config import load_settings


COMMAND_HANDLERS = build_command_handlers()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IRIS-Bot: Forex ML trading research framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run-backtest",
        choices=command_choices(COMMAND_HANDLERS),
        help="Command to run (default: run-backtest)",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        default=False,
        help="For run-backtest: run economic backtest fold-by-fold instead of a single pass over the test split.",
    )
    parser.add_argument(
        "--intrabar-policy",
        choices=["conservative", "optimistic"],
        default=None,
        metavar="POLICY",
        help="Override IRIS_BACKTEST_INTRABAR_POLICY for run-backtest.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    raise SystemExit(
        run_cli_command(
            command=args.command,
            settings=settings,
            command_handlers=COMMAND_HANDLERS,
            intrabar_policy_override=args.intrabar_policy,
            walk_forward=args.walk_forward,
        )
    )


if __name__ == "__main__":
    main()
