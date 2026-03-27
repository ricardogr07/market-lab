from __future__ import annotations

import argparse

from marketlab.config import load_config
from marketlab.log import configure_logging
from marketlab.pipeline import backtest, prepare_data, run_experiment, train_models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marketlab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("prepare-data", "backtest", "run-experiment", "train-models"):
        command = subparsers.add_parser(command_name)
        command.add_argument("--config", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "prepare-data":
        _, panel_path = prepare_data(config)
        print(panel_path)
        return 0

    if args.command == "backtest":
        artifacts = backtest(config)
        print(artifacts.run_dir)
        return 0

    if args.command == "run-experiment":
        artifacts = run_experiment(config)
        print(artifacts.run_dir)
        return 0

    if args.command == "train-models":
        artifacts = train_models(config)
        print(artifacts.run_dir)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
