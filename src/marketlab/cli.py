from __future__ import annotations

import argparse

from marketlab._version import get_version
from marketlab.config import load_config
from marketlab.log import configure_logging
from marketlab.pipeline import backtest, prepare_data, run_experiment, train_models
from marketlab.resources.templates import CONFIG_TEMPLATE_NAMES, write_config_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marketlab")
    parser.add_argument("--version", action="version", version=f"marketlab {get_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("prepare-data", "backtest", "run-experiment", "train-models"):
        command = subparsers.add_parser(command_name)
        command.add_argument("--config", required=True)

    subparsers.add_parser("list-configs")

    write_config = subparsers.add_parser("write-config")
    write_config.add_argument("--name", required=True, choices=CONFIG_TEMPLATE_NAMES)
    write_config.add_argument("--output", required=True)
    write_config.add_argument("--force", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-configs":
        for template_name in CONFIG_TEMPLATE_NAMES:
            print(template_name)
        return 0

    if args.command == "write-config":
        try:
            output_path = write_config_template(args.name, args.output, force=args.force)
        except (FileExistsError, KeyError) as exc:
            parser.error(str(exc))
        print(output_path)
        return 0

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
