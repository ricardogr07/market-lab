from __future__ import annotations

import argparse
import json

from marketlab._version import get_version
from marketlab.config import load_config
from marketlab.log import configure_logging
from marketlab.paper import (
    decide_paper_proposal,
    get_paper_status,
    run_agent_approval_loop,
    run_paper_decision,
    run_paper_report,
    run_paper_submit,
    run_scheduler_loop,
)
from marketlab.pipeline import backtest, prepare_data, run_experiment, train_models
from marketlab.resources.templates import CONFIG_TEMPLATE_NAMES, write_config_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marketlab")
    parser.add_argument("--version", action="version", version=f"marketlab {get_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("prepare-data", "backtest", "run-experiment", "train-models"):
        command = subparsers.add_parser(command_name)
        command.add_argument("--config", required=True)

    paper_decision = subparsers.add_parser("paper-decision")
    paper_decision.add_argument("--config", required=True)

    paper_submit = subparsers.add_parser("paper-submit")
    paper_submit.add_argument("--config", required=True)

    paper_approve = subparsers.add_parser("paper-approve")
    paper_approve.add_argument("--config", required=True)
    paper_approve.add_argument("--proposal-id", required=True)
    paper_approve.add_argument("--decision", required=True, choices=("approve", "reject"))
    paper_approve.add_argument("--actor", required=True, choices=("agent", "manual"))

    paper_status = subparsers.add_parser("paper-status")
    paper_status.add_argument("--config", required=True)

    paper_agent_approve = subparsers.add_parser("paper-agent-approve")
    paper_agent_approve.add_argument("--config", required=True)
    paper_agent_approve.add_argument("--once", action="store_true")

    paper_scheduler = subparsers.add_parser("paper-scheduler")
    paper_scheduler.add_argument("--config", required=True)
    paper_scheduler.add_argument("--once", action="store_true")

    paper_report = subparsers.add_parser("paper-report")
    paper_report.add_argument("--config", required=True)
    paper_report.add_argument("--start", required=True)
    paper_report.add_argument("--end", required=True)

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

    if args.command == "paper-decision":
        result = run_paper_decision(config)
        print(result.get("proposal_path", result["status_path"]))
        return 0

    if args.command == "paper-submit":
        result = run_paper_submit(config)
        print(result.get("submission_path", result["status_path"]))
        return 0

    if args.command == "paper-approve":
        result = decide_paper_proposal(
            config,
            proposal_id=args.proposal_id,
            decision=args.decision,
            actor=args.actor,
        )
        print(result["approval_path"])
        return 0

    if args.command == "paper-status":
        print(json.dumps(get_paper_status(config), indent=2, sort_keys=True))
        return 0

    if args.command == "paper-agent-approve":
        run_agent_approval_loop(config, once=args.once)
        return 0

    if args.command == "paper-scheduler":
        run_scheduler_loop(config, once=args.once)
        return 0

    if args.command == "paper-report":
        result = run_paper_report(config, start_date=args.start, end_date=args.end)
        print(result["report_path"])
        return 0

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
