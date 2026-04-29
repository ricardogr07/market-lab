from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from time import perf_counter

from marketlab._version import get_version
from marketlab.config import load_config
from marketlab.env import load_env_file
from marketlab.log import (
    ExecutionContext,
    bind_execution_context,
    configure_logging,
    duration_ms_since,
    emit_structured_log,
)
from marketlab.paper import (
    decide_paper_proposal,
    get_paper_status,
    run_agent_approval_loop,
    run_paper_decision,
    run_paper_report,
    run_paper_submit,
    run_scheduler_loop,
)
from marketlab.paper.observability import (
    paper_execution_context,
    root_execution_context,
)
from marketlab.pipeline import backtest, prepare_data, run_experiment, train_models
from marketlab.resources.templates import CONFIG_TEMPLATE_NAMES, write_config_template

LOGGER = logging.getLogger(__name__)


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


def _run_logged_paper_command(
    command_name: str,
    *,
    action: Callable[[ExecutionContext], tuple[int, str | None]],
    proposal_id: str | None = None,
) -> int:
    root_context = root_execution_context(
        deployment="local_cli",
        phase=command_name,
        proposal_id=proposal_id,
        details={"command": command_name},
    )
    emit_structured_log(
        LOGGER,
        logging.INFO,
        f"Starting {command_name} command.",
        event="paper.command.start",
        execution_context=root_context,
    )
    start_time = perf_counter()
    try:
        with bind_execution_context(root_context):
            exit_code, outcome = action(root_context)
    except Exception as exc:
        emit_structured_log(
            LOGGER,
            logging.ERROR,
            f"{command_name} command failed.",
            event="paper.command.error",
            execution_context=paper_execution_context(
                root_context,
                phase=command_name,
                outcome="error",
                duration_ms=duration_ms_since(start_time),
                details={"command": command_name},
            ),
            exc_info=exc,
        )
        raise
    emit_structured_log(
        LOGGER,
        logging.INFO,
        f"Finished {command_name} command.",
        event="paper.command.finish",
        execution_context=paper_execution_context(
            root_context,
            phase=command_name,
            outcome=outcome or "success",
            duration_ms=duration_ms_since(start_time),
            details={"command": command_name},
        ),
    )
    return exit_code


def _run_paper_decision_command(
    config,
    *,
    execution_context: ExecutionContext,
) -> tuple[int, str | None]:
    result = run_paper_decision(config, execution_context=execution_context)
    print(result.get("proposal_path", result["status_path"]))
    return 0, str(result.get("status", {}).get("status", "success"))


def _run_paper_submit_command(
    config,
    *,
    execution_context: ExecutionContext,
) -> tuple[int, str | None]:
    result = run_paper_submit(config, execution_context=execution_context)
    print(result.get("submission_path", result["status_path"]))
    return 0, str(result.get("status", {}).get("status", "success"))


def _run_paper_approve_command(
    config,
    *,
    proposal_id: str,
    decision: str,
    actor: str,
    execution_context: ExecutionContext,
) -> tuple[int, str | None]:
    result = decide_paper_proposal(
        config,
        proposal_id=proposal_id,
        decision=decision,
        actor=actor,
        execution_context=execution_context,
    )
    print(result["approval_path"])
    return 0, str(result.get("status", {}).get("status", "success"))


def _run_paper_status_command(
    config,
    *,
    execution_context: ExecutionContext,
) -> tuple[int, str | None]:
    del execution_context
    print(json.dumps(get_paper_status(config), indent=2, sort_keys=True))
    return 0, "success"


def _run_paper_agent_approve_command(
    config,
    *,
    once: bool,
    execution_context: ExecutionContext,
) -> tuple[int, str | None]:
    del execution_context
    run_agent_approval_loop(config, once=once)
    return 0, "success"


def _run_paper_scheduler_command(
    config,
    *,
    once: bool,
    execution_context: ExecutionContext,
) -> tuple[int, str | None]:
    del execution_context
    run_scheduler_loop(config, once=once)
    return 0, "success"


def _run_paper_report_command(
    config,
    *,
    start_date: str,
    end_date: str,
    execution_context: ExecutionContext,
) -> tuple[int, str | None]:
    del execution_context
    result = run_paper_report(config, start_date=start_date, end_date=end_date)
    print(result["report_path"])
    return 0, "success"


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

    load_env_file()
    config = load_config(args.config)

    if args.command == "paper-decision":
        return _run_logged_paper_command(
            args.command,
            action=lambda execution_context: _run_paper_decision_command(
                config,
                execution_context=execution_context,
            ),
        )

    if args.command == "paper-submit":
        return _run_logged_paper_command(
            args.command,
            action=lambda execution_context: _run_paper_submit_command(
                config,
                execution_context=execution_context,
            ),
        )

    if args.command == "paper-approve":
        return _run_logged_paper_command(
            args.command,
            proposal_id=args.proposal_id,
            action=lambda execution_context: _run_paper_approve_command(
                config,
                proposal_id=args.proposal_id,
                decision=args.decision,
                actor=args.actor,
                execution_context=execution_context,
            ),
        )

    if args.command == "paper-status":
        return _run_logged_paper_command(
            args.command,
            action=lambda execution_context: _run_paper_status_command(
                config,
                execution_context=execution_context,
            ),
        )

    if args.command == "paper-agent-approve":
        return _run_logged_paper_command(
            args.command,
            action=lambda execution_context: _run_paper_agent_approve_command(
                config,
                once=args.once,
                execution_context=execution_context,
            ),
        )

    if args.command == "paper-scheduler":
        return _run_logged_paper_command(
            args.command,
            action=lambda execution_context: _run_paper_scheduler_command(
                config,
                once=args.once,
                execution_context=execution_context,
            ),
        )

    if args.command == "paper-report":
        return _run_logged_paper_command(
            args.command,
            action=lambda execution_context: _run_paper_report_command(
                config,
                start_date=args.start,
                end_date=args.end,
                execution_context=execution_context,
            ),
        )

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
