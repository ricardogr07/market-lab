from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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
from marketlab.paper.contracts import PaperHostedExecutionContext, PaperHostedPhase
from marketlab.paper.notifications import build_telegram_paper_notification_sink
from marketlab.paper.observability import (
    hosted_execution_details,
    paper_execution_context,
    root_execution_context,
)
from marketlab.paper.outbox import (
    PAPER_NOTIFICATION_EVENT_TYPE,
    build_paper_outbox_publisher,
    deliver_pending_paper_notifications,
    deliver_pending_paper_outbox,
)
from marketlab.paper.persistence import (
    build_paper_uow_factory,
    migrate_paper_postgres_database,
    sync_paper_review_artifacts,
)
from marketlab.paper.service_bus import (
    PAPER_APPROVAL_REQUEST_EVENT_TYPE,
    receive_paper_approval_requests,
)
from marketlab.pipeline import backtest, prepare_data, run_experiment, train_models
from marketlab.reports.phase8_bull_counterfactual import (
    write_phase8_bull_counterfactual,
)
from marketlab.reports.phase8_bull_participation import write_phase8_bull_participation
from marketlab.reports.phase8_grid_compare import write_phase8_grid_comparison
from marketlab.reports.phase8_methodology import write_phase8_methodology_review
from marketlab.reports.phase8_regime_policy_sweep import (
    write_phase8_regime_policy_sweep,
)
from marketlab.reports.phase8_score_diagnostic import write_phase8_score_diagnostic
from marketlab.reports.phase8_selection_probe import write_phase8_selection_probe
from marketlab.reports.phase8_summary import write_phase8_run_summary
from marketlab.reports.phase8_target_diagnostic import write_phase8_target_diagnostic
from marketlab.reports.phase8_target_profile_sweep import (
    write_phase8_target_profile_sweep,
)
from marketlab.resources.templates import CONFIG_TEMPLATE_NAMES, write_config_template
from marketlab.shadow import cli as shadow_cli
from marketlab.targets import build_modeling_dataset

LOGGER = logging.getLogger(__name__)

HOSTED_METADATA_ENV = {
    "deployment_id": "MARKETLAB_DEPLOYMENT_ID",
    "environment": "MARKETLAB_ENVIRONMENT",
    "execution_id": "MARKETLAB_EXECUTION_ID",
    "correlation_id": "MARKETLAB_CORRELATION_ID",
    "idempotency_key": "MARKETLAB_IDEMPOTENCY_KEY",
    "trigger_source": "MARKETLAB_TRIGGER_SOURCE",
    "requested_at": "MARKETLAB_REQUESTED_AT",
    "config_version": "MARKETLAB_CONFIG_VERSION",
    "image_digest": "MARKETLAB_IMAGE_DIGEST",
}
HOSTED_COMMAND_PHASES: dict[str, PaperHostedPhase] = {
    "paper-decision": "decision",
    "paper-submit": "submit",
    "paper-approve": "agent_approve",
    "paper-agent-approve": "agent_approve",
    "paper-scheduler": "decision",
}
SHADOW_COMMANDS = {
    "phase9-shadow-decision": "main",
    "phase9-shadow-scheduler": "scheduler_main",
    "phase9-shadow-status": "status_main",
    "phase9-shadow-report": "report_main",
}


def _add_hosted_execution_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--deployment-id", dest="hosted_deployment_id")
    command.add_argument("--environment", dest="hosted_environment")
    command.add_argument("--execution-id", dest="hosted_execution_id")
    command.add_argument("--correlation-id", dest="hosted_correlation_id")
    command.add_argument("--idempotency-key", dest="hosted_idempotency_key")
    command.add_argument("--trigger-source", dest="hosted_trigger_source")
    command.add_argument("--requested-at", dest="hosted_requested_at")
    command.add_argument("--config-version", dest="hosted_config_version")
    command.add_argument("--image-digest", dest="hosted_image_digest")


def _hosted_value(args: argparse.Namespace, field_name: str) -> str:
    explicit = str(getattr(args, f"hosted_{field_name}", "") or "").strip()
    if explicit != "":
        return explicit
    return os.environ.get(HOSTED_METADATA_ENV[field_name], "").strip()


def _resolve_hosted_execution_context(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    phase: PaperHostedPhase,
    allow_env_phase: bool = False,
) -> PaperHostedExecutionContext | None:
    values = {
        field_name: _hosted_value(args, field_name)
        for field_name in HOSTED_METADATA_ENV
    }
    env_phase = os.environ.get("MARKETLAB_PHASE", "").strip()
    if env_phase != "" and not allow_env_phase and env_phase != phase:
        parser.error(f"MARKETLAB_PHASE={env_phase!r} does not match command phase {phase!r}.")
    resolved_phase = env_phase if allow_env_phase and env_phase != "" else phase
    if not any(values.values()) and env_phase == "":
        return None
    missing = [field_name for field_name, value in values.items() if value == ""]
    if missing:
        parser.error(
            "Hosted paper execution metadata is incomplete; missing: "
            + ", ".join(missing)
        )
    try:
        return PaperHostedExecutionContext.from_metadata(
            {
                **values,
                "phase": resolved_phase,
            }
        )
    except ValueError as exc:
        parser.error(str(exc))
    raise AssertionError("parser.error should exit")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marketlab")
    parser.add_argument("--version", action="version", version=f"marketlab {get_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name in ("prepare-data", "backtest", "run-experiment", "train-models"):
        command = subparsers.add_parser(command_name)
        command.add_argument("--config", required=True)

    paper_decision = subparsers.add_parser("paper-decision")
    paper_decision.add_argument("--config", required=True)
    _add_hosted_execution_arguments(paper_decision)

    paper_submit = subparsers.add_parser("paper-submit")
    paper_submit.add_argument("--config", required=True)
    _add_hosted_execution_arguments(paper_submit)

    paper_approve = subparsers.add_parser("paper-approve")
    paper_approve.add_argument("--config", required=True)
    paper_approve.add_argument("--proposal-id", required=True)
    paper_approve.add_argument("--decision", required=True, choices=("approve", "reject"))
    paper_approve.add_argument("--actor", required=True, choices=("agent", "manual"))
    _add_hosted_execution_arguments(paper_approve)

    paper_status = subparsers.add_parser("paper-status")
    paper_status.add_argument("--config", required=True)

    paper_agent_approve = subparsers.add_parser("paper-agent-approve")
    paper_agent_approve.add_argument("--config", required=True)
    paper_agent_approve.add_argument("--once", action="store_true")
    _add_hosted_execution_arguments(paper_agent_approve)

    paper_scheduler = subparsers.add_parser("paper-scheduler")
    paper_scheduler.add_argument("--config", required=True)
    paper_scheduler.add_argument("--once", action="store_true")
    _add_hosted_execution_arguments(paper_scheduler)

    paper_report = subparsers.add_parser("paper-report")
    paper_report.add_argument("--config", required=True)
    paper_report.add_argument("--start", required=True)
    paper_report.add_argument("--end", required=True)

    paper_db_migrate = subparsers.add_parser("paper-db-migrate")
    paper_db_migrate.add_argument("--config", required=True)

    paper_outbox_deliver = subparsers.add_parser("paper-outbox-deliver")
    paper_outbox_deliver.add_argument("--config", required=True)
    paper_outbox_deliver.add_argument("--limit", type=int, default=100)

    paper_notifications_deliver = subparsers.add_parser("paper-notifications-deliver")
    paper_notifications_deliver.add_argument("--config", required=True)
    paper_notifications_deliver.add_argument("--limit", type=int, default=100)

    paper_blob_sync = subparsers.add_parser("paper-blob-sync")
    paper_blob_sync.add_argument("--config", required=True)

    paper_service_bus_receive = subparsers.add_parser("paper-service-bus-receive")
    paper_service_bus_receive.add_argument("--config", required=True)
    paper_service_bus_receive.add_argument("--max-messages", type=int, default=10)
    paper_service_bus_receive.add_argument("--max-wait-seconds", type=float, default=5.0)

    phase9_shadow_decision = subparsers.add_parser("phase9-shadow-decision")
    phase9_shadow_decision.add_argument("--config", required=True)
    phase9_shadow_decision.add_argument("--evaluation", required=True)
    phase9_shadow_decision.add_argument("--panel")
    phase9_shadow_decision.add_argument("--as-of")

    phase9_shadow_scheduler = subparsers.add_parser("phase9-shadow-scheduler")
    phase9_shadow_scheduler.add_argument("--config", required=True)
    phase9_shadow_scheduler.add_argument("--once", action="store_true")
    phase9_shadow_scheduler.add_argument("--as-of")

    phase9_shadow_status = subparsers.add_parser("phase9-shadow-status")
    phase9_shadow_status.add_argument("--config", required=True)
    phase9_shadow_status.add_argument("--as-of")

    phase9_shadow_report = subparsers.add_parser("phase9-shadow-report")
    phase9_shadow_report.add_argument("--config", required=True)
    phase9_shadow_report.add_argument("--as-of")

    subparsers.add_parser("list-configs")

    write_config = subparsers.add_parser("write-config")
    write_config.add_argument("--name", required=True, choices=CONFIG_TEMPLATE_NAMES)
    write_config.add_argument("--output", required=True)
    write_config.add_argument("--force", action="store_true")

    phase8_summary = subparsers.add_parser("phase8-summary")
    phase8_summary.add_argument("--run-dir", required=True)
    phase8_summary.add_argument("--output")

    phase8_selection_probe = subparsers.add_parser("phase8-selection-probe")
    phase8_selection_probe.add_argument("--run-dir", required=True)
    phase8_selection_probe.add_argument("--output-dir")

    phase8_bull_participation = subparsers.add_parser("phase8-bull-participation")
    phase8_bull_participation.add_argument("--run-dir", required=True)
    phase8_bull_participation.add_argument("--config")
    phase8_bull_participation.add_argument("--output-dir")

    phase8_score_diagnostic = subparsers.add_parser("phase8-score-diagnostic")
    phase8_score_diagnostic.add_argument("--run-dir", required=True)
    phase8_score_diagnostic.add_argument("--output-dir")

    phase8_target_diagnostic = subparsers.add_parser("phase8-target-diagnostic")
    phase8_target_diagnostic.add_argument("--run-dir", required=True)
    phase8_target_diagnostic.add_argument("--config")
    phase8_target_diagnostic.add_argument("--output-dir")

    phase8_target_profile_sweep = subparsers.add_parser("phase8-target-profile-sweep")
    phase8_target_profile_sweep.add_argument("--config", required=True)
    phase8_target_profile_sweep.add_argument("--output")

    phase8_bull_counterfactual = subparsers.add_parser("phase8-bull-counterfactual")
    phase8_bull_counterfactual.add_argument("--run-dir", required=True)
    phase8_bull_counterfactual.add_argument("--config", required=True)
    phase8_bull_counterfactual.add_argument("--output-dir")

    phase8_regime_policy_sweep = subparsers.add_parser("phase8-regime-policy-sweep")
    phase8_regime_policy_sweep.add_argument("--run-dir", required=True)
    phase8_regime_policy_sweep.add_argument("--config", required=True)
    phase8_regime_policy_sweep.add_argument("--output-dir")

    phase8_methodology_review = subparsers.add_parser("phase8-methodology-review")
    phase8_methodology_review.add_argument("--run-dir", required=True)
    phase8_methodology_review.add_argument("--output")

    phase8_grid_compare = subparsers.add_parser("phase8-grid-compare")
    phase8_grid_compare.add_argument("--runs-root", default="artifacts/runs")
    phase8_grid_compare.add_argument("--run-dir", action="append")
    phase8_grid_compare.add_argument("--output")
    phase8_grid_compare.add_argument("--experiment-prefix", default="btc_phase8")

    return parser


def _run_logged_paper_command(
    command_name: str,
    *,
    action: Callable[[ExecutionContext], tuple[int, str | None]],
    proposal_id: str | None = None,
    hosted_context: PaperHostedExecutionContext | None = None,
) -> int:
    root_context = root_execution_context(
        deployment=hosted_context.deployment_id if hosted_context is not None else "local_cli",
        phase=command_name,
        proposal_id=proposal_id,
        details=hosted_execution_details(hosted_context, {"command": command_name}),
        execution_id=hosted_context.execution_id if hosted_context is not None else None,
        correlation_id=hosted_context.correlation_id if hosted_context is not None else None,
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
                details=hosted_execution_details(hosted_context, {"command": command_name}),
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
            details=hosted_execution_details(hosted_context, {"command": command_name}),
        ),
    )
    return exit_code


def _run_paper_decision_command(
    config,
    *,
    execution_context: ExecutionContext,
    hosted_context: PaperHostedExecutionContext | None = None,
) -> tuple[int, str | None]:
    result = run_paper_decision(
        config,
        execution_context=execution_context,
        hosted_context=hosted_context,
    )
    print(result.get("proposal_path", result["status_path"]))
    return 0, str(result.get("status", {}).get("status", "success"))


def _run_paper_submit_command(
    config,
    *,
    execution_context: ExecutionContext,
    hosted_context: PaperHostedExecutionContext | None = None,
) -> tuple[int, str | None]:
    result = run_paper_submit(
        config,
        execution_context=execution_context,
        hosted_context=hosted_context,
    )
    print(result.get("submission_path", result["status_path"]))
    return 0, str(result.get("status", {}).get("status", "success"))


def _run_paper_approve_command(
    config,
    *,
    proposal_id: str,
    decision: str,
    actor: str,
    execution_context: ExecutionContext,
    hosted_context: PaperHostedExecutionContext | None = None,
) -> tuple[int, str | None]:
    result = decide_paper_proposal(
        config,
        proposal_id=proposal_id,
        decision=decision,
        actor=actor,
        execution_context=execution_context,
        hosted_context=hosted_context,
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
    hosted_context: PaperHostedExecutionContext | None = None,
) -> tuple[int, str | None]:
    del execution_context
    run_agent_approval_loop(config, once=once, hosted_context=hosted_context)
    return 0, "success"


def _run_paper_scheduler_command(
    config,
    *,
    once: bool,
    execution_context: ExecutionContext,
    hosted_context: PaperHostedExecutionContext | None = None,
) -> tuple[int, str | None]:
    del execution_context
    run_scheduler_loop(config, once=once, hosted_context=hosted_context)
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


def _run_paper_outbox_delivery_command(
    config,
    *,
    limit: int,
    execution_context: ExecutionContext,
) -> tuple[int, str | None]:
    del execution_context
    publisher = build_paper_outbox_publisher(config)
    if publisher is None:
        raise ValueError(
            "paper-outbox-deliver requires a configured paper.azure.service_bus_backend."
        )
    result = deliver_pending_paper_outbox(
        uow_factory=build_paper_uow_factory(config),
        publisher=publisher,
        limit=limit,
        event_types=frozenset((PAPER_APPROVAL_REQUEST_EVENT_TYPE,)),
    )
    print(
        json.dumps(
            {
                "delivered_message_ids": result.delivered_message_ids,
                "failed_message_ids": result.failed_message_ids,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return (1 if result.failed_message_ids else 0), (
        "failed" if result.failed_message_ids else "success"
    )


def _run_paper_notification_delivery_command(
    config,
    *,
    limit: int,
    execution_context: ExecutionContext,
) -> tuple[int, str | None]:
    del execution_context
    result = deliver_pending_paper_notifications(
        uow_factory=build_paper_uow_factory(config),
        sink=build_telegram_paper_notification_sink(config),
        limit=limit,
    )
    print(
        json.dumps(
            {
                "event_type": PAPER_NOTIFICATION_EVENT_TYPE,
                "delivered_message_ids": result.delivered_message_ids,
                "failed_message_ids": result.failed_message_ids,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return (1 if result.failed_message_ids else 0), (
        "failed" if result.failed_message_ids else "success"
    )


def _run_paper_blob_sync_command(
    config,
    *,
    execution_context: ExecutionContext,
) -> tuple[int, str | None]:
    del execution_context
    writes = sync_paper_review_artifacts(config)
    print(
        json.dumps(
            {
                "artifact_count": len(writes),
                "blob_uris": [write.blob_uri for write in writes],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0, "success"


def _run_paper_service_bus_receive_command(
    config,
    *,
    max_messages: int,
    max_wait_seconds: float,
    execution_context: ExecutionContext,
) -> tuple[int, str | None]:
    del execution_context
    result = receive_paper_approval_requests(
        config,
        max_messages=max_messages,
        max_wait_seconds=max_wait_seconds,
    )
    print(
        json.dumps(
            {
                "completed_message_ids": result.completed_message_ids,
                "abandoned_message_ids": result.abandoned_message_ids,
                "failure_messages": result.failure_messages,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return (1 if result.failure_messages else 0), (
        "failed" if result.failure_messages else "success"
    )


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_args)

    if args.command in SHADOW_COMMANDS:
        shadow_main = getattr(shadow_cli, SHADOW_COMMANDS[args.command])
        return shadow_main(raw_args[1:])

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

    if args.command == "phase8-summary":
        print(write_phase8_run_summary(args.run_dir, output_path=args.output))
        return 0

    if args.command == "phase8-selection-probe":
        selections_path, summary_path = write_phase8_selection_probe(
            args.run_dir,
            output_dir=args.output_dir,
        )
        print(selections_path)
        print(summary_path)
        return 0

    if args.command == "phase8-bull-participation":
        detail_path, summary_path = write_phase8_bull_participation(
            args.run_dir,
            config_path=args.config,
            output_dir=args.output_dir,
        )
        print(detail_path)
        print(summary_path)
        return 0

    if args.command == "phase8-score-diagnostic":
        detail_path, summary_path = write_phase8_score_diagnostic(
            args.run_dir,
            output_dir=args.output_dir,
        )
        print(detail_path)
        print(summary_path)
        return 0

    if args.command == "phase8-target-diagnostic":
        detail_path, summary_path = write_phase8_target_diagnostic(
            args.run_dir,
            config_path=args.config,
            output_dir=args.output_dir,
        )
        print(detail_path)
        print(summary_path)
        return 0

    if args.command == "phase8-target-profile-sweep":
        load_env_file()
        config = load_config(args.config)
        panel, _ = prepare_data(config)
        modeling_dataset = build_modeling_dataset(panel, config)
        sweep_output_path = (
            args.output
            if args.output is not None
            else "artifacts/runs/phase8_btc_target_profile_sweep.csv"
        )
        print(
            write_phase8_target_profile_sweep(
                modeling_dataset,
                config=config,
                output_path=sweep_output_path,
            )
        )
        return 0

    if args.command == "phase8-bull-counterfactual":
        detail_path, summary_path, gate_path = write_phase8_bull_counterfactual(
            args.run_dir,
            config_path=args.config,
            output_dir=args.output_dir,
        )
        print(detail_path)
        print(summary_path)
        print(gate_path)
        return 0

    if args.command == "phase8-regime-policy-sweep":
        detail_path, summary_path = write_phase8_regime_policy_sweep(
            args.run_dir,
            config_path=args.config,
            output_dir=args.output_dir,
        )
        print(detail_path)
        print(summary_path)
        return 0

    if args.command == "phase8-methodology-review":
        print(write_phase8_methodology_review(args.run_dir, output_path=args.output))
        return 0

    if args.command == "phase8-grid-compare":
        print(
            write_phase8_grid_comparison(
                runs_root=args.runs_root,
                run_dirs=args.run_dir,
                output_path=args.output,
                experiment_prefix=args.experiment_prefix,
            )
        )
        return 0

    load_env_file()
    hosted_context = None
    if args.command in HOSTED_COMMAND_PHASES:
        hosted_context = _resolve_hosted_execution_context(
            args,
            parser,
            phase=HOSTED_COMMAND_PHASES[args.command],
            allow_env_phase=args.command == "paper-scheduler",
        )
    config = load_config(args.config)

    if args.command == "paper-db-migrate":
        try:
            schema_version = migrate_paper_postgres_database(config)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Applied paper database schema version: {schema_version}")
        return 0

    if args.command == "paper-outbox-deliver":
        return _run_logged_paper_command(
            args.command,
            action=lambda execution_context: _run_paper_outbox_delivery_command(
                config,
                limit=args.limit,
                execution_context=execution_context,
            ),
        )

    if args.command == "paper-notifications-deliver":
        return _run_logged_paper_command(
            args.command,
            action=lambda execution_context: _run_paper_notification_delivery_command(
                config,
                limit=args.limit,
                execution_context=execution_context,
            ),
        )

    if args.command == "paper-blob-sync":
        return _run_logged_paper_command(
            args.command,
            action=lambda execution_context: _run_paper_blob_sync_command(
                config,
                execution_context=execution_context,
            ),
        )

    if args.command == "paper-service-bus-receive":
        return _run_logged_paper_command(
            args.command,
            action=lambda execution_context: _run_paper_service_bus_receive_command(
                config,
                max_messages=args.max_messages,
                max_wait_seconds=args.max_wait_seconds,
                execution_context=execution_context,
            ),
        )

    if args.command == "paper-decision":
        return _run_logged_paper_command(
            args.command,
            hosted_context=hosted_context,
            action=lambda execution_context: _run_paper_decision_command(
                config,
                execution_context=execution_context,
                hosted_context=hosted_context,
            ),
        )

    if args.command == "paper-submit":
        return _run_logged_paper_command(
            args.command,
            hosted_context=hosted_context,
            action=lambda execution_context: _run_paper_submit_command(
                config,
                execution_context=execution_context,
                hosted_context=hosted_context,
            ),
        )

    if args.command == "paper-approve":
        return _run_logged_paper_command(
            args.command,
            proposal_id=args.proposal_id,
            hosted_context=hosted_context,
            action=lambda execution_context: _run_paper_approve_command(
                config,
                proposal_id=args.proposal_id,
                decision=args.decision,
                actor=args.actor,
                execution_context=execution_context,
                hosted_context=hosted_context,
            ),
        )

    if args.command == "paper-status":
        return _run_logged_paper_command(
            args.command,
            hosted_context=hosted_context,
            action=lambda execution_context: _run_paper_status_command(
                config,
                execution_context=execution_context,
            ),
        )

    if args.command == "paper-agent-approve":
        return _run_logged_paper_command(
            args.command,
            hosted_context=hosted_context,
            action=lambda execution_context: _run_paper_agent_approve_command(
                config,
                once=args.once,
                execution_context=execution_context,
                hosted_context=hosted_context,
            ),
        )

    if args.command == "paper-scheduler":
        return _run_logged_paper_command(
            args.command,
            hosted_context=hosted_context,
            action=lambda execution_context: _run_paper_scheduler_command(
                config,
                once=args.once,
                execution_context=execution_context,
                hosted_context=hosted_context,
            ),
        )

    if args.command == "paper-report":
        return _run_logged_paper_command(
            args.command,
            hosted_context=hosted_context,
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
        backtest_artifacts = backtest(config)
        print(backtest_artifacts.run_dir)
        return 0

    if args.command == "run-experiment":
        experiment_artifacts = run_experiment(config)
        print(experiment_artifacts.run_dir)
        return 0

    if args.command == "train-models":
        training_artifacts = train_models(config)
        print(training_artifacts.run_dir)
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
