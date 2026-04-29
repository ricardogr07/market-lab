from __future__ import annotations

import json
import logging

from marketlab.log import (
    STRUCTURED_LOG_KEYS,
    bind_execution_context,
    configure_logging,
    create_execution_context,
    emit_structured_log,
)


def _stderr_records(stderr: str) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in stderr.splitlines()
        if line.strip() != ""
    ]


def test_configure_logging_emits_stable_json_payload_to_stderr(
    capsys,
) -> None:
    configure_logging()
    logger = logging.getLogger("marketlab.tests.log")
    emit_structured_log(
        logger,
        logging.INFO,
        "hello structured logs",
        event="test.log.event",
        execution_context=create_execution_context(
            phase="paper-test",
            deployment="unit_test",
            details={"alpha": 1},
        ),
    )

    captured = capsys.readouterr()
    records = _stderr_records(captured.err)

    assert captured.out == ""
    assert len(records) == 1
    assert list(records[0].keys()) == list(STRUCTURED_LOG_KEYS)
    assert records[0]["message"] == "hello structured logs"
    assert records[0]["event"] == "test.log.event"
    assert records[0]["phase"] == "paper-test"
    assert records[0]["deployment"] == "unit_test"
    assert records[0]["details"] == {"alpha": 1}
    assert records[0]["proposal_id"] is None
    assert records[0]["duration_ms"] is None


def test_bound_execution_context_is_used_when_no_explicit_context_is_passed(
    capsys,
) -> None:
    configure_logging()
    logger = logging.getLogger("marketlab.tests.bound")
    context = create_execution_context(
        phase="paper-bound",
        deployment="unit_test",
        proposal_id="proposal-123",
    )

    with bind_execution_context(context):
        emit_structured_log(
            logger,
            logging.INFO,
            "bound log",
            event="test.bound.event",
        )

    captured = capsys.readouterr()
    records = _stderr_records(captured.err)

    assert len(records) == 1
    assert records[0]["execution_id"] == context.execution_id
    assert records[0]["correlation_id"] == context.correlation_id
    assert records[0]["proposal_id"] == "proposal-123"
    assert records[0]["phase"] == "paper-bound"
