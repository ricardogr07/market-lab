from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tests._paper_fakes import build_phase7_paper_config

import marketlab.paper.persistence as persistence
from marketlab.config import PaperAzureConfig
from marketlab.paper.persistence.blob import AzureBlobPaperArtifactStore
from marketlab.paper.persistence.filesystem import build_filesystem_paper_artifact_store
from marketlab.paper.state import PaperStateStore


@dataclass
class _RecordingBlobContainer:
    uploads: list[dict[str, object]] = field(default_factory=list)

    def upload_blob(
        self,
        name: str,
        data: bytes,
        *,
        overwrite: bool,
    ) -> None:
        self.uploads.append(
            {
                "name": name,
                "data": data,
                "overwrite": overwrite,
            }
        )


def _azure_blob_config(tmp_path: Path):
    config = build_phase7_paper_config(tmp_path, symbol="QQQ")
    config.paper.azure = PaperAzureConfig(
        artifact_backend="azure_blob",
        blob_account_url="https://marketlabartifacts.blob.core.windows.net",
        blob_container_name="qqq-paper-artifacts",
        artifact_environment="uat",
        artifact_deployment_id="qqq-paper-uat",
    )
    return config


def test_azure_blob_artifact_store_preserves_filesystem_json_meanings(tmp_path: Path) -> None:
    config = _azure_blob_config(tmp_path)
    container = _RecordingBlobContainer()
    azure_store = AzureBlobPaperArtifactStore(config, container_client=container)
    filesystem_store = build_filesystem_paper_artifact_store(config)
    trade_date = "2026-04-13"
    account_snapshot = {"equity": "1000.00", "cash": "500.00"}
    order_preview = {"proposal_id": "proposal-1", "side": "buy", "notional": 500.0}

    filesystem_account_path = filesystem_store.write_trade_account_snapshot(
        trade_date=trade_date,
        payload=account_snapshot,
    )
    filesystem_preview_path = filesystem_store.write_trade_order_preview(
        trade_date=trade_date,
        payload=order_preview,
    )
    account_uri = azure_store.write_trade_account_snapshot(
        trade_date=trade_date,
        payload=account_snapshot,
    )
    preview_uri = azure_store.write_trade_order_preview(
        trade_date=trade_date,
        payload=order_preview,
    )

    assert account_uri == (
        "https://marketlabartifacts.blob.core.windows.net/qqq-paper-artifacts/"
        "paper/uat/qqq-paper-uat/trades/2026-04-13/account_snapshot.json"
    )
    assert preview_uri == (
        "https://marketlabartifacts.blob.core.windows.net/qqq-paper-artifacts/"
        "paper/uat/qqq-paper-uat/trades/2026-04-13/order_preview.json"
    )
    assert [upload["name"] for upload in container.uploads] == [
        "paper/uat/qqq-paper-uat/trades/2026-04-13/account_snapshot.json",
        "paper/uat/qqq-paper-uat/trades/2026-04-13/order_preview.json",
    ]
    assert [upload["overwrite"] for upload in container.uploads] == [True, True]
    assert container.uploads[0]["data"] == filesystem_account_path.read_bytes()
    assert container.uploads[1]["data"] == filesystem_preview_path.read_bytes()


def test_azure_blob_artifact_store_rejects_invalid_trade_date(tmp_path: Path) -> None:
    container = _RecordingBlobContainer()
    store = AzureBlobPaperArtifactStore(
        _azure_blob_config(tmp_path),
        container_client=container,
    )

    with pytest.raises(ValueError, match="ISO-8601 date"):
        store.write_trade_order_preview(
            trade_date="2026/04/13",
            payload={"proposal_id": "proposal-1"},
        )

    assert container.uploads == []


def test_artifact_store_selector_uses_azure_blob_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _azure_blob_config(tmp_path)
    sentinel = object()
    calls: list[object] = []

    def _build(received_config):
        calls.append(received_config)
        return sentinel

    monkeypatch.setattr(persistence, "build_azure_blob_paper_artifact_store", _build)

    assert persistence.build_paper_artifact_store(config) is sentinel
    assert calls == [config]


def test_azure_blob_artifact_store_syncs_full_local_review_surface(tmp_path: Path) -> None:
    config = _azure_blob_config(tmp_path)
    store = PaperStateStore(config)
    trade_date = "2026-04-13"
    trade_payloads = {
        "proposal.json": {"proposal_id": "proposal-1"},
        "evidence.json": {"proposal_id": "proposal-1", "models": []},
        "approval.json": {"proposal_id": "proposal-1", "approval_status": "approved"},
        "submission.json": {"proposal_id": "proposal-1", "status": "submitted"},
        "order_status.json": {"id": "order-1", "status": "filled"},
        "order_preview.json": {"proposal_id": "proposal-1", "side": "buy"},
        "account_snapshot.json": {"equity": "1000.00"},
    }
    for filename, payload in trade_payloads.items():
        (store.trade_dir(trade_date) / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    store.write_status({"event": "paper-submit", "status": "submitted"})
    store.write_notification_record(
        stage="paper-submit",
        outcome="submitted",
        payload={"delivery_status": "delivered"},
    )
    report_path = store.report_dir("2026-04-01", "2026-04-30") / "paper_report.md"
    report_path.write_text("# Paper report\n", encoding="utf-8")
    container = _RecordingBlobContainer()
    blob_store = AzureBlobPaperArtifactStore(config, container_client=container)

    writes = blob_store.sync_review_artifacts()

    expected_source_paths = {
        store.trade_dir(trade_date) / filename for filename in trade_payloads
    } | {
        store.status_path,
        report_path,
        *store.notifications_root.glob("*.json"),
    }
    assert {write.source_path for write in writes} == expected_source_paths
    uploaded_by_name = {str(upload["name"]): upload["data"] for upload in container.uploads}
    for write in writes:
        expected_name = "paper/uat/qqq-paper-uat/" + write.source_path.relative_to(
            store.state_root.parent
        ).as_posix()
        if write.source_path == report_path:
            expected_name = "paper/uat/qqq-paper-uat/reports/2026-04-01_2026-04-30/paper_report.md"
        elif write.source_path == store.status_path:
            expected_name = "paper/uat/qqq-paper-uat/state/status.json"
        elif write.source_path.is_relative_to(store.trades_root):
            expected_name = "paper/uat/qqq-paper-uat/trades/" + write.source_path.relative_to(
                store.trades_root
            ).as_posix()
        elif write.source_path.is_relative_to(store.notifications_root):
            expected_name = "paper/uat/qqq-paper-uat/notifications/" + write.source_path.relative_to(
                store.notifications_root
            ).as_posix()
        assert uploaded_by_name[expected_name] == write.source_path.read_bytes()
