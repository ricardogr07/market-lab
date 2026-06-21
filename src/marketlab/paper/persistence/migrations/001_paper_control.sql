CREATE TABLE paper_proposals (
    proposal_id TEXT PRIMARY KEY,
    effective_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json JSONB NOT NULL
);

CREATE TABLE paper_evidence (
    trade_date TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    payload_json JSONB NOT NULL
);

CREATE TABLE paper_approvals (
    trade_date TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    payload_json JSONB NOT NULL
);

CREATE TABLE paper_submissions (
    trade_date TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    payload_json JSONB NOT NULL
);

CREATE TABLE paper_order_statuses (
    trade_date TEXT PRIMARY KEY,
    payload_json JSONB NOT NULL
);

CREATE TABLE paper_status (
    singleton_key SMALLINT PRIMARY KEY CHECK (singleton_key = 1),
    payload_json JSONB NOT NULL
);

CREATE TABLE paper_deployment_records (
    environment TEXT NOT NULL,
    deployment_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    execution_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    PRIMARY KEY (environment, deployment_id, phase, execution_id)
);

CREATE TABLE paper_phase_run_records (
    idempotency_key TEXT PRIMARY KEY,
    phase TEXT NOT NULL,
    payload_json JSONB NOT NULL
);
