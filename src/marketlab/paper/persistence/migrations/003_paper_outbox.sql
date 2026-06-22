CREATE TABLE paper_outbox (
    message_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    payload_json JSONB NOT NULL,
    created_at TEXT NOT NULL,
    delivery_status TEXT NOT NULL CHECK (delivery_status IN ('pending', 'failed', 'delivered')),
    delivery_attempts INTEGER NOT NULL CHECK (delivery_attempts >= 0),
    delivered_at TEXT,
    last_error TEXT
);

CREATE INDEX paper_outbox_pending_idx
    ON paper_outbox (delivery_status, created_at, message_id);

CREATE INDEX paper_outbox_pending_event_idx
    ON paper_outbox (delivery_status, event_type, created_at, message_id);
