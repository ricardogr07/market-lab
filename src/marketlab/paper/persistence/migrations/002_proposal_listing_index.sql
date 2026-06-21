CREATE INDEX paper_proposals_order_idx
    ON paper_proposals (effective_date DESC, created_at DESC, proposal_id DESC);
