# Emir Scanner v1.9.22 — Outcome Reconciliation Integrity

- Keeps exact row-count verification for the six scan-owned result tables.
- Verifies shared `cak_outcome_memory` with an at-least-expected contract because bounded terminal maintenance can legitimately add rows after the primary result commit.
- Separates contract-verified rows from observed database rows so maintenance output remains visible without being misclassified as missing data.
- Prevents a fully persisted 400-ticker scan from being labelled partial solely because post-commit outcome seeding added rows.
- Still rejects outcome readback when the observed count is below the number of outcome rows the scan attempted to write.
- Adds regression coverage for both legitimate extra rows and genuinely missing expected rows.
