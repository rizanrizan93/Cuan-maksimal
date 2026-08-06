# Release Notes — v1.6.3

## Added

- Progressive `ALL_ELIGIBLE` deep review as the default scope.
- Fast Top 30, Balanced Top 60, and Custom deep-review scopes.
- One-click **Mulai Scan** and dashboard **Scan Ulang** controls.
- Deep target/progress/scope metrics during active jobs.
- Database result-transfer reconciliation and exact verified-row metrics.
- Persisted dashboard score fields in radar payloads.
- Persisted derived Top-3 summary in `cak_scan_jobs.result_summary`.
- `dashboard_persistence.py` for testable transfer accounting.

## Changed

- KSEI/news/fundamental stages now process the full eligible target progressively when `ALL_ELIGIBLE` is selected.
- A loaded historical scan reconstructs a database verification report from paginated persisted rows.
- Optional uploaded evidence files are rewound before repeat reads.

## Database

No schema migration. Schema remains v7 with 11 tables.
