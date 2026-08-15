# Emir Scanner v1.9.21 — Terminal Maintenance Latency

- Detaches the synchronous free-tier housekeeping trigger from `cak_scan_jobs`.
- Commits `COMPLETED`/partial terminal state before any outcome or retention maintenance.
- Runs bounded outcome resolution and seeding as best-effort RPC calls after the terminal checkpoint.
- Preserves existing Python-side scan-history pruning without allowing maintenance failure to invalidate ranking results.
- Includes the v1.9.20 runtime dependency reload fix already present on `main`.
- Adds regression tests for Supabase `57014` statement-timeout behavior and terminal-operation ordering.

Apply `database/migration_v11_terminal_maintenance_latency.sql` to the Emir Supabase project before resuming an affected `FINALIZE` checkpoint, then run `database/verify_v11_terminal_maintenance_latency.sql`.
