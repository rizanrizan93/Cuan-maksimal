# Emir Scanner v1.9.15 — Free Tier Storage Safety

Storage-safety release for Supabase Free deployments. Analytical scoring and Guarded Real Money semantics are unchanged.

## Changes
- Bounded scan history: keep only the latest 2 published scan runs and 2 terminal resumable jobs.
- Durable research memory is compacted and bounded per ticker/family.
- Identical evidence uses semantic hashing so fetch timestamps do not create duplicate memory rows.
- OHLCV-derived broker/orderbook proxies remain in MARKET_FEATURES/current cache and are not duplicated into durable research memory.
- Housekeeping is best-effort and cannot abort a scan.
- Resumable JOB_VERSION remains 1.9.14-compatible so existing checkpoints are not invalidated.

## Database recovery note
The existing Emir Supabase project is currently not accepting connections after running out of disk. Do not rerun schema migrations against that outage state. Once PostgreSQL accepts connections, prune historical rows before starting a new 400-ticker scan.
