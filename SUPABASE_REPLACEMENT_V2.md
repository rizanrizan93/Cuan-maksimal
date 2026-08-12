# Supabase Replacement — Emir Scanner v1.9.15

## Current production target
- Organization: `rizanrizan93's Org` (`ihbbipsefiyzguterpnh`)
- Project: `Idx emir framework v2`
- Ref: `vbtpwpmkfxzqeuvztcmz`
- Region: Singapore (`ap-southeast-1`)
- API URL: `https://vbtpwpmkfxzqeuvztcmz.supabase.co`
- Plan: Free

## Old project
- `Idx emir framework` (`utgrknbmtmhpjurvcabg`)
- Region: Tokyo (`ap-northeast-1`)
- Status: paused/recovery source; do not delete until the replacement is proven in production.
- Failure mode observed: PostgreSQL WAL recovery loop with `No space left on device`.

## Installed database contract
Fresh project received schema migrations v1 through v8 plus the v8 persistence-integrity hardening.

Runtime tables:
- `cak_scan_runs`
- `cak_radar_snapshots`
- `cak_narrative_events`
- `cak_provider_audit`
- `cak_direct_evidence`
- `cak_autonomous_evidence`
- `cak_outcome_memory`
- `cak_ohlcv_cache`
- `cak_source_cache`
- `cak_scan_jobs`
- `cak_scan_job_chunks`
- `cak_research_memory`

All scanner tables have RLS enabled. `anon` and `authenticated` table privileges are revoked. The application is intended to use a Supabase backend secret/service-role key only.

## Free-tier storage policy
v1.9.15 keeps only bounded scan/job history, compacts research memory, semantically deduplicates evidence, and preserves current OHLCV/source/MARKET_FEATURES caches.

## Streamlit cutover
Update Streamlit Secrets with:

```toml
CAK_DATABASE_ENABLED = "true"
CAK_DATABASE_SCHEMA = "public"
CAK_FREE_TIER_STORAGE_MODE = "1"
SUPABASE_URL = "https://vbtpwpmkfxzqeuvztcmz.supabase.co"
SUPABASE_SECRET_KEY = "<replacement project's sb_secret_* backend key>"
```

Do not use the publishable/anon key. `persistence.py` intentionally rejects it for database writes.

After cutover, run one cold 400-ticker scan, then a second scan on the same completed IDX session to confirm MARKET_FEATURES/OHLCV cache reuse and bounded persistence.
