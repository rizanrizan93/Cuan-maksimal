# Emir Scanner v1.9.15 — Free Tier Storage Safety

Storage-safety release for Supabase Free deployments. Analytical scoring and Guarded Real Money semantics are unchanged.

## Changes
- Bounded scan history: keep only the latest 2 published scan runs and 2 terminal resumable jobs.
- Durable research memory is compacted and bounded per ticker/family.
- Identical evidence uses semantic hashing so fetch timestamps do not create duplicate memory rows.
- OHLCV-derived broker/orderbook proxies remain in MARKET_FEATURES/current cache and are not duplicated into durable research memory.
- Housekeeping is best-effort and cannot abort a scan.
- Resumable JOB_VERSION remains 1.9.14-compatible so existing checkpoints are not invalidated.

## Production database replacement
The original project `Idx emir framework` (`utgrknbmtmhpjurvcabg`, Tokyo) exhausted its disk and entered PostgreSQL recovery loops. It has been paused, not deleted.

Replacement production database:
- project: `Idx emir framework v2`
- project ref: `vbtpwpmkfxzqeuvztcmz`
- region: `ap-southeast-1` (Singapore)
- API URL: `https://vbtpwpmkfxzqeuvztcmz.supabase.co`
- plan: Supabase Free
- schema: Emir v8 + persistence-integrity hotfix
- RLS: enabled on all scanner tables; anon/authenticated table access revoked
- initial database size after schema install: approximately 11 MB

The Streamlit deployment must use the replacement project's backend secret/service-role key together with the URL above. Publishable/anon keys are intentionally rejected by the scanner database bridge.
