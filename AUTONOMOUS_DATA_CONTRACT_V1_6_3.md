# Autonomous Data Contract — v1.6.3

## Input

Required:

```text
CSV with ticker column
```

Optional direct evidence remains supported for broker inventory, narrative events, ownership/free float, bid-offer, IDX integrity, and verified outcomes.

## Progressive acquisition

```text
benchmark
→ OHLCV for entire universe
→ eligible discovery ranking
→ progressive KSEI/news/fundamental review according to selected scope
→ final Emir profile and scenario
```

`ALL_ELIGIBLE` means every ticker with valid OHLCV, not every uploaded string. Provider failures and insufficient history remain blocked.

## Persistence boundary

Source data and checkpoints:

```text
cak_ohlcv_cache
cak_source_cache
cak_scan_jobs
cak_scan_job_chunks
```

Point-in-time result data:

```text
cak_scan_runs
cak_radar_snapshots
cak_narrative_events
cak_provider_audit
cak_direct_evidence
cak_autonomous_evidence
cak_outcome_memory
```

Dashboard-derived factor fields are stored inside radar payloads. Top-3 metadata is stored in the scan job result summary. Database transfer status is observational and never substitutes for analytical evidence.

## Proxy boundary

Broker inventory and bid-offer remain OHLCV/EOD proxies unless direct verified evidence is supplied. They cannot identify beneficial owners or live queue depth.
