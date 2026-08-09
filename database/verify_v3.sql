-- Expected: 5 table rows and 6 index rows after migrations v1-v3.

select table_name
from information_schema.tables
where table_schema = 'public'
  and table_name in (
    'cak_scan_runs',
    'cak_radar_snapshots',
    'cak_narrative_events',
    'cak_provider_audit',
    'cak_direct_evidence'
  )
order by table_name;

select indexname
from pg_indexes
where schemaname = 'public'
  and indexname in (
    'idx_cak_radar_ticker_asof',
    'idx_cak_events_ticker_date',
    'idx_cak_provider_scan',
    'idx_cak_provider_ticker',
    'idx_cak_direct_evidence_scan',
    'idx_cak_direct_evidence_ticker_date'
  )
order by indexname;
