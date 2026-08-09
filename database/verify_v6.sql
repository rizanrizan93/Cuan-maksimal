-- Expected: 9 TABLE rows and 14 INDEX rows after migrations v1-v6.
select table_name
from information_schema.tables
where table_schema = 'public'
  and table_name in (
    'cak_scan_runs','cak_radar_snapshots','cak_narrative_events','cak_provider_audit',
    'cak_direct_evidence','cak_outcome_memory','cak_autonomous_evidence',
    'cak_ohlcv_cache','cak_source_cache'
  )
order by table_name;

select indexname
from pg_indexes
where schemaname = 'public'
  and tablename in ('cak_ohlcv_cache','cak_source_cache')
order by indexname;

select
    has_table_privilege('service_role', 'public.cak_ohlcv_cache', 'select,insert,update,delete') as ohlcv_cache_permission_ok,
    has_table_privilege('service_role', 'public.cak_source_cache', 'select,insert,update,delete') as source_cache_permission_ok;
