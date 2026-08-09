-- Expected after migrations v1-v7: 11 scanner tables.
select table_name
from information_schema.tables
where table_schema = 'public'
  and table_name in (
    'cak_scan_runs','cak_radar_snapshots','cak_narrative_events','cak_provider_audit',
    'cak_direct_evidence','cak_outcome_memory','cak_autonomous_evidence',
    'cak_ohlcv_cache','cak_source_cache','cak_scan_jobs','cak_scan_job_chunks'
  )
order by table_name;

select
    has_table_privilege('service_role', 'public.cak_scan_jobs', 'select,insert,update,delete') as scan_jobs_permission_ok,
    has_table_privilege('service_role', 'public.cak_scan_job_chunks', 'select,insert,update,delete') as scan_job_chunks_permission_ok;
