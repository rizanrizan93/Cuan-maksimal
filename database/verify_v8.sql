-- Read-only runtime preflight for IDX Emir Autonomous Scanner schema v8.
-- Every row must report relation_exists=true and all three service-role privileges=true.
with required(table_name) as (
    values
        ('cak_scan_runs'),
        ('cak_radar_snapshots'),
        ('cak_narrative_events'),
        ('cak_provider_audit'),
        ('cak_direct_evidence'),
        ('cak_autonomous_evidence'),
        ('cak_outcome_memory'),
        ('cak_ohlcv_cache'),
        ('cak_source_cache'),
        ('cak_scan_jobs'),
        ('cak_scan_job_chunks'),
        ('cak_research_memory')
), resolved as (
    select table_name, to_regclass(format('public.%I', table_name)) as relation
    from required
)
select
    table_name,
    relation is not null as relation_exists,
    case when relation is null then false else has_table_privilege('service_role', relation, 'SELECT') end as service_role_select,
    case when relation is null then false else has_table_privilege('service_role', relation, 'INSERT') end as service_role_insert,
    case when relation is null then false else has_table_privilege('service_role', relation, 'UPDATE') end as service_role_update
from resolved
order by table_name;

select count(*) as research_memory_rows from public.cak_research_memory;
select family, count(*) from public.cak_research_memory group by family order by count(*) desc;
