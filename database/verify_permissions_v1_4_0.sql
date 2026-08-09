-- Expected: seven rows, all permission columns true.
select
    table_name,
    has_table_privilege('service_role', format('public.%I', table_name), 'SELECT') as can_select,
    has_table_privilege('service_role', format('public.%I', table_name), 'INSERT') as can_insert,
    has_table_privilege('service_role', format('public.%I', table_name), 'UPDATE') as can_update,
    has_table_privilege('service_role', format('public.%I', table_name), 'DELETE') as can_delete,
    case
        when has_table_privilege('service_role', format('public.%I', table_name), 'SELECT')
         and has_table_privilege('service_role', format('public.%I', table_name), 'INSERT')
         and has_table_privilege('service_role', format('public.%I', table_name), 'UPDATE')
         and has_table_privilege('service_role', format('public.%I', table_name), 'DELETE')
        then 'PERMISSION_OK'
        else 'PERMISSION_MISSING'
    end as state
from (values
    ('cak_scan_runs'),
    ('cak_radar_snapshots'),
    ('cak_narrative_events'),
    ('cak_provider_audit'),
    ('cak_direct_evidence'),
    ('cak_autonomous_evidence'),
    ('cak_outcome_memory')
) as scanner_tables(table_name)
order by table_name;
