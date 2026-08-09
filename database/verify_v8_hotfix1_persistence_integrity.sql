-- Read-only verification for v1.9.8 persistence/security integrity.
select
    p.proname,
    coalesce(array_to_string(p.proconfig, ','), '') as function_config,
    has_function_privilege('anon', p.oid, 'EXECUTE') as anon_execute,
    has_function_privilege('authenticated', p.oid, 'EXECUTE') as authenticated_execute
from pg_proc p
join pg_namespace n on n.oid = p.pronamespace
where n.nspname = 'public'
  and p.proname in ('cak_touch_updated_at', 'set_scanner_updated_at', 'rls_auto_enable')
order by p.proname;

select to_regclass('public.idx_cak_narrative_events_scan_id') is not null
    as narrative_scan_index_exists;
