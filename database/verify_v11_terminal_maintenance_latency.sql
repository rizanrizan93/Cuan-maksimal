select
  case when count(*) = 0 then 'PASS' else 'FAIL' end as terminal_trigger_detached,
  count(*) as blocking_terminal_triggers
from pg_trigger
where tgrelid = 'public.cak_scan_jobs'::regclass
  and not tgisinternal
  and tgname = 'trg_cak_free_tier_housekeeping';

select
  case when count(*) = 3 then 'PASS' else 'FAIL' end as maintenance_functions_available,
  count(*) as function_count
from pg_proc p
join pg_namespace n on n.oid = p.pronamespace
where n.nspname = 'public'
  and p.proname in (
    'cak_seed_outcomes_for_scan',
    'cak_resolve_outcome_memory',
    'cak_free_tier_housekeeping'
  );
