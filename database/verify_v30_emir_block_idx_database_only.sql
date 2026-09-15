-- Returns zero rows when the v30 database-only contract is healthy.
with checks as (
  select 'DATABASE_SIZE_OVER_500_MIB' failure
  where pg_database_size(current_database()) >= 524288000
  union all
  select 'MARKET_WINDOW_NOT_SIX_MONTHS'
  where (select min(trade_date) from public.cak_idx_market_daily) <
        (((now() at time zone 'Asia/Jakarta')::date)-interval '6 months')::date
  union all
  select 'LATEST_MARKET_RANK_MISMATCH'
  where (select max(trade_date) from public.cak_idx_market_daily where source_verified)
        is distinct from
        (select max(rank_date) from public.cak_idx_rank_daily)
  union all
  select 'TOP3_NOT_EXACTLY_THREE'
  where (select count(*) from public.cak_idx_top3_execution
         where rank_date=(select max(rank_date) from public.cak_idx_top3_execution)) <> 3
  union all
  select 'UNVERIFIED_MARKET_ROWS'
  where exists(select 1 from public.cak_idx_market_daily where not source_verified)
  union all
  select 'PUBLIC_FUNCTION_EXECUTE_LEAK'
  where exists(
    select 1 from pg_proc p join pg_namespace n on n.oid=p.pronamespace
    where n.nspname='public' and p.proname in (
      'cak_idx_refresh_reference_v3','cak_idx_refresh_shareholders_v3',
      'cak_idx_refresh_financial_catalog_v3','cak_refresh_idx_ranking_v2',
      'cak_idx_run_eod_v3','cak_load_latest_idx_rank_v2','cak_idx_database_health_v2',
      'cak_prune_storage_v2')
      and (has_function_privilege('anon',p.oid,'execute') or
           has_function_privilege('authenticated',p.oid,'execute')))
)
select * from checks;

