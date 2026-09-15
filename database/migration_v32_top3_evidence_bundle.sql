-- One database-only payload for execution: score, company, ownership and risk events.
update public.cak_idx_endpoint_catalog
set enabled=true,route_state='VERIFIED',updated_at=clock_timestamp()
where endpoint_key='company_announcements'
  and path='/primary/ListedCompany/GetProfileAnnouncement';

create or replace function public.cak_load_latest_idx_top3_v3()
returns jsonb
language sql
stable
security definer
set search_path = pg_catalog, public
as $$
  with latest as (
    select max(rank_date) rank_date from public.cak_idx_top3_execution
  )
  select coalesce(jsonb_agg(
    to_jsonb(t) || jsonb_build_object(
      'company_name',c.company_name,
      'sector',c.sector,
      'subsector',c.subsector,
      'ownership',coalesce(o.ownership,'{}'::jsonb),
      'recent_events',coalesce(e.events,'[]'::jsonb)
    ) order by t.execution_rank
  ),'[]'::jsonb)
  from public.cak_idx_top3_execution t
  join latest l on l.rank_date=t.rank_date
  left join lateral (
    select x.company_name,x.sector,x.subsector
    from public.cak_idx_company_snapshot x
    where x.ticker=t.ticker and x.source_verified
    order by x.observed_on desc,x.ingested_at desc limit 1
  ) c on true
  left join lateral (
    select jsonb_strip_nulls(jsonb_build_object(
      'observed_on',x.observed_on,
      'controller_pct',x.controller_pct,
      'public_pct',x.public_pct,
      'treasury_pct',x.treasury_pct,
      'holder_count',x.holder_count,
      'source_url',x.source_url,
      'source_verified',x.source_verified
    )) ownership
    from public.cak_idx_ownership_snapshot x
    where x.ticker=t.ticker and x.source_verified
    order by x.observed_on desc,x.ingested_at desc limit 1
  ) o on true
  left join lateral (
    select jsonb_agg(jsonb_strip_nulls(jsonb_build_object(
      'event_family',z.event_family,'event_type',z.event_type,
      'event_date',z.event_date,'publication_date',z.publication_date,
      'title',z.title,'source_url',z.source_url
    )) order by coalesce(z.publication_date,z.event_date) desc) events
    from (
      select x.event_family,x.event_type,x.event_date,x.publication_date,x.title,x.source_url
      from public.cak_idx_events x
      where x.ticker=t.ticker and x.source_verified
        and coalesce(x.publication_date,x.event_date)>=t.rank_date-30
      order by coalesce(x.publication_date,x.event_date) desc limit 10
    ) z
  ) e on true;
$$;

create or replace function public.cak_idx_database_health_v2()
returns jsonb
language sql
stable
security definer
set search_path = pg_catalog, public
as $$
  select jsonb_build_object(
    'state',case when pg_database_size(current_database())>=492830720 then 'HARD_STOP'
                 when pg_database_size(current_database())>=440401920 then 'WARNING' else 'NORMAL' end,
    'database_bytes',pg_database_size(current_database()),'quota_bytes',524288000,
    'latest_market_date',(select max(trade_date) from public.cak_idx_market_daily where source_verified),
    'latest_rank_date',(select max(rank_date) from public.cak_idx_rank_daily),
    'market_rows',(select count(*) from public.cak_idx_market_daily),
    'market_tickers',(select count(distinct ticker) from public.cak_idx_market_daily),
    'fundamental_tickers',(select count(distinct ticker) from public.cak_idx_fundamental_snapshot where source_verified),
    'shareholder_tickers',(select count(distinct ticker) from (
      select ticker from public.cak_idx_ownership_snapshot where source_verified
      union select ticker from public.cak_idx_shareholder_snapshot where source_verified
    ) s),
    'financial_catalog_tickers',(select count(distinct ticker) from public.cak_idx_financial_filing_catalog where source_verified),
    'verified_endpoint_count',(select count(*) from public.cak_idx_endpoint_catalog where enabled and route_state='VERIFIED'),
    'latest_top3_count',(select count(*) from public.cak_idx_top3_execution
      where rank_date=(select max(rank_date) from public.cak_idx_top3_execution)),
    'data_mode','DATABASE_ONLY','feature_contract','EMIR_BLOCK_IDX_FEATURE_V2')
$$;

revoke all on function public.cak_load_latest_idx_top3_v3() from public,anon,authenticated;
revoke all on function public.cak_idx_database_health_v2() from public,anon,authenticated;
grant execute on function public.cak_load_latest_idx_top3_v3() to service_role;
grant execute on function public.cak_idx_database_health_v2() to service_role;

comment on function public.cak_load_latest_idx_top3_v3() is
  'EMIR database-only Top 3 bundle with official company, ownership and recent risk/catalyst evidence.';
