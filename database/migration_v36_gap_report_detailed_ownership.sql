-- Database-only deep review coverage contract for the selected Top 900 universe.
create or replace function public.cak_idx_database_gap_report_v1(p_limit integer default 900)
returns jsonb
language sql
stable
security definer
set search_path = pg_catalog, public
as $$
  with latest as (
    select max(rank_date) rank_date from public.cak_idx_rank_daily
  ), ownership as (
    select ticker from public.cak_idx_ownership_snapshot where source_verified
    union
    select ticker from public.cak_idx_shareholder_snapshot where source_verified
  ), company as (
    select distinct ticker from public.cak_idx_company_snapshot where source_verified
  ), filing as (
    select distinct ticker from public.cak_idx_financial_filing_catalog where source_verified
  ), selected as (
    select r.*,
      o.ticker is not null ownership_ready,
      c.ticker is not null company_ready,
      f.ticker is not null filing_catalog_ready
    from public.cak_idx_rank_daily r
    join latest l on l.rank_date=r.rank_date
    left join ownership o using(ticker)
    left join company c using(ticker)
    left join filing f using(ticker)
    order by r.overall_rank
    limit least(greatest(p_limit,1),900)
  ), gaps as (
    select rank_date,overall_rank,ticker,observations,data_completeness_pct,
      official_fundamental_coverage_pct,execution_eligible,blocker,
      observations>=80 market_history_ready,
      coalesce(official_fundamental_coverage_pct,0)>=50 fundamental_metrics_ready,
      ownership_ready,company_ready,filing_catalog_ready,
      array_remove(array[
        case when observations<80 then 'MARKET_HISTORY_80_SESSIONS' end,
        case when coalesce(official_fundamental_coverage_pct,0)<50 then 'OFFICIAL_FUNDAMENTAL_METRICS' end,
        case when not ownership_ready then 'OFFICIAL_OWNERSHIP' end,
        case when not company_ready then 'COMPANY_REFERENCE' end,
        case when not filing_catalog_ready then 'FINANCIAL_FILING_CATALOG' end
      ],null)::text[] missing_required,
      (observations<80 or coalesce(official_fundamental_coverage_pct,0)<50) critical_gap
    from selected
  ), event_state as (
    select max(coalesce(publication_date,event_date)) latest_event_date
    from public.cak_idx_events where source_verified
  )
  select jsonb_build_object(
    'summary',jsonb_build_object(
      'rank_date',(select rank_date from latest),
      'target_universe',900,
      'selected_tickers',(select count(*) from gaps),
      'deep_reviewed_tickers',(select count(*) from gaps),
      'fully_complete_tickers',(select count(*) from gaps where cardinality(missing_required)=0),
      'critical_gap_tickers',(select count(*) from gaps where critical_gap),
      'missing_market_history',(select count(*) from gaps where not market_history_ready),
      'missing_fundamental_metrics',(select count(*) from gaps where not fundamental_metrics_ready),
      'missing_ownership',(select count(*) from gaps where not ownership_ready),
      'missing_company_reference',(select count(*) from gaps where not company_ready),
      'missing_filing_catalog',(select count(*) from gaps where not filing_catalog_ready),
      'execution_eligible',(select count(*) from gaps where execution_eligible),
      'latest_event_date',(select latest_event_date from event_state),
      'event_feed_ready',coalesce((select latest_event_date from event_state)>=(select rank_date from latest),false),
      'data_mode','DATABASE_ONLY_DEEP_900'
    ),
    'rows',coalesce((
      select jsonb_agg(to_jsonb(x) order by x.overall_rank)
      from (
        select rank_date,overall_rank,ticker,observations,data_completeness_pct,
          official_fundamental_coverage_pct,execution_eligible,blocker,
          market_history_ready,fundamental_metrics_ready,ownership_ready,
          company_ready,filing_catalog_ready,missing_required,critical_gap,
          case when critical_gap then 'EXECUTION_BLOCKED_MISSING_CRITICAL_EVIDENCE'
               when cardinality(missing_required)>0 then 'DEEP_REVIEW_CONTEXT_INCOMPLETE'
               else 'DEEP_REVIEW_COMPLETE' end gap_effect
        from gaps where cardinality(missing_required)>0
      ) x
    ),'[]'::jsonb)
  )
$$;

revoke all on function public.cak_idx_database_gap_report_v1(integer) from public,anon,authenticated;
grant execute on function public.cak_idx_database_gap_report_v1(integer) to service_role;

comment on function public.cak_idx_database_gap_report_v1(integer) is
  'Top 900 database-only deep review coverage and per-ticker missing-evidence report.';

