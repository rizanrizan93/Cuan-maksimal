-- EMIR Block IDX database-first acceptance checks.
-- Run with a privileged backend connection after migration v30 and backfill.

with required(name) as (
  values
  ('cak_idx_endpoint_catalog'),('cak_idx_ingestion_runs'),('cak_idx_payload_manifest'),
  ('cak_idx_ingestion_failures'),('cak_idx_market_daily'),('cak_idx_index_daily'),
  ('cak_idx_broker_market_daily'),('cak_idx_events'),('cak_idx_company_snapshot'),
  ('cak_idx_fundamental_snapshot'),('cak_idx_rank_daily'),('cak_idx_top3_execution')
)
select r.name,
       to_regclass('public.'||r.name) is not null as present
from required r order by r.name;

select route_state,enabled,count(*) endpoints
from public.cak_idx_endpoint_catalog
group by route_state,enabled order by route_state,enabled;

select min(trade_date) first_session,max(trade_date) latest_session,
       count(distinct trade_date) sessions,count(*) rows,
       count(distinct ticker) tickers
from public.cak_idx_market_daily
where source_verified;

select endpoint_key,min(target_date) first_date,max(target_date) latest_date,
       count(*) captures,sum(rows_accepted) accepted
from public.cak_idx_payload_manifest
group by endpoint_key order by endpoint_key;

with latest as (select max(rank_date) d from public.cak_idx_rank_daily)
select r.rank_date,count(*) ranked,
       count(*) filter(where execution_eligible) execution_ready,
       min(overall_rank) min_rank,max(overall_rank) max_rank,
       count(*) filter(where active_suspension and execution_eligible) suspended_bypass,
       count(*) filter(where recent_dilution and execution_eligible) dilution_bypass,
       count(*) filter(where coalesce(rr_tp1,0)<1.8 and execution_eligible) rr_bypass
from public.cak_idx_rank_daily r,latest
where r.rank_date=latest.d group by r.rank_date;

select * from public.cak_load_latest_idx_top3_v1();

select grantee,table_name,privilege_type
from information_schema.role_table_grants
where table_schema='public' and table_name like 'cak_idx_%'
  and grantee in ('anon','authenticated')
order by table_name,grantee,privilege_type;
