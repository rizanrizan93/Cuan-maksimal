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


-- v31/v32 storage, scheduler and salvage acceptance.
select public.cak_capture_storage_v1() as storage_state;

select backup_name,backup_sha256,logical_dump_complete,salvage_policy,
       imported_rows,validation_state,imported_at
from public.cak_backup_salvage_manifest;

select jobname,schedule,active,command
from cron.job
where jobname in ('emir-block-idx-eod-1745-wib','emir-block-idx-eod-retry-1845-wib')
order by jobname;

select count(*) filter(where trade_date < ((now() at time zone 'Asia/Jakarta')::date-interval '6 months')::date)
         as market_rows_outside_retention,
       count(*) filter(where source_url not like 'https://block.idx.id/%')
         as market_rows_nonofficial_url,
       count(*) filter(where not source_verified)
         as market_rows_unverified
from public.cak_idx_market_daily;
