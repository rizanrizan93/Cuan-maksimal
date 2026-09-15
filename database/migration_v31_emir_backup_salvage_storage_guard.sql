-- EMIR v2 clean restore compatibility layer and bounded-storage policy.
-- The legacy 2026-08-26 dump was truncated during cak_research_memory COPY;
-- only validated, completed COPY blocks and compact snapshots are imported.

create table if not exists public.cak_backup_salvage_manifest (
  backup_sha256 text primary key,
  backup_name text not null,
  source_uncompressed_bytes bigint not null,
  logical_dump_complete boolean not null,
  salvage_policy text not null,
  imported_rows jsonb not null default '{}'::jsonb,
  validation_state text not null,
  imported_at timestamptz not null default now()
);

create table if not exists public.cak_autonomous_evidence (
  evidence_id text primary key, scan_id text, ticker text not null,
  evidence_type text not null, observed_at timestamptz,
  source_verified boolean not null default false,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(), updated_at timestamptz not null default now()
);
create index if not exists cak_autonomous_evidence_lookup_idx
  on public.cak_autonomous_evidence(ticker,evidence_type,observed_at desc);

create table if not exists public.cak_direct_evidence (
  evidence_id text primary key, scan_id text not null, ticker text not null,
  evidence_type text not null, observed_at timestamptz,
  source_verified boolean not null default false,
  payload jsonb not null default '{}'::jsonb, created_at timestamptz not null default now()
);
create index if not exists cak_direct_evidence_lookup_idx
  on public.cak_direct_evidence(ticker,evidence_type,observed_at desc);

create table if not exists public.cak_narrative_events (
  event_id text primary key, scan_id text not null, ticker text not null,
  published_at timestamptz, title text, publisher text, source_url text,
  payload jsonb not null default '{}'::jsonb, created_at timestamptz not null default now()
);
create index if not exists cak_narrative_events_recent_idx
  on public.cak_narrative_events(ticker,published_at desc) where published_at is not null;

create table if not exists public.cak_ohlcv_cache (
  ticker text not null, period text not null default '5y', first_session_date date,
  last_session_date date, bars integer not null default 0, provider text,
  quality_state text, checked_at timestamptz not null, last_scan_id text,
  content_sha256 text not null, payload jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now(), updated_at timestamptz not null default now(),
  primary key(ticker,period)
);

create table if not exists public.cak_outcome_memory (
  outcome_id text primary key, scan_id text, ticker text not null, signal_date date,
  horizon_days integer, outcome_verified boolean not null default false,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(), updated_at timestamptz not null default now()
);
create index if not exists cak_outcome_memory_ticker_date_idx
  on public.cak_outcome_memory(ticker,signal_date desc);

create table if not exists public.cak_provider_audit (
  audit_id text primary key, scan_id text not null, ticker text, provider text not null,
  status text, payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);
create index if not exists cak_provider_audit_recent_idx
  on public.cak_provider_audit(created_at desc);

create table if not exists public.cak_radar_snapshots (
  scan_id text not null, ticker text not null, as_of timestamptz not null,
  public_method_state text, action text, conviction_score double precision,
  coverage_pct double precision, production_ready boolean not null default false,
  payload jsonb not null default '{}'::jsonb, created_at timestamptz not null default now(),
  primary key(scan_id,ticker)
);
create index if not exists cak_radar_snapshots_recent_idx
  on public.cak_radar_snapshots(as_of desc,ticker);

create table if not exists public.cak_research_memory (
  memory_id text primary key, ticker text not null, family text not null,
  effective_period date, observed_at timestamptz, provider text, source_url text,
  source_verified boolean not null default false, official_source boolean not null default false,
  content_sha256 text not null, last_scan_id text,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(), updated_at timestamptz not null default now()
);
create index if not exists cak_research_memory_lookup_idx
  on public.cak_research_memory(ticker,family,effective_period desc,observed_at desc);

create table if not exists public.cak_scan_job_chunks (
  chunk_id text primary key, scan_id text not null, stage text not null, chunk_no integer not null,
  ticker_count integer not null default 0, processed_count integer not null default 0,
  failed_count integer not null default 0, status text not null,
  started_at timestamptz, completed_at timestamptz,
  payload jsonb not null default '{}'::jsonb, created_at timestamptz not null default now()
);
create index if not exists cak_scan_job_chunks_scan_idx
  on public.cak_scan_job_chunks(scan_id,stage,chunk_no);

create table if not exists public.cak_scan_jobs (
  scan_id text primary key, universe_hash text not null, scanner_version text not null,
  status text not null default 'CREATED', current_stage text not null default 'BENCHMARK',
  current_offset integer not null default 0, current_chunk integer not null default 0,
  chunk_size integer not null default 20, total_tickers integer not null default 0,
  processed_tickers integer not null default 0, failed_tickers integer not null default 0,
  progress_pct numeric not null default 0, scan_mode text, result_status text,
  universe jsonb not null default '[]'::jsonb, settings jsonb not null default '{}'::jsonb,
  shortlist jsonb not null default '[]'::jsonb, failures jsonb not null default '{}'::jsonb,
  result_summary jsonb not null default '{}'::jsonb, last_error text,
  heartbeat_at timestamptz, created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.cak_scan_runs (
  scan_id text primary key, as_of timestamptz not null, scanner_version text not null,
  scan_mode text, ticker_count integer not null default 0,
  production_ready_count integer not null default 0, status text not null,
  created_at timestamptz not null default now()
);
create index if not exists cak_scan_runs_as_of_idx on public.cak_scan_runs(as_of desc);

create table if not exists public.cak_source_cache (
  cache_key text primary key, ticker text not null, family text not null, provider text,
  status text, checked_at timestamptz not null, valid_until timestamptz not null,
  latest_observed_at timestamptz, last_scan_id text, content_sha256 text not null,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(), updated_at timestamptz not null default now()
);
create index if not exists cak_source_cache_lookup_idx
  on public.cak_source_cache(ticker,family,valid_until desc);

create table if not exists public.independent_data_backfill (
  ticker text primary key, as_of timestamptz not null,
  fundamental_payload jsonb not null default '{}'::jsonb,
  fundamental_history_payload jsonb not null default '[]'::jsonb,
  ohlcv_metadata_payload jsonb not null default '{}'::jsonb,
  accumulation_payload jsonb not null default '{}'::jsonb,
  provider_audit_payload jsonb not null default '{}'::jsonb,
  fundamental_coverage_pct numeric not null default 0,
  ohlcv_coverage_pct numeric not null default 0, overall_coverage_pct numeric not null default 0,
  source_state text not null default 'PUBLIC_DIRECT_NOT_OFFICIAL', content_hash text not null,
  created_at timestamptz not null default now(), updated_at timestamptz not null default now()
);

create table if not exists public.ohlcv_daily_cache (
  ticker text primary key, payload jsonb not null default '[]'::jsonb,
  bar_count integer not null default 0, first_bar_date date, last_bar_date date,
  source_family text, source_tier text, source_checked_at timestamptz not null default now(),
  refresh_state text not null default 'MISSING', last_error text, content_hash text,
  model_version text not null, schema_version text not null,
  created_at timestamptz not null default now(), updated_at timestamptz not null default now()
);

create table if not exists public.cak_storage_policy_v1 (
  policy_version text primary key,
  quota_bytes bigint not null check(quota_bytes=524288000),
  warning_bytes bigint not null check(warning_bytes<quota_bytes),
  hard_stop_bytes bigint not null check(hard_stop_bytes<=quota_bytes),
  market_retention_months integer not null check(market_retention_months=6),
  audit_retention_days integer not null check(audit_retention_days between 7 and 90),
  frozen_at timestamptz not null default now()
);
insert into public.cak_storage_policy_v1
  (policy_version,quota_bytes,warning_bytes,hard_stop_bytes,market_retention_months,audit_retention_days)
values('EMIR_STORAGE_UNDER_500MIB_V1',524288000,440401920,492830720,6,30)
on conflict(policy_version) do update set
  quota_bytes=excluded.quota_bytes,warning_bytes=excluded.warning_bytes,
  hard_stop_bytes=excluded.hard_stop_bytes,market_retention_months=excluded.market_retention_months,
  audit_retention_days=excluded.audit_retention_days;

create table if not exists public.cak_storage_measurement_v1 (
  measured_at timestamptz primary key default now(), database_bytes bigint not null,
  public_bytes bigint not null, cak_idx_bytes bigint not null,
  quota_bytes bigint not null, quota_ratio numeric not null, state text not null,
  largest_relations jsonb not null default '[]'::jsonb
);

create or replace function public.cak_capture_storage_v1()
returns jsonb language plpgsql security definer set search_path=pg_catalog,public as $fn$
declare v_db bigint; v_public bigint; v_idx bigint; v_quota bigint; v_warning bigint;
  v_hard bigint; v_state text; v_largest jsonb; v_measured_at timestamptz;
begin
  select quota_bytes,warning_bytes,hard_stop_bytes into v_quota,v_warning,v_hard
  from public.cak_storage_policy_v1 where policy_version='EMIR_STORAGE_UNDER_500MIB_V1';
  v_db:=pg_database_size(current_database());
  select coalesce(sum(pg_total_relation_size(c.oid)),0)::bigint into v_public
  from pg_class c join pg_namespace n on n.oid=c.relnamespace
  where n.nspname='public' and c.relkind in('r','m');
  select coalesce(sum(pg_total_relation_size(c.oid)),0)::bigint into v_idx
  from pg_class c join pg_namespace n on n.oid=c.relnamespace
  where n.nspname='public' and c.relkind in('r','m') and c.relname like 'cak_idx_%';
  select coalesce(jsonb_agg(x.payload order by x.total_bytes desc),'[]'::jsonb) into v_largest
  from (
    select pg_total_relation_size(c.oid) total_bytes,
      jsonb_build_object('relation',c.relname,'total_bytes',pg_total_relation_size(c.oid)) payload
    from pg_class c join pg_namespace n on n.oid=c.relnamespace
    where n.nspname='public' and c.relkind in('r','m')
    order by pg_total_relation_size(c.oid) desc limit 15
  ) x;
  v_state:=case when v_db>=v_hard then 'HARD_STOP' when v_db>=v_warning then 'WARNING' else 'NORMAL' end;
  v_measured_at:=clock_timestamp();
  insert into public.cak_storage_measurement_v1
    (measured_at,database_bytes,public_bytes,cak_idx_bytes,quota_bytes,quota_ratio,state,largest_relations)
  values(v_measured_at,v_db,v_public,v_idx,v_quota,v_db::numeric/v_quota,v_state,v_largest)
  on conflict(measured_at) do update set
    database_bytes=excluded.database_bytes,public_bytes=excluded.public_bytes,
    cak_idx_bytes=excluded.cak_idx_bytes,quota_bytes=excluded.quota_bytes,
    quota_ratio=excluded.quota_ratio,state=excluded.state,
    largest_relations=excluded.largest_relations;
  delete from public.cak_storage_measurement_v1 where measured_at<now()-interval '180 days';
  return jsonb_build_object('state',v_state,'database_bytes',v_db,'public_bytes',v_public,
    'cak_idx_bytes',v_idx,'quota_bytes',v_quota,'quota_ratio',round(v_db::numeric/v_quota,4),
    'largest_relations',v_largest);
end $fn$;

create or replace function public.cak_prune_storage_v1(
  p_as_of date default ((now() at time zone 'Asia/Jakarta')::date)
)
returns jsonb language plpgsql security definer set search_path=pg_catalog,public as $fn$
declare v_cutoff date; v_deleted bigint:=0; v_n bigint; v_result jsonb;
begin
  v_cutoff:=(p_as_of-interval '6 months')::date;
  delete from public.cak_idx_market_daily where trade_date<v_cutoff; get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_idx_index_daily where trade_date<v_cutoff; get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_idx_broker_market_daily where trade_date<v_cutoff; get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_idx_payload_manifest where target_date<v_cutoff; get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_idx_events where event_date<(p_as_of-interval '18 months')::date; get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_idx_ingestion_failures where last_seen_at<now()-interval '90 days'; get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_idx_ingestion_runs where started_at<now()-interval '90 days'; get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_idx_rank_daily where rank_date<v_cutoff; get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_idx_top3_execution where rank_date<v_cutoff; get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_idx_company_snapshot c where exists(
    select 1 from (select ticker,observed_on,row_number() over(partition by ticker order by observed_on desc) rn
      from public.cak_idx_company_snapshot) x where x.ticker=c.ticker and x.observed_on=c.observed_on and x.rn>2
  ); get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_idx_fundamental_snapshot f where exists(
    select 1 from (select ticker,period_end,observed_on,payload_hash,
      row_number() over(partition by ticker order by period_end desc,observed_on desc) rn
      from public.cak_idx_fundamental_snapshot) x
    where x.ticker=f.ticker and x.period_end=f.period_end and x.observed_on=f.observed_on
      and x.payload_hash=f.payload_hash and x.rn>8
  ); get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_provider_audit where created_at<now()-interval '30 days'; get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_radar_snapshots r where r.scan_id not in(
    select scan_id from public.cak_scan_runs order by as_of desc limit 5
  ); get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_scan_job_chunks where created_at<now()-interval '30 days'; get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_scan_jobs where created_at<now()-interval '30 days'; get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  delete from public.cak_narrative_events where coalesce(published_at,created_at)<now()-interval '6 months'; get diagnostics v_n=row_count; v_deleted:=v_deleted+v_n;
  select public.cak_capture_storage_v1() into v_result;
  return v_result||jsonb_build_object('deleted_rows',v_deleted,'market_cutoff',v_cutoff);
end $fn$;

do $fn$ declare r record; begin
  for r in select unnest(array[
    'cak_backup_salvage_manifest','cak_autonomous_evidence','cak_direct_evidence',
    'cak_narrative_events','cak_ohlcv_cache','cak_outcome_memory','cak_provider_audit',
    'cak_radar_snapshots','cak_research_memory','cak_scan_job_chunks','cak_scan_jobs',
    'cak_scan_runs','cak_source_cache','independent_data_backfill','ohlcv_daily_cache',
    'cak_storage_policy_v1','cak_storage_measurement_v1'
  ]) table_name loop
    execute format('alter table public.%I enable row level security',r.table_name);
    execute format('revoke all on table public.%I from public,anon,authenticated',r.table_name);
    execute format('grant select,insert,update,delete on table public.%I to service_role',r.table_name);
  end loop;
end $fn$;

revoke all on function public.cak_capture_storage_v1(),public.cak_prune_storage_v1(date)
from public,anon,authenticated;
grant execute on function public.cak_capture_storage_v1(),public.cak_prune_storage_v1(date)
to service_role;
