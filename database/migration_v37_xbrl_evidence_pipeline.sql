-- Official IDX XBRL ingestion queue, worker controls, final ranking, and bounded retention.
create extension if not exists pg_net;

create table if not exists public.cak_idx_fundamental_ingestion_queue (
  rank_date date not null,
  ticker text not null,
  overall_rank integer not null,
  report_year integer,
  report_period text,
  period_end date,
  publication_date date,
  source_file_id text,
  source_file_name text,
  source_file_path text,
  source_file_size bigint,
  status text not null default 'PENDING',
  attempts integer not null default 0,
  next_retry_at timestamptz,
  claimed_at timestamptz,
  completed_at timestamptz,
  worker_id text,
  payload_hash text,
  coverage_pct numeric,
  error_code text,
  parser_version text not null default 'EMIR_IDX_XBRL_EDGE_V1',
  updated_at timestamptz not null default now(),
  primary key (rank_date,ticker),
  constraint cak_idx_fundamental_queue_ticker_ck check (ticker ~ '^[A-Z0-9]{4,8}$'),
  constraint cak_idx_fundamental_queue_status_ck check (status in (
    'PENDING','RUNNING','RETRY','COMPLETE','PARSE_FAILED',
    'OFFICIAL_FILING_UNAVAILABLE','SPECIAL_INSTRUMENT'
  )),
  constraint cak_idx_fundamental_queue_attempts_ck check (attempts between 0 and 10),
  constraint cak_idx_fundamental_queue_path_ck check (
    source_file_path is null or source_file_path like '/Portals/%'
  )
);

create index if not exists cak_idx_fundamental_queue_claim_idx
  on public.cak_idx_fundamental_ingestion_queue (status,next_retry_at,rank_date desc,overall_rank)
  where status in ('PENDING','RETRY');

create table if not exists public.cak_idx_worker_control (
  worker_name text primary key,
  token_hash text not null,
  enabled boolean not null default true,
  updated_at timestamptz not null default now()
);

alter table public.cak_idx_fundamental_ingestion_queue enable row level security;
alter table public.cak_idx_worker_control enable row level security;
revoke all on public.cak_idx_fundamental_ingestion_queue from public,anon,authenticated;
revoke all on public.cak_idx_worker_control from public,anon,authenticated;
grant select,insert,update,delete on public.cak_idx_fundamental_ingestion_queue to service_role;
grant select,insert,update,delete on public.cak_idx_worker_control to service_role;

create or replace function public.cak_idx_set_worker_token_v1(
  p_worker_name text,
  p_token text
)
returns void
language plpgsql
security definer
set search_path = pg_catalog,public,extensions
as $$
begin
  if length(coalesce(p_token,'')) < 32 then
    raise exception 'WORKER_TOKEN_TOO_SHORT';
  end if;
  insert into public.cak_idx_worker_control(worker_name,token_hash,enabled,updated_at)
  values(lower(trim(p_worker_name)),encode(extensions.digest(p_token,'sha256'),'hex'),true,now())
  on conflict(worker_name) do update set
    token_hash=excluded.token_hash,enabled=true,updated_at=now();
end
$$;

create or replace function public.cak_idx_validate_worker_token_v1(
  p_worker_name text,
  p_token text
)
returns boolean
language sql
stable
security definer
set search_path = pg_catalog,public,extensions
as $$
  select exists(
    select 1 from public.cak_idx_worker_control
    where worker_name=lower(trim(p_worker_name)) and enabled
      and token_hash=encode(extensions.digest(coalesce(p_token,''),'sha256'),'hex')
  )
$$;

create or replace function public.cak_idx_seed_fundamental_queue_v1(
  p_rank_date date default null
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog,public
as $$
declare
  v_rank_date date;
  v_rows integer;
begin
  select coalesce(p_rank_date,max(rank_date)) into v_rank_date
  from public.cak_idx_rank_daily;
  if v_rank_date is null then
    raise exception 'RANK_DATE_NOT_AVAILABLE';
  end if;

  with selected as (
    select r.ticker,r.overall_rank,r.adtv_20d
    from public.cak_idx_rank_daily r
    where r.rank_date=v_rank_date
    order by r.overall_rank
    limit 900
  ), latest_market as (
    select distinct on(m.ticker) m.ticker,m.stock_name
    from public.cak_idx_market_daily m
    join selected s using(ticker)
    order by m.ticker,m.trade_date desc
  ), catalog as (
    select distinct on(f.ticker)
      f.ticker,f.report_year,f.report_period,f.period_end,
      coalesce(f.file_modified_at::date,f.ingested_at::date) publication_date,
      f.attachments
    from public.cak_idx_financial_filing_catalog f
    join selected s using(ticker)
    where f.source_verified
    order by f.ticker,f.period_end desc,f.file_modified_at desc,f.ingested_at desc
  ), preferred_file as (
    select c.ticker,c.report_year,c.report_period,c.period_end,c.publication_date,
      a.item->>'File_ID' source_file_id,
      a.item->>'File_Name' source_file_name,
      a.item->>'File_Path' source_file_path,
      public.cak_idx_safe_numeric_v1(a.item->>'File_Size')::bigint source_file_size
    from catalog c
    join lateral (
      select x item
      from jsonb_array_elements(c.attachments) x
      where lower(coalesce(x->>'File_Name','')) in ('instance.zip','inlinexbrl.zip')
        and coalesce(x->>'File_Path','') like '/Portals/%'
        and coalesce(public.cak_idx_safe_numeric_v1(x->>'File_Size'),0) between 1 and 25000000
      order by case lower(x->>'File_Name') when 'instance.zip' then 0 else 1 end,
               x->>'File_ID'
      limit 1
    ) a on true
  ), source_rows as (
    select s.*,m.stock_name,p.report_year,p.report_period,p.period_end,p.publication_date,
      p.source_file_id,p.source_file_name,p.source_file_path,p.source_file_size,
      exists(
        select 1 from public.cak_idx_fundamental_snapshot f
        where f.ticker=s.ticker and f.period_end=p.period_end and f.source_verified
          and f.coverage_pct>=50
          and f.raw_payload->>'source_file_id'=p.source_file_id
      ) already_complete
    from selected s
    left join latest_market m using(ticker)
    left join preferred_file p using(ticker)
  )
  insert into public.cak_idx_fundamental_ingestion_queue(
    rank_date,ticker,overall_rank,report_year,report_period,period_end,publication_date,
    source_file_id,source_file_name,source_file_path,source_file_size,status,
    completed_at,error_code,updated_at
  )
  select v_rank_date,ticker,overall_rank,report_year,report_period,period_end,publication_date,
    source_file_id,source_file_name,source_file_path,source_file_size,
    case
      when stock_name ilike 'MVS %' then 'SPECIAL_INSTRUMENT'
      when source_file_id is null then 'OFFICIAL_FILING_UNAVAILABLE'
      when already_complete then 'COMPLETE'
      else 'PENDING'
    end,
    case when already_complete then now() else null end,
    case
      when stock_name ilike 'MVS %' then 'NON_ORDINARY_MVS'
      when source_file_id is null then 'OFFICIAL_XBRL_NOT_PUBLISHED'
      else null
    end,
    now()
  from source_rows
  on conflict(rank_date,ticker) do update set
    overall_rank=excluded.overall_rank,
    report_year=excluded.report_year,
    report_period=excluded.report_period,
    period_end=excluded.period_end,
    publication_date=excluded.publication_date,
    source_file_id=excluded.source_file_id,
    source_file_name=excluded.source_file_name,
    source_file_path=excluded.source_file_path,
    source_file_size=excluded.source_file_size,
    status=case
      when public.cak_idx_fundamental_ingestion_queue.source_file_id is distinct from excluded.source_file_id
        then excluded.status
      when public.cak_idx_fundamental_ingestion_queue.status='RUNNING'
        and public.cak_idx_fundamental_ingestion_queue.claimed_at<now()-interval '15 minutes'
        then 'RETRY'
      else public.cak_idx_fundamental_ingestion_queue.status
    end,
    attempts=case
      when public.cak_idx_fundamental_ingestion_queue.source_file_id is distinct from excluded.source_file_id
        then 0 else public.cak_idx_fundamental_ingestion_queue.attempts end,
    error_code=case
      when public.cak_idx_fundamental_ingestion_queue.source_file_id is distinct from excluded.source_file_id
        then excluded.error_code else public.cak_idx_fundamental_ingestion_queue.error_code end,
    updated_at=now();

  get diagnostics v_rows=row_count;
  return jsonb_build_object(
    'state','SEEDED','rank_date',v_rank_date,'selected_tickers',v_rows,
    'pending',(select count(*) from public.cak_idx_fundamental_ingestion_queue where rank_date=v_rank_date and status in ('PENDING','RETRY')),
    'complete',(select count(*) from public.cak_idx_fundamental_ingestion_queue where rank_date=v_rank_date and status='COMPLETE'),
    'officially_unavailable',(select count(*) from public.cak_idx_fundamental_ingestion_queue where rank_date=v_rank_date and status='OFFICIAL_FILING_UNAVAILABLE'),
    'special_instruments',(select count(*) from public.cak_idx_fundamental_ingestion_queue where rank_date=v_rank_date and status='SPECIAL_INSTRUMENT')
  );
end
$$;

create or replace function public.cak_idx_claim_fundamental_jobs_v1(
  p_limit integer default 8,
  p_worker_id text default 'edge'
)
returns setof public.cak_idx_fundamental_ingestion_queue
language sql
volatile
security definer
set search_path = pg_catalog,public
as $$
  with latest as (
    select max(rank_date) rank_date from public.cak_idx_fundamental_ingestion_queue
  ), claim as (
    select q.ctid
    from public.cak_idx_fundamental_ingestion_queue q join latest l using(rank_date)
    where q.status in ('PENDING','RETRY')
      and coalesce(q.next_retry_at,'-infinity'::timestamptz)<=now()
      and q.attempts<3
    order by q.overall_rank
    for update skip locked
    limit least(greatest(p_limit,1),12)
  )
  update public.cak_idx_fundamental_ingestion_queue q
  set status='RUNNING',attempts=q.attempts+1,claimed_at=now(),
      worker_id=left(coalesce(p_worker_id,'edge'),80),updated_at=now()
  from claim c where q.ctid=c.ctid
  returning q.*
$$;

create or replace function public.cak_idx_finish_fundamental_job_v1(
  p_rank_date date,
  p_ticker text,
  p_success boolean,
  p_payload_hash text default null,
  p_coverage_pct numeric default null,
  p_error_code text default null
)
returns text
language plpgsql
security definer
set search_path = pg_catalog,public
as $$
declare
  v_attempts integer;
  v_status text;
begin
  select attempts into v_attempts
  from public.cak_idx_fundamental_ingestion_queue
  where rank_date=p_rank_date and ticker=upper(trim(p_ticker))
  for update;
  if not found then raise exception 'QUEUE_JOB_NOT_FOUND'; end if;

  if p_success then
    v_status:='COMPLETE';
  elsif v_attempts>=3 then
    v_status:='PARSE_FAILED';
  else
    v_status:='RETRY';
  end if;

  update public.cak_idx_fundamental_ingestion_queue set
    status=v_status,
    next_retry_at=case when v_status='RETRY' then now()+interval '5 minutes' else null end,
    completed_at=case when v_status in ('COMPLETE','PARSE_FAILED') then now() else null end,
    payload_hash=case when p_success then p_payload_hash else payload_hash end,
    coverage_pct=case when p_success then p_coverage_pct else coverage_pct end,
    error_code=case when p_success then null else left(coalesce(p_error_code,'UNKNOWN_ERROR'),160) end,
    updated_at=now()
  where rank_date=p_rank_date and ticker=upper(trim(p_ticker));
  return v_status;
end
$$;

create or replace function public.cak_idx_finalize_fundamental_evidence_v1(
  p_rank_date date default null
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog,public
as $$
declare
  v_rank_date date;
  v_open integer;
  v_rank jsonb;
begin
  select coalesce(p_rank_date,max(rank_date)) into v_rank_date
  from public.cak_idx_fundamental_ingestion_queue;
  if v_rank_date is null then return jsonb_build_object('state','QUEUE_EMPTY'); end if;

  update public.cak_idx_fundamental_ingestion_queue
  set status='RETRY',next_retry_at=now(),error_code='STALE_WORKER_RECLAIMED',updated_at=now()
  where rank_date=v_rank_date and status='RUNNING' and claimed_at<now()-interval '15 minutes';

  select count(*) into v_open
  from public.cak_idx_fundamental_ingestion_queue
  where rank_date=v_rank_date and status in ('PENDING','RUNNING','RETRY');
  if v_open>0 then
    return jsonb_build_object('state','EVIDENCE_PENDING','rank_date',v_rank_date,'open_jobs',v_open);
  end if;

  v_rank:=public.cak_refresh_idx_ranking_v2(v_rank_date);
  return jsonb_build_object(
    'state','FINALIZED','rank_date',v_rank_date,'ranking',v_rank,
    'complete',(select count(*) from public.cak_idx_fundamental_ingestion_queue where rank_date=v_rank_date and status='COMPLETE'),
    'parse_failed',(select count(*) from public.cak_idx_fundamental_ingestion_queue where rank_date=v_rank_date and status='PARSE_FAILED'),
    'officially_unavailable',(select count(*) from public.cak_idx_fundamental_ingestion_queue where rank_date=v_rank_date and status='OFFICIAL_FILING_UNAVAILABLE'),
    'special_instruments',(select count(*) from public.cak_idx_fundamental_ingestion_queue where rank_date=v_rank_date and status='SPECIAL_INSTRUMENT')
  );
end
$$;

create or replace function public.cak_idx_prune_fundamental_evidence_v1(
  p_as_of date default current_date
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog,public
as $$
declare
  v_queue integer:=0;
  v_snapshot integer:=0;
begin
  delete from public.cak_idx_fundamental_ingestion_queue
  where rank_date<p_as_of-14;
  get diagnostics v_queue=row_count;

  with ranked as (
    select ctid,row_number() over(
      partition by ticker order by observed_on desc,period_end desc,ingested_at desc
    ) rn
    from public.cak_idx_fundamental_snapshot
  )
  delete from public.cak_idx_fundamental_snapshot f
  using ranked r
  where f.ctid=r.ctid and r.rn>3 and f.observed_on<p_as_of-interval '6 months';
  get diagnostics v_snapshot=row_count;
  return jsonb_build_object('state','PRUNED','queue_rows',v_queue,'snapshot_rows',v_snapshot);
end
$$;

create or replace function public.cak_idx_kick_xbrl_worker_v1()
returns bigint
language plpgsql
security definer
set search_path = pg_catalog,public,vault,net
as $$
declare
  v_url text;
  v_key text;
  v_token text;
begin
  select decrypted_secret into v_url from vault.decrypted_secrets where name='emir_project_url' order by created_at desc limit 1;
  select decrypted_secret into v_key from vault.decrypted_secrets where name='emir_worker_anon_key' order by created_at desc limit 1;
  select decrypted_secret into v_token from vault.decrypted_secrets where name='emir_xbrl_worker_token' order by created_at desc limit 1;
  if v_url is null or v_key is null or v_token is null then return null; end if;
  return net.http_post(
    url:=v_url||'/functions/v1/emir-xbrl-worker',
    headers:=jsonb_build_object(
      'Content-Type','application/json','apikey',v_key,
      'Authorization','Bearer '||v_key,'x-emir-worker-token',v_token
    ),
    body:=jsonb_build_object('chain',true,'limit',8),
    timeout_milliseconds:=10000
  );
end
$$;

do $$
declare r record;
begin
  for r in select jobid from cron.job where jobname in (
    'emir-xbrl-seed-1935-wib','emir-xbrl-worker-1940-wib',
    'emir-xbrl-worker-retry-2030-wib','emir-xbrl-finalize-2110-wib',
    'emir-xbrl-finalize-retry-2200-wib','emir-xbrl-prune-2215-wib'
  ) loop perform cron.unschedule(r.jobid); end loop;
  perform cron.schedule('emir-xbrl-seed-1935-wib','35 12 * * 1-5',
    $cron$select public.cak_idx_seed_fundamental_queue_v1(((now() at time zone 'Asia/Jakarta')::date));$cron$);
  perform cron.schedule('emir-xbrl-worker-1940-wib','40 12 * * 1-5',
    $cron$select public.cak_idx_kick_xbrl_worker_v1();$cron$);
  perform cron.schedule('emir-xbrl-worker-retry-2030-wib','30 13 * * 1-5',
    $cron$select public.cak_idx_kick_xbrl_worker_v1();$cron$);
  perform cron.schedule('emir-xbrl-finalize-2110-wib','10 14 * * 1-5',
    $cron$select public.cak_idx_finalize_fundamental_evidence_v1(((now() at time zone 'Asia/Jakarta')::date));$cron$);
  perform cron.schedule('emir-xbrl-finalize-retry-2200-wib','0 15 * * 1-5',
    $cron$select public.cak_idx_finalize_fundamental_evidence_v1(((now() at time zone 'Asia/Jakarta')::date));$cron$);
  perform cron.schedule('emir-xbrl-prune-2215-wib','15 15 * * 1-5',
    $cron$select public.cak_idx_prune_fundamental_evidence_v1(((now() at time zone 'Asia/Jakarta')::date));$cron$);
end $$;

revoke all on function public.cak_idx_set_worker_token_v1(text,text) from public,anon,authenticated;
revoke all on function public.cak_idx_validate_worker_token_v1(text,text) from public,anon,authenticated;
revoke all on function public.cak_idx_seed_fundamental_queue_v1(date) from public,anon,authenticated;
revoke all on function public.cak_idx_claim_fundamental_jobs_v1(integer,text) from public,anon,authenticated;
revoke all on function public.cak_idx_finish_fundamental_job_v1(date,text,boolean,text,numeric,text) from public,anon,authenticated;
revoke all on function public.cak_idx_finalize_fundamental_evidence_v1(date) from public,anon,authenticated;
revoke all on function public.cak_idx_prune_fundamental_evidence_v1(date) from public,anon,authenticated;
revoke all on function public.cak_idx_kick_xbrl_worker_v1() from public,anon,authenticated;
grant execute on function public.cak_idx_set_worker_token_v1(text,text) to service_role;
grant execute on function public.cak_idx_validate_worker_token_v1(text,text) to service_role;
grant execute on function public.cak_idx_seed_fundamental_queue_v1(date) to service_role;
grant execute on function public.cak_idx_claim_fundamental_jobs_v1(integer,text) to service_role;
grant execute on function public.cak_idx_finish_fundamental_job_v1(date,text,boolean,text,numeric,text) to service_role;
grant execute on function public.cak_idx_finalize_fundamental_evidence_v1(date) to service_role;
grant execute on function public.cak_idx_prune_fundamental_evidence_v1(date) to service_role;
grant execute on function public.cak_idx_kick_xbrl_worker_v1() to service_role;
