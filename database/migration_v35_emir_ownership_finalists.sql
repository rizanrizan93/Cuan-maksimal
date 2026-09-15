-- Unique finalist evidence from CompanyProfilesDetail.
-- Company announcements and TradingInfoSS remain derived from the already complete
-- all-announcement and stock-summary planes to avoid duplicate storage.

create table if not exists public.cak_idx_ownership_snapshot (
  ticker text not null,
  observed_on date not null,
  controller_pct numeric,
  public_pct numeric,
  treasury_pct numeric,
  holder_count integer not null default 0,
  source_url text not null,
  payload_hash text not null,
  source_verified boolean not null default true,
  raw_payload jsonb not null default '{}'::jsonb,
  ingested_at timestamptz not null default now(),
  primary key(ticker,observed_on,payload_hash)
);
create index if not exists cak_idx_ownership_latest_idx
  on public.cak_idx_ownership_snapshot(ticker,observed_on desc);
alter table public.cak_idx_ownership_snapshot enable row level security;
revoke all on table public.cak_idx_ownership_snapshot from public,anon,authenticated;
grant select,insert,update,delete on table public.cak_idx_ownership_snapshot to service_role;

create or replace function public.cak_idx_ingest_ownership_ticker_v2(
  p_ticker text,p_observed_on date
)
returns jsonb
language plpgsql
security definer
set search_path=pg_catalog,public,extensions
as $fn$
declare v_ticker text; v_url text; v_payload jsonb; v_hash text; v_holders jsonb;
  v_controller numeric; v_public numeric; v_treasury numeric; v_count integer;
begin
  v_ticker:=upper(replace(trim(p_ticker),'.JK',''));
  if v_ticker !~ '^[A-Z0-9.-]{2,12}$' then raise exception 'invalid ticker'; end if;
  v_url:='https://block.idx.id/primary/ListedCompany/GetCompanyProfilesDetail?emitenType=s&kodeEmiten='||v_ticker;
  v_payload:=public.cak_idx_http_json_v1(v_url);
  if coalesce((v_payload->>'ResultCount')::integer,0)<>1 then
    raise exception 'company profile unavailable for %',v_ticker;
  end if;
  v_holders:=coalesce(v_payload->'PemegangSaham','[]'::jsonb);
  select
    sum(coalesce(public.cak_idx_safe_numeric_v1(x->>'Persentase'),0))
      filter(where coalesce((x->>'Pengendali')::boolean,false)),
    sum(coalesce(public.cak_idx_safe_numeric_v1(x->>'Persentase'),0))
      filter(where lower(coalesce(x->>'Kategori','')) like 'masyarakat%'),
    sum(coalesce(public.cak_idx_safe_numeric_v1(x->>'Persentase'),0))
      filter(where lower(coalesce(x->>'Kategori','')) like '%treasury%'),
    count(*)::integer
  into v_controller,v_public,v_treasury,v_count
  from jsonb_array_elements(v_holders) x;
  v_hash:=encode(extensions.digest(convert_to(v_payload::text,'UTF8'),'sha256'),'hex');
  insert into public.cak_idx_ownership_snapshot(
    ticker,observed_on,controller_pct,public_pct,treasury_pct,holder_count,
    source_url,payload_hash,source_verified,raw_payload
  ) values(
    v_ticker,p_observed_on,v_controller,v_public,v_treasury,coalesce(v_count,0),
    v_url,v_hash,true,v_payload
  )
  on conflict(ticker,observed_on,payload_hash) do update set
    controller_pct=excluded.controller_pct,public_pct=excluded.public_pct,
    treasury_pct=excluded.treasury_pct,holder_count=excluded.holder_count,
    source_url=excluded.source_url,source_verified=true,raw_payload=excluded.raw_payload,
    ingested_at=clock_timestamp();
  insert into public.cak_idx_payload_manifest(
    provider,endpoint_key,target_date,payload_hash,source_url,rows_received,
    rows_accepted,validation_state,producer_version
  ) values(
    'BLOCK_IDX','company_profile',p_observed_on,v_hash,v_url,
    jsonb_array_length(v_holders),1,'VALID','EMIR_BLOCK_IDX_DB_EOD_V2'
  )
  on conflict(provider,endpoint_key,target_date,payload_hash) do update set
    rows_received=excluded.rows_received,rows_accepted=1,validation_state='VALID',
    producer_version=excluded.producer_version,fetched_at=clock_timestamp();
  return jsonb_build_object('ticker',v_ticker,'controller_pct',v_controller,
    'public_pct',v_public,'treasury_pct',v_treasury,'holder_count',v_count);
end $fn$;

create or replace function public.cak_idx_refresh_finalist_ownership_v2(
  p_rank_date date,p_limit integer default 60
)
returns jsonb
language plpgsql
security definer
set search_path=pg_catalog,public
as $fn$
declare v_ok integer:=0; v_failed integer:=0; v_message text; r record;
begin
  for r in
    select ticker from public.cak_idx_rank_daily
    where rank_date=p_rank_date order by overall_rank limit greatest(3,least(p_limit,100))
  loop
    begin
      perform public.cak_idx_ingest_ownership_ticker_v2(r.ticker,p_rank_date);
      v_ok:=v_ok+1;
    exception when others then
      v_failed:=v_failed+1; v_message:=left(sqlerrm,700);
      insert into public.cak_idx_ingestion_failures(
        endpoint_key,target_date,failure_class,message,retryable,last_seen_at,attempt_count
      ) values('company_profile',p_rank_date,sqlstate,r.ticker||' '||v_message,true,clock_timestamp(),1)
      on conflict(endpoint_key,target_date,failure_class,message) do update set
        last_seen_at=clock_timestamp(),attempt_count=public.cak_idx_ingestion_failures.attempt_count+1;
    end;
  end loop;
  delete from public.cak_idx_ownership_snapshot o
  where o.observed_on<(p_rank_date-interval '6 months')::date
     or exists(
       select 1 from (
         select ticker,observed_on,payload_hash,
           row_number() over(partition by ticker order by observed_on desc,ingested_at desc) rn
         from public.cak_idx_ownership_snapshot
       ) x where x.ticker=o.ticker and x.observed_on=o.observed_on
         and x.payload_hash=o.payload_hash and x.rn>2
     );
  return jsonb_build_object('rank_date',p_rank_date,'loaded',v_ok,'failed',v_failed);
end $fn$;

revoke all on function public.cak_idx_ingest_ownership_ticker_v2(text,date),
  public.cak_idx_refresh_finalist_ownership_v2(date,integer)
from public,anon,authenticated;
grant execute on function public.cak_idx_ingest_ownership_ticker_v2(text,date),
  public.cak_idx_refresh_finalist_ownership_v2(date,integer)
to service_role;

update public.cak_idx_endpoint_catalog set enabled=false,
  scoring_role='DERIVED_FROM_COMPLETE_ALL_ANNOUNCEMENT'
where endpoint_key='company_announcements';
update public.cak_idx_endpoint_catalog set enabled=false,
  scoring_role='DERIVED_FROM_STOCK_SUMMARY_NO_DUPLICATE_STORAGE'
where endpoint_key='trading_info';

do $fn$
declare v_job bigint;
begin
  select jobid into v_job from cron.job where jobname='emir-finalist-ownership-weekly';
  if v_job is not null then perform cron.unschedule(v_job); end if;
  perform cron.schedule(
    'emir-finalist-ownership-weekly','30 12 * * 1',
    $command$select public.cak_idx_refresh_finalist_ownership_v2(
      (select max(rank_date) from public.cak_idx_rank_daily),60
    );$command$
  );
end $fn$;
