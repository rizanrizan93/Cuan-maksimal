-- Company reference and resilient non-core EOD evidence.

create or replace function public.cak_idx_ingest_company_reference_v2(
  p_observed_on date default ((now() at time zone 'Asia/Jakarta')::date)
)
returns jsonb
language plpgsql
security definer
set search_path=pg_catalog,public,extensions
as $fn$
declare v_url text; v_payload jsonb; v_rows integer:=0; v_reported integer:=0;
begin
  v_url:='https://block.idx.id/primary/ListedCompany/GetCompanyProfiles?emitenType=s&start=0&length=2000';
  v_payload:=public.cak_idx_http_json_v1(v_url);
  v_reported:=coalesce((v_payload->>'recordsTotal')::integer,0);
  if v_reported<800 or jsonb_array_length(coalesce(v_payload->'data','[]'::jsonb))<>v_reported then
    raise exception 'incomplete company directory: accepted %, reported %',
      jsonb_array_length(coalesce(v_payload->'data','[]'::jsonb)),v_reported;
  end if;
  insert into public.cak_idx_company_snapshot(
    ticker,observed_on,company_name,sector,subsector,listing_date,
    source_url,payload_hash,source_verified,raw_payload
  )
  select upper(x->>'KodeEmiten'),p_observed_on,nullif(x->>'NamaEmiten',''),
    nullif(x->>'Sektor',''),nullif(x->>'SubSektor',''),
    case when length(coalesce(x->>'TanggalPencatatan',''))>=10
      then substring(x->>'TanggalPencatatan' from 1 for 10)::date end,
    v_url,encode(extensions.digest(convert_to(x::text,'UTF8'),'sha256'),'hex'),true,x
  from jsonb_array_elements(v_payload->'data') x
  where upper(coalesce(x->>'KodeEmiten','')) ~ '^[A-Z0-9.-]{2,12}$'
  on conflict(ticker,observed_on) do update set
    company_name=excluded.company_name,sector=excluded.sector,
    subsector=excluded.subsector,listing_date=excluded.listing_date,
    source_url=excluded.source_url,payload_hash=excluded.payload_hash,
    source_verified=true,raw_payload=excluded.raw_payload,ingested_at=clock_timestamp();
  get diagnostics v_rows=row_count;
  insert into public.cak_idx_payload_manifest(
    provider,endpoint_key,target_date,payload_hash,source_url,rows_received,
    rows_accepted,validation_state,producer_version
  ) values(
    'BLOCK_IDX','companies',p_observed_on,
    encode(extensions.digest(convert_to(v_payload::text,'UTF8'),'sha256'),'hex'),
    v_url,v_reported,v_rows,'VALID','EMIR_BLOCK_IDX_DB_EOD_V2'
  )
  on conflict(provider,endpoint_key,target_date,payload_hash) do update set
    rows_received=excluded.rows_received,rows_accepted=excluded.rows_accepted,
    validation_state=excluded.validation_state,producer_version=excluded.producer_version,
    fetched_at=clock_timestamp();
  return jsonb_build_object('observed_on',p_observed_on,'rows',v_rows,'reported',v_reported);
end $fn$;

create or replace function public.cak_idx_ingest_announcements_all_v2(p_from date,p_to date)
returns jsonb
language plpgsql
security definer
set search_path=pg_catalog,public
as $fn$
declare v_first jsonb; v_page jsonb; v_pages integer:=1; v_total integer:=0;
  v_failures integer:=0; v_message text;
begin
  begin
    v_first:=public.cak_idx_ingest_announcement_page_v2(p_from,p_to,1);
    v_pages:=least(coalesce((v_first->>'pages')::integer,1),100);
    v_total:=coalesce((v_first->>'rows')::integer,0);
  exception when others then
    v_failures:=1; v_message:=left(sqlerrm,700);
    insert into public.cak_idx_ingestion_failures(
      endpoint_key,target_date,failure_class,message,retryable,last_seen_at,attempt_count
    ) values('announcements',p_to,sqlstate,'page=1 '||v_message,true,clock_timestamp(),1)
    on conflict(endpoint_key,target_date,failure_class,message) do update set
      last_seen_at=clock_timestamp(),attempt_count=public.cak_idx_ingestion_failures.attempt_count+1;
    return jsonb_build_object('state','PARTIAL','pages_attempted',1,'rows',0,'failures',1);
  end;
  for i in 2..v_pages loop
    begin
      v_page:=public.cak_idx_ingest_announcement_page_v2(p_from,p_to,i);
      v_total:=v_total+coalesce((v_page->>'rows')::integer,0);
    exception when others then
      v_failures:=v_failures+1; v_message:=left(sqlerrm,700);
      insert into public.cak_idx_ingestion_failures(
        endpoint_key,target_date,failure_class,message,retryable,last_seen_at,attempt_count
      ) values('announcements',p_to,sqlstate,'page='||i::text||' '||v_message,true,clock_timestamp(),1)
      on conflict(endpoint_key,target_date,failure_class,message) do update set
        last_seen_at=clock_timestamp(),attempt_count=public.cak_idx_ingestion_failures.attempt_count+1;
    end;
  end loop;
  return jsonb_build_object(
    'state',case when v_failures=0 then 'COMPLETE' else 'PARTIAL' end,
    'pages_attempted',v_pages,'rows',v_total,'failures',v_failures,
    'reported_total',coalesce((v_first->>'total')::integer,0)
  );
end $fn$;

revoke all on function public.cak_idx_ingest_company_reference_v2(date),
  public.cak_idx_ingest_announcements_all_v2(date,date)
from public,anon,authenticated;
grant execute on function public.cak_idx_ingest_company_reference_v2(date),
  public.cak_idx_ingest_announcements_all_v2(date,date)
to service_role;

create or replace function public.cak_idx_run_eod_v2(
  p_trade_date date default ((now() at time zone 'Asia/Jakarta')::date)
)
returns jsonb
language plpgsql
security definer
set search_path=pg_catalog,public
as $fn$
declare v_run_key text; v_market jsonb; v_events jsonb; v_announcements jsonb;
  v_rank jsonb; v_storage jsonb;
begin
  v_run_key:='EMIR_BLOCK_IDX_DB_EOD_V2:'||p_trade_date::text;
  insert into public.cak_idx_ingestion_runs(
    run_key,mode,requested_from,requested_to,status,producer_version
  ) values(v_run_key,'daily',p_trade_date,p_trade_date,'RUNNING','EMIR_BLOCK_IDX_DB_EOD_V2')
  on conflict(run_key) do update set status='RUNNING',started_at=clock_timestamp(),completed_at=null;
  v_storage:=public.cak_prune_storage_v1(p_trade_date);
  if v_storage->>'state'='HARD_STOP' then raise exception 'storage hard-stop before EOD'; end if;
  v_market:=public.cak_idx_ingest_official_market_day_v2(p_trade_date);
  begin
    v_events:=public.cak_idx_ingest_official_events_v2(p_trade_date-14,p_trade_date);
  exception when others then
    v_events:=jsonb_build_object('state','PARTIAL','error',left(sqlerrm,700));
  end;
  begin
    v_announcements:=public.cak_idx_ingest_announcements_all_v2(p_trade_date-14,p_trade_date);
  exception when others then
    v_announcements:=jsonb_build_object('state','PARTIAL','error',left(sqlerrm,700));
  end;
  if v_market->>'state'='VALID' then v_rank:=public.cak_refresh_idx_ranking_v1(p_trade_date); end if;
  v_storage:=public.cak_prune_storage_v1(p_trade_date);
  update public.cak_idx_ingestion_runs set
    status=case when v_market->>'state'='VALID' then 'COMPLETED' else 'SOURCE_NOT_READY' end,
    core_sessions_attempted=1,
    core_sessions_loaded=case when v_market->>'state'='VALID' then 1 else 0 end,
    rows_persisted=coalesce((v_market->>'stock_rows')::integer,0)
      +coalesce((v_market->>'index_rows')::integer,0)
      +coalesce((v_market->>'broker_rows')::integer,0)
      +coalesce((v_events->>'event_rows')::integer,0)
      +coalesce((v_announcements->>'rows')::integer,0),
    detail=jsonb_build_object('market',v_market,'events',v_events,
      'announcements',v_announcements,'ranking',v_rank,'storage',v_storage),
    completed_at=clock_timestamp()
  where run_key=v_run_key;
  return jsonb_build_object('market',v_market,'events',v_events,
    'announcements',v_announcements,'ranking',v_rank,'storage',v_storage);
exception when others then
  update public.cak_idx_ingestion_runs set status='FAILED',failures=failures+1,
    detail=jsonb_build_object('sqlstate',sqlstate,'error',left(sqlerrm,1000)),
    completed_at=clock_timestamp()
  where run_key=v_run_key;
  raise;
end $fn$;

revoke all on function public.cak_idx_run_eod_v2(date) from public,anon,authenticated;
grant execute on function public.cak_idx_run_eod_v2(date) to service_role;

do $fn$
declare v_job bigint;
begin
  select jobid into v_job from cron.job where jobname='emir-company-reference-weekly';
  if v_job is not null then perform cron.unschedule(v_job); end if;
  perform cron.schedule(
    'emir-company-reference-weekly','15 12 * * 1',
    $command$select public.cak_idx_ingest_company_reference_v2(((now() at time zone 'Asia/Jakarta')::date));$command$
  );
end $fn$;
