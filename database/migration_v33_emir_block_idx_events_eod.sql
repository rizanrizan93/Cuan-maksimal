-- Official event/risk evidence from verified Block IDX routes.

create or replace function public.cak_idx_ingest_official_events_v2(
  p_from date,
  p_to date
)
returns jsonb
language plpgsql
security definer
set search_path=pg_catalog,public,extensions
as $fn$
declare
  v_from text:=to_char(p_from,'YYYYMMDD');
  v_to text:=to_char(p_to,'YYYYMMDD');
  v_uma_url text;
  v_suspension_url text;
  v_issued_url text;
  v_announcement_url text;
  v_uma jsonb;
  v_suspension jsonb;
  v_issued jsonb;
  v_announcement jsonb;
  v_rows integer:=0;
begin
  if p_from>p_to or p_to-p_from>370 then
    raise exception 'event window must be ordered and no wider than 370 days';
  end if;
  perform pg_advisory_xact_lock(hashtext('emir-block-idx-events-'||p_from::text||'-'||p_to::text));
  v_uma_url:='https://block.idx.id/primary/NewsAnnouncement/GetUma?dateFrom='||v_from||'&dateTo='||v_to||'&indexfrom=0&pagesize=5000';
  v_suspension_url:='https://block.idx.id/primary/NewsAnnouncement/GetSuspension?dateFrom='||v_from||'&dateTo='||v_to||'&indexfrom=0&pagesize=5000';
  v_issued_url:='https://block.idx.id/primary/ListingActivity/GetIssuedHistory?caType=&dateFrom='||v_from||'&dateTo='||v_to||'&start=0&length=5000';
  v_announcement_url:='https://block.idx.id/primary/NewsAnnouncement/GetAllAnnouncement?keywords=&dateFrom='||v_from||'&dateTo='||v_to||'&pageNumber=1&pageSize=1000&lang=id';
  v_uma:=public.cak_idx_http_json_v1(v_uma_url);
  v_suspension:=public.cak_idx_http_json_v1(v_suspension_url);
  v_issued:=public.cak_idx_http_json_v1(v_issued_url);
  v_announcement:=public.cak_idx_http_json_v1(v_announcement_url);

  with raw as (
    select 'UMA'::text family,'UMA'::text event_type,
      x->>'CompanyID' ticker,x->>'UMADate' event_time,null::text publication_time,
      coalesce(nullif(x->>'UMAID',''),nullif(x->>'AnnouncementNo',''),nullif(x->>'Attachment','')) source_ref,
      x->>'Judul' title,v_uma_url source_url,x payload
    from jsonb_array_elements(coalesce(v_uma->'Results','[]'::jsonb)) x
    union all
    select 'SUSPENSION',
      case when lower(coalesce(x->>'Info_Type','')||' '||coalesce(x->>'Judul',''))
                  ~ '(upt|pembukaan|unsuspend)' then 'UNSUSPEND' else 'SUSPEND' end,
      x->>'Kode',x->>'Date',null,
      coalesce(nullif(x->>'AnnouncementNo',''),nullif(x->>'Data_Download','')),
      x->>'Judul',v_suspension_url,x
    from jsonb_array_elements(coalesce(v_suspension->'Results','[]'::jsonb)) x
    union all
    select 'ISSUED_HISTORY',
      coalesce(nullif(upper(replace(x->>'JenisTindakan',' ','_')),''),'CAPITAL_ACTION'),
      x->>'KodeEmiten',x->>'TanggalPencatatan',null,
      coalesce(nullif(x->>'id',''),nullif(x->>'ID','')),
      x->>'JenisTindakan',v_issued_url,x
    from jsonb_array_elements(coalesce(v_issued->'data','[]'::jsonb)) x
    union all
    select 'ANNOUNCEMENT','ANNOUNCEMENT',
      x->>'Code',x->>'PublishDate',x->>'PublishDate',
      coalesce(nullif(x->>'AnnouncementNo',''),nullif(x->>'Id','')),
      x->>'Title',v_announcement_url,x
    from jsonb_array_elements(coalesce(v_announcement->'Items','[]'::jsonb)) x
  ), clean as (
    select family,event_type,
      case when upper(coalesce(ticker,'')) ~ '^[A-Z0-9.-]{2,12}$' then upper(ticker) end ticker,
      substring(event_time from 1 for 10)::date event_date,
      case when publication_time is not null then substring(publication_time from 1 for 10)::date end publication_date,
      coalesce(nullif(source_ref,''),encode(extensions.digest(convert_to(payload::text,'UTF8'),'sha256'),'hex')) source_ref,
      nullif(title,'') title,source_url,payload
    from raw where event_time is not null and length(event_time)>=10
  )
  insert into public.cak_idx_events(
    event_family,event_type,ticker,event_date,publication_date,source_ref,title,
    source_url,payload_hash,source_verified,raw_payload
  )
  select family,event_type,ticker,event_date,publication_date,left(source_ref,512),title,
    source_url,encode(extensions.digest(convert_to(payload::text,'UTF8'),'sha256'),'hex'),
    true,payload
  from clean
  on conflict(event_family,event_type,event_date,source_ref,payload_hash) do update set
    ticker=excluded.ticker,publication_date=excluded.publication_date,title=excluded.title,
    source_url=excluded.source_url,source_verified=true,raw_payload=excluded.raw_payload,
    ingested_at=clock_timestamp();
  get diagnostics v_rows=row_count;

  insert into public.cak_idx_payload_manifest(
    provider,endpoint_key,target_date,payload_hash,source_url,rows_received,
    rows_accepted,validation_state,producer_version
  ) values
    ('BLOCK_IDX','uma',p_to,encode(extensions.digest(convert_to(v_uma::text,'UTF8'),'sha256'),'hex'),v_uma_url,
      jsonb_array_length(coalesce(v_uma->'Results','[]'::jsonb)),jsonb_array_length(coalesce(v_uma->'Results','[]'::jsonb)),'VALID','EMIR_BLOCK_IDX_DB_EOD_V2'),
    ('BLOCK_IDX','suspension',p_to,encode(extensions.digest(convert_to(v_suspension::text,'UTF8'),'sha256'),'hex'),v_suspension_url,
      jsonb_array_length(coalesce(v_suspension->'Results','[]'::jsonb)),jsonb_array_length(coalesce(v_suspension->'Results','[]'::jsonb)),'VALID','EMIR_BLOCK_IDX_DB_EOD_V2'),
    ('BLOCK_IDX','issued_history',p_to,encode(extensions.digest(convert_to(v_issued::text,'UTF8'),'sha256'),'hex'),v_issued_url,
      jsonb_array_length(coalesce(v_issued->'data','[]'::jsonb)),jsonb_array_length(coalesce(v_issued->'data','[]'::jsonb)),'VALID','EMIR_BLOCK_IDX_DB_EOD_V2'),
    ('BLOCK_IDX','announcements',p_to,encode(extensions.digest(convert_to(v_announcement::text,'UTF8'),'sha256'),'hex'),v_announcement_url,
      jsonb_array_length(coalesce(v_announcement->'Items','[]'::jsonb)),jsonb_array_length(coalesce(v_announcement->'Items','[]'::jsonb)),'VALID','EMIR_BLOCK_IDX_DB_EOD_V2')
  on conflict(provider,endpoint_key,target_date,payload_hash) do update set
    rows_received=excluded.rows_received,rows_accepted=excluded.rows_accepted,
    validation_state=excluded.validation_state,producer_version=excluded.producer_version,
    fetched_at=clock_timestamp();

  return jsonb_build_object(
    'from',p_from,'to',p_to,'event_rows',v_rows,
    'uma',jsonb_array_length(coalesce(v_uma->'Results','[]'::jsonb)),
    'suspension',jsonb_array_length(coalesce(v_suspension->'Results','[]'::jsonb)),
    'issued_history',jsonb_array_length(coalesce(v_issued->'data','[]'::jsonb)),
    'announcements',jsonb_array_length(coalesce(v_announcement->'Items','[]'::jsonb))
  );
end $fn$;

create or replace function public.cak_idx_ingest_announcement_page_v2(
  p_from date,p_to date,p_page integer
)
returns jsonb
language plpgsql
security definer
set search_path=pg_catalog,public,extensions
as $fn$
declare v_url text; v_payload jsonb; v_rows integer:=0;
begin
  if p_from>p_to or p_to-p_from>370 or p_page<1 or p_page>100 then
    raise exception 'invalid announcement page request';
  end if;
  v_url:='https://block.idx.id/primary/NewsAnnouncement/GetAllAnnouncement?keywords=&dateFrom='
    ||to_char(p_from,'YYYYMMDD')||'&dateTo='||to_char(p_to,'YYYYMMDD')
    ||'&pageNumber='||p_page::text||'&pageSize=1000&lang=id';
  v_payload:=public.cak_idx_http_json_v1(v_url);
  insert into public.cak_idx_events(
    event_family,event_type,ticker,event_date,publication_date,source_ref,title,
    source_url,payload_hash,source_verified,raw_payload
  )
  select 'ANNOUNCEMENT','ANNOUNCEMENT',
    case when upper(coalesce(x->>'Code','')) ~ '^[A-Z0-9.-]{2,12}$' then upper(x->>'Code') end,
    substring(x->>'PublishDate' from 1 for 10)::date,
    substring(x->>'PublishDate' from 1 for 10)::date,
    left(coalesce(nullif(x->>'AnnouncementNo',''),nullif(x->>'Id',''),
      encode(extensions.digest(convert_to(x::text,'UTF8'),'sha256'),'hex')),512),
    nullif(x->>'Title',''),v_url,
    encode(extensions.digest(convert_to(x::text,'UTF8'),'sha256'),'hex'),true,x
  from jsonb_array_elements(coalesce(v_payload->'Items','[]'::jsonb)) x
  where x->>'PublishDate' is not null and length(x->>'PublishDate')>=10
  on conflict(event_family,event_type,event_date,source_ref,payload_hash) do update set
    ticker=excluded.ticker,publication_date=excluded.publication_date,title=excluded.title,
    source_url=excluded.source_url,source_verified=true,raw_payload=excluded.raw_payload,
    ingested_at=clock_timestamp();
  get diagnostics v_rows=row_count;
  insert into public.cak_idx_payload_manifest(
    provider,endpoint_key,target_date,payload_hash,source_url,rows_received,
    rows_accepted,validation_state,producer_version
  ) values(
    'BLOCK_IDX','announcements',p_to,
    encode(extensions.digest(convert_to(v_payload::text,'UTF8'),'sha256'),'hex'),v_url,
    jsonb_array_length(coalesce(v_payload->'Items','[]'::jsonb)),v_rows,
    case when jsonb_array_length(coalesce(v_payload->'Items','[]'::jsonb))>0 then 'VALID' else 'NO_DATA' end,
    'EMIR_BLOCK_IDX_DB_EOD_V2'
  )
  on conflict(provider,endpoint_key,target_date,payload_hash) do update set
    rows_received=excluded.rows_received,rows_accepted=excluded.rows_accepted,
    validation_state=excluded.validation_state,producer_version=excluded.producer_version,
    fetched_at=clock_timestamp();
  return jsonb_build_object('page',coalesce((v_payload->>'PageNumber')::integer,p_page),
    'pages',coalesce((v_payload->>'PageCount')::integer,1),
    'total',coalesce((v_payload->>'ItemCount')::integer,0),'rows',v_rows);
end $fn$;

create or replace function public.cak_idx_ingest_announcements_all_v2(p_from date,p_to date)
returns jsonb
language plpgsql
security definer
set search_path=pg_catalog,public
as $fn$
declare v_first jsonb; v_page jsonb; v_pages integer; v_total integer:=0;
begin
  v_first:=public.cak_idx_ingest_announcement_page_v2(p_from,p_to,1);
  v_pages:=least(coalesce((v_first->>'pages')::integer,1),100);
  v_total:=coalesce((v_first->>'rows')::integer,0);
  for i in 2..v_pages loop
    v_page:=public.cak_idx_ingest_announcement_page_v2(p_from,p_to,i);
    v_total:=v_total+coalesce((v_page->>'rows')::integer,0);
  end loop;
  return jsonb_build_object('pages',v_pages,'rows',v_total,
    'reported_total',coalesce((v_first->>'total')::integer,0));
end $fn$;

revoke all on function public.cak_idx_ingest_official_events_v2(date,date)
from public,anon,authenticated;
grant execute on function public.cak_idx_ingest_official_events_v2(date,date)
to service_role;
revoke all on function public.cak_idx_ingest_announcement_page_v2(date,date,integer),
  public.cak_idx_ingest_announcements_all_v2(date,date)
from public,anon,authenticated;
grant execute on function public.cak_idx_ingest_announcement_page_v2(date,date,integer),
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
declare v_run_key text; v_market jsonb; v_events jsonb; v_announcements jsonb; v_rank jsonb; v_storage jsonb;
begin
  v_run_key:='EMIR_BLOCK_IDX_DB_EOD_V2:'||p_trade_date::text;
  insert into public.cak_idx_ingestion_runs(
    run_key,mode,requested_from,requested_to,status,producer_version
  ) values(v_run_key,'daily',p_trade_date,p_trade_date,'RUNNING','EMIR_BLOCK_IDX_DB_EOD_V2')
  on conflict(run_key) do update set status='RUNNING',started_at=clock_timestamp(),completed_at=null;
  v_storage:=public.cak_prune_storage_v1(p_trade_date);
  if v_storage->>'state'='HARD_STOP' then raise exception 'storage hard-stop before EOD'; end if;
  v_market:=public.cak_idx_ingest_official_market_day_v2(p_trade_date);
  v_events:=public.cak_idx_ingest_official_events_v2(p_trade_date-14,p_trade_date);
  v_announcements:=public.cak_idx_ingest_announcements_all_v2(p_trade_date-14,p_trade_date);
  if v_market->>'state'='VALID' then v_rank:=public.cak_refresh_idx_ranking_v1(p_trade_date); end if;
  v_storage:=public.cak_prune_storage_v1(p_trade_date);
  update public.cak_idx_ingestion_runs set
    status=case when v_market->>'state'='VALID' then 'COMPLETED' else 'SOURCE_NOT_READY' end,
    core_sessions_attempted=1,
    core_sessions_loaded=case when v_market->>'state'='VALID' then 1 else 0 end,
    rows_persisted=coalesce((v_market->>'stock_rows')::integer,0)
      +coalesce((v_market->>'index_rows')::integer,0)
      +coalesce((v_market->>'broker_rows')::integer,0)
      +coalesce((v_events->>'event_rows')::integer,0),
    detail=jsonb_build_object('market',v_market,'events',v_events,'announcements',v_announcements,'ranking',v_rank,'storage',v_storage),
    completed_at=clock_timestamp()
  where run_key=v_run_key;
  return jsonb_build_object('market',v_market,'events',v_events,'announcements',v_announcements,'ranking',v_rank,'storage',v_storage);
exception when others then
  update public.cak_idx_ingestion_runs set status='FAILED',failures=failures+1,
    detail=jsonb_build_object('sqlstate',sqlstate,'error',left(sqlerrm,1000)),
    completed_at=clock_timestamp()
  where run_key=v_run_key;
  raise;
end $fn$;

revoke all on function public.cak_idx_run_eod_v2(date) from public,anon,authenticated;
grant execute on function public.cak_idx_run_eod_v2(date) to service_role;
