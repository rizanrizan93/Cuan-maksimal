-- Server-side official Block IDX EOD fallback for EMIR.
-- Keeps the core six-month fact plane current even when the external runner is unavailable.

create extension if not exists http with schema extensions;
create extension if not exists pg_cron with schema pg_catalog;

create or replace function public.cak_idx_safe_numeric_v1(p_value text)
returns numeric
language plpgsql
immutable
strict
set search_path=pg_catalog
as $fn$
begin
  return replace(p_value,',','')::numeric;
exception when invalid_text_representation or numeric_value_out_of_range then
  return null;
end $fn$;

create or replace function public.cak_idx_http_json_v1(p_url text)
returns jsonb
language plpgsql
security definer
set search_path=pg_catalog,extensions
as $fn$
declare v_response extensions.http_response; v_payload jsonb;
begin
  if p_url not like 'https://block.idx.id/primary/%' then
    raise exception 'official Block IDX URL required';
  end if;
  v_response:=extensions.http_get(p_url);
  if v_response.status<>200 then
    raise exception 'Block IDX HTTP status %',v_response.status;
  end if;
  begin
    v_payload:=v_response.content::jsonb;
  exception when others then
    raise exception 'Block IDX returned invalid JSON';
  end;
  return v_payload;
end $fn$;

create or replace function public.cak_idx_ingest_official_market_day_v2(p_trade_date date)
returns jsonb
language plpgsql
security definer
set search_path=pg_catalog,public,extensions
as $fn$
declare
  v_key text:=to_char(p_trade_date,'YYYYMMDD');
  v_stock_url text;
  v_index_url text;
  v_broker_url text;
  v_stock jsonb;
  v_index jsonb;
  v_broker jsonb;
  v_stock_rows integer:=0;
  v_index_rows integer:=0;
  v_broker_rows integer:=0;
begin
  perform pg_advisory_xact_lock(hashtext('emir-block-idx-'||p_trade_date::text));
  v_stock_url:='https://block.idx.id/primary/TradingSummary/GetStockSummary?length=2000&start=0&date='||v_key;
  v_index_url:='https://block.idx.id/primary/TradingSummary/GetIndexSummary?length=1000&start=0&date='||v_key;
  v_broker_url:='https://block.idx.id/primary/TradingSummary/GetBrokerSummary?length=200&start=0&date='||v_key;
  v_stock:=public.cak_idx_http_json_v1(v_stock_url);
  v_index:=public.cak_idx_http_json_v1(v_index_url);
  v_broker:=public.cak_idx_http_json_v1(v_broker_url);

  insert into public.cak_idx_market_daily(
    trade_date,ticker,stock_name,previous,open,high,low,close,change,
    volume,traded_value,frequency,foreign_buy,foreign_sell,foreign_net,
    listed_shares,tradable_shares,bid,offer,bid_volume,offer_volume,
    non_regular_volume,non_regular_value,source_url,payload_hash,
    source_verified,provenance_state
  )
  select
    p_trade_date,upper(x->>'StockCode'),nullif(x->>'StockName',''),
    public.cak_idx_safe_numeric_v1(x->>'Previous'),
    public.cak_idx_safe_numeric_v1(x->>'OpenPrice'),
    public.cak_idx_safe_numeric_v1(x->>'High'),
    public.cak_idx_safe_numeric_v1(x->>'Low'),
    public.cak_idx_safe_numeric_v1(x->>'Close'),
    public.cak_idx_safe_numeric_v1(x->>'Change'),
    coalesce(public.cak_idx_safe_numeric_v1(x->>'Volume'),0),
    coalesce(public.cak_idx_safe_numeric_v1(x->>'Value'),0),
    coalesce(public.cak_idx_safe_numeric_v1(x->>'Frequency'),0),
    coalesce(public.cak_idx_safe_numeric_v1(x->>'ForeignBuy'),0),
    coalesce(public.cak_idx_safe_numeric_v1(x->>'ForeignSell'),0),
    coalesce(public.cak_idx_safe_numeric_v1(x->>'ForeignBuy'),0)
      -coalesce(public.cak_idx_safe_numeric_v1(x->>'ForeignSell'),0),
    public.cak_idx_safe_numeric_v1(x->>'ListedShares'),
    public.cak_idx_safe_numeric_v1(x->>'TradebleShares'),
    public.cak_idx_safe_numeric_v1(x->>'Bid'),
    public.cak_idx_safe_numeric_v1(x->>'Offer'),
    public.cak_idx_safe_numeric_v1(x->>'BidVolume'),
    public.cak_idx_safe_numeric_v1(x->>'OfferVolume'),
    public.cak_idx_safe_numeric_v1(x->>'NonRegularVolume'),
    public.cak_idx_safe_numeric_v1(x->>'NonRegularValue'),
    v_stock_url,encode(extensions.digest(convert_to(x::text,'UTF8'),'sha256'),'hex'),
    true,'VERIFIED_OFFICIAL_BLOCK_IDX_STOCK_SUMMARY'
  from jsonb_array_elements(coalesce(v_stock->'data','[]'::jsonb)) x
  where substring(x->>'Date' from 1 for 10)=p_trade_date::text
    and upper(coalesce(x->>'StockCode','')) ~ '^[A-Z0-9.-]{2,12}$'
    and coalesce(public.cak_idx_safe_numeric_v1(x->>'Volume'),0)>=0
    and coalesce(public.cak_idx_safe_numeric_v1(x->>'Value'),0)>=0
    and coalesce(public.cak_idx_safe_numeric_v1(x->>'Frequency'),0)>=0
  on conflict(trade_date,ticker) do update set
    stock_name=excluded.stock_name,previous=excluded.previous,open=excluded.open,
    high=excluded.high,low=excluded.low,close=excluded.close,change=excluded.change,
    volume=excluded.volume,traded_value=excluded.traded_value,frequency=excluded.frequency,
    foreign_buy=excluded.foreign_buy,foreign_sell=excluded.foreign_sell,
    foreign_net=excluded.foreign_net,listed_shares=excluded.listed_shares,
    tradable_shares=excluded.tradable_shares,bid=excluded.bid,offer=excluded.offer,
    bid_volume=excluded.bid_volume,offer_volume=excluded.offer_volume,
    non_regular_volume=excluded.non_regular_volume,
    non_regular_value=excluded.non_regular_value,source_url=excluded.source_url,
    payload_hash=excluded.payload_hash,source_verified=true,
    provenance_state=excluded.provenance_state,ingested_at=clock_timestamp();
  get diagnostics v_stock_rows=row_count;

  insert into public.cak_idx_index_daily(
    trade_date,index_code,previous,highest,lowest,close,number_of_stock,change,
    volume,traded_value,frequency,market_capital,source_url,payload_hash,source_verified
  )
  select
    p_trade_date,upper(x->>'IndexCode'),
    public.cak_idx_safe_numeric_v1(x->>'Previous'),
    public.cak_idx_safe_numeric_v1(x->>'Highest'),
    public.cak_idx_safe_numeric_v1(x->>'Lowest'),
    public.cak_idx_safe_numeric_v1(x->>'Close'),
    public.cak_idx_safe_numeric_v1(x->>'NumberOfStock'),
    public.cak_idx_safe_numeric_v1(x->>'Change'),
    public.cak_idx_safe_numeric_v1(x->>'Volume'),
    public.cak_idx_safe_numeric_v1(x->>'Value'),
    public.cak_idx_safe_numeric_v1(x->>'Frequency'),
    public.cak_idx_safe_numeric_v1(x->>'MarketCapital'),
    v_index_url,encode(extensions.digest(convert_to(x::text,'UTF8'),'sha256'),'hex'),true
  from jsonb_array_elements(coalesce(v_index->'data','[]'::jsonb)) x
  where substring(x->>'Date' from 1 for 10)=p_trade_date::text
    and nullif(x->>'IndexCode','') is not null
  on conflict(trade_date,index_code) do update set
    previous=excluded.previous,highest=excluded.highest,lowest=excluded.lowest,
    close=excluded.close,number_of_stock=excluded.number_of_stock,change=excluded.change,
    volume=excluded.volume,traded_value=excluded.traded_value,
    frequency=excluded.frequency,market_capital=excluded.market_capital,
    source_url=excluded.source_url,payload_hash=excluded.payload_hash,
    source_verified=true,ingested_at=clock_timestamp();
  get diagnostics v_index_rows=row_count;

  insert into public.cak_idx_broker_market_daily(
    trade_date,broker_code,broker_name,traded_value,volume,frequency,semantic_scope,
    source_url,payload_hash,source_verified
  )
  select
    p_trade_date,upper(x->>'IDFirm'),nullif(x->>'FirmName',''),
    coalesce(public.cak_idx_safe_numeric_v1(x->>'Value'),0),
    coalesce(public.cak_idx_safe_numeric_v1(x->>'Volume'),0),
    coalesce(public.cak_idx_safe_numeric_v1(x->>'Frequency'),0),
    'MARKET_WIDE_NO_TICKER_BUY_SELL_SPLIT',
    v_broker_url,encode(extensions.digest(convert_to(x::text,'UTF8'),'sha256'),'hex'),true
  from jsonb_array_elements(coalesce(v_broker->'data','[]'::jsonb)) x
  where substring(x->>'Date' from 1 for 10)=p_trade_date::text
    and nullif(x->>'IDFirm','') is not null
  on conflict(trade_date,broker_code) do update set
    broker_name=excluded.broker_name,traded_value=excluded.traded_value,
    volume=excluded.volume,frequency=excluded.frequency,
    semantic_scope=excluded.semantic_scope,source_url=excluded.source_url,
    payload_hash=excluded.payload_hash,source_verified=true,
    ingested_at=clock_timestamp();
  get diagnostics v_broker_rows=row_count;

  insert into public.cak_idx_payload_manifest(
    provider,endpoint_key,target_date,payload_hash,source_url,rows_received,
    rows_accepted,validation_state,producer_version
  )
  values
    ('BLOCK_IDX','stock_summary',p_trade_date,
      encode(extensions.digest(convert_to(v_stock::text,'UTF8'),'sha256'),'hex'),v_stock_url,
      coalesce((v_stock->>'recordsTotal')::integer,0),v_stock_rows,
      case when v_stock_rows>0 then 'VALID' else 'NO_DATA' end,'EMIR_BLOCK_IDX_DB_EOD_V2'),
    ('BLOCK_IDX','index_summary',p_trade_date,
      encode(extensions.digest(convert_to(v_index::text,'UTF8'),'sha256'),'hex'),v_index_url,
      coalesce((v_index->>'recordsTotal')::integer,0),v_index_rows,
      case when v_index_rows>0 then 'VALID' else 'NO_DATA' end,'EMIR_BLOCK_IDX_DB_EOD_V2'),
    ('BLOCK_IDX','broker_summary',p_trade_date,
      encode(extensions.digest(convert_to(v_broker::text,'UTF8'),'sha256'),'hex'),v_broker_url,
      coalesce((v_broker->>'recordsTotal')::integer,0),v_broker_rows,
      case when v_broker_rows>0 then 'VALID' else 'NO_DATA' end,'EMIR_BLOCK_IDX_DB_EOD_V2')
  on conflict(provider,endpoint_key,target_date,payload_hash) do update set
    rows_received=excluded.rows_received,rows_accepted=excluded.rows_accepted,
    validation_state=excluded.validation_state,producer_version=excluded.producer_version,
    fetched_at=clock_timestamp();

  return jsonb_build_object(
    'trade_date',p_trade_date,'stock_rows',v_stock_rows,'index_rows',v_index_rows,
    'broker_rows',v_broker_rows,
    'state',case when v_stock_rows>0 and v_index_rows>0 then 'VALID' else 'NO_DATA' end
  );
exception when others then
  insert into public.cak_idx_ingestion_failures(
    endpoint_key,target_date,failure_class,message,retryable,last_seen_at,attempt_count
  ) values(
    'server_market_bundle',p_trade_date,sqlstate,left(sqlerrm,1000),true,clock_timestamp(),1
  )
  on conflict(endpoint_key,target_date,failure_class,message) do update set
    last_seen_at=clock_timestamp(),attempt_count=public.cak_idx_ingestion_failures.attempt_count+1;
  raise;
end $fn$;

create or replace function public.cak_idx_run_eod_v2(
  p_trade_date date default ((now() at time zone 'Asia/Jakarta')::date)
)
returns jsonb
language plpgsql
security definer
set search_path=pg_catalog,public
as $fn$
declare v_run_key text; v_market jsonb; v_rank jsonb; v_storage jsonb;
begin
  v_run_key:='EMIR_BLOCK_IDX_DB_EOD_V2:'||p_trade_date::text;
  insert into public.cak_idx_ingestion_runs(
    run_key,mode,requested_from,requested_to,status,producer_version
  ) values(v_run_key,'daily',p_trade_date,p_trade_date,'RUNNING','EMIR_BLOCK_IDX_DB_EOD_V2')
  on conflict(run_key) do update set status='RUNNING',started_at=clock_timestamp(),completed_at=null;

  v_storage:=public.cak_prune_storage_v1(p_trade_date);
  if v_storage->>'state'='HARD_STOP' then
    raise exception 'storage hard-stop before EOD';
  end if;
  v_market:=public.cak_idx_ingest_official_market_day_v2(p_trade_date);
  if v_market->>'state'='VALID' then
    v_rank:=public.cak_refresh_idx_ranking_v1(p_trade_date);
  end if;
  v_storage:=public.cak_prune_storage_v1(p_trade_date);
  update public.cak_idx_ingestion_runs set
    status=case when v_market->>'state'='VALID' then 'COMPLETED' else 'SOURCE_NOT_READY' end,
    core_sessions_attempted=1,
    core_sessions_loaded=case when v_market->>'state'='VALID' then 1 else 0 end,
    rows_persisted=coalesce((v_market->>'stock_rows')::integer,0)
      +coalesce((v_market->>'index_rows')::integer,0)
      +coalesce((v_market->>'broker_rows')::integer,0),
    detail=jsonb_build_object('market',v_market,'ranking',v_rank,'storage',v_storage),
    completed_at=clock_timestamp()
  where run_key=v_run_key;
  return jsonb_build_object('market',v_market,'ranking',v_rank,'storage',v_storage);
exception when others then
  update public.cak_idx_ingestion_runs set status='FAILED',failures=failures+1,
    detail=jsonb_build_object('sqlstate',sqlstate,'error',left(sqlerrm,1000)),
    completed_at=clock_timestamp()
  where run_key=v_run_key;
  raise;
end $fn$;

revoke all on function public.cak_idx_safe_numeric_v1(text),
  public.cak_idx_http_json_v1(text),
  public.cak_idx_ingest_official_market_day_v2(date),
  public.cak_idx_run_eod_v2(date)
from public,anon,authenticated;
grant execute on function public.cak_idx_safe_numeric_v1(text),
  public.cak_idx_http_json_v1(text),
  public.cak_idx_ingest_official_market_day_v2(date),
  public.cak_idx_run_eod_v2(date)
to service_role;

do $fn$
declare v_job bigint;
begin
  select jobid into v_job from cron.job where jobname='emir-block-idx-eod-1745-wib';
  if v_job is not null then perform cron.unschedule(v_job); end if;
  select jobid into v_job from cron.job where jobname='emir-block-idx-eod-retry-1845-wib';
  if v_job is not null then perform cron.unschedule(v_job); end if;
  perform cron.schedule(
    'emir-block-idx-eod-1745-wib','45 10 * * 1-5',
    $command$select public.cak_idx_run_eod_v2(((now() at time zone 'Asia/Jakarta')::date));$command$
  );
  perform cron.schedule(
    'emir-block-idx-eod-retry-1845-wib','45 11 * * 1-5',
    $command$select public.cak_idx_run_eod_v2(((now() at time zone 'Asia/Jakarta')::date));$command$
  );
end $fn$;
