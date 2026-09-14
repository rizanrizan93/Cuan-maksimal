-- EMIR official Block IDX database-first evidence plane.
-- Additive, EMIR-private (cak_idx_*), idempotent and fail-closed.

create table if not exists public.cak_idx_endpoint_catalog (
  endpoint_key text primary key,
  family text not null,
  path text not null,
  acquisition_class text not null check (acquisition_class in ('A','B','C','D','E','F','G')),
  cadence text not null,
  history_supported boolean not null default false,
  scoring_role text not null,
  route_state text not null default 'VERIFIED' check (route_state in ('VERIFIED','CATALOG_ONLY','DISABLED')),
  enabled boolean not null default true,
  updated_at timestamptz not null default now()
);

insert into public.cak_idx_endpoint_catalog
(endpoint_key,family,path,acquisition_class,cadence,history_supported,scoring_role,route_state,enabled)
values
('stock_summary','MARKET_DAILY','/primary/TradingSummary/GetStockSummary','A','EOD',true,'PRICE_LIQUIDITY_FOREIGN','VERIFIED',true),
('index_summary','INDEX_DAILY','/primary/TradingSummary/GetIndexSummary','A','EOD',true,'MARKET_SECTOR_REGIME','VERIFIED',true),
('broker_summary','BROKER_MARKET_DAILY','/primary/TradingSummary/GetBrokerSummary','D','EOD',true,'MARKET_ONLY_NOT_TICKER_FLOW','VERIFIED',true),
('companies','COMPANY_REFERENCE','/primary/ListedCompany/GetCompanyProfiles','C','MONTHLY',false,'UNIVERSE_IDENTITY','VERIFIED',true),
('company_profile','COMPANY_DETAIL','/primary/ListedCompany/GetCompanyProfilesDetail','C','ROTATING_WEEKLY',false,'OWNERSHIP_CONTROLLER','VERIFIED',true),
('financial_report','FINANCIAL_REPORT','/primary/ListedCompany/GetFinancialReport','B','EVENT',true,'OFFICIAL_FUNDAMENTAL','VERIFIED',true),
('announcements','ANNOUNCEMENT','/primary/NewsAnnouncement/GetAllAnnouncement','B','EOD_DELTA',true,'CATALYST_RISK','VERIFIED',true),
('company_announcements','ANNOUNCEMENT','/primary/ListedCompany/GetProfileAnnouncement','D','FINALIST',true,'CATALYST_CONFIRMATION','VERIFIED',true),
('uma','RISK_EVENT','/primary/NewsAnnouncement/GetUma','B','EOD_DELTA',true,'EXECUTION_GUARD','VERIFIED',true),
('suspension','RISK_EVENT','/primary/NewsAnnouncement/GetSuspension','B','EOD_DELTA',true,'HARD_BLOCK','VERIFIED',true),
('issued_history','CAPITAL_ACTION','/primary/ListingActivity/GetIssuedHistory','B','EOD_DELTA',true,'DILUTION_CAPITAL_ACTION','VERIFIED',true),
('trading_info','TRADING_DETAIL','/primary/ListedCompany/GetTradingInfoSS','D','FINALIST',true,'FINALIST_DIAGNOSTIC','VERIFIED',true),
('top_movers','REDUNDANT_DERIVED', '','E','NONE',false,'DERIVE_FROM_STOCK_SUMMARY','CATALOG_ONLY',false),
('market_activity','REDUNDANT_EVENT', '','E','NONE',true,'USE_UMA_AND_SUSPENSION','CATALOG_ONLY',false),
('news','REDUNDANT_NEWS', '','E','NONE',true,'USE_ANNOUNCEMENTS','CATALOG_ONLY',false),
('ownership_files','OWNERSHIP_FILE_INDEX', '','B','EVENT',true,'KSEI_REPORTED_OWNERSHIP','CATALOG_ONLY',false),
('calendar','TRADING_CALENDAR', '','A','DAILY',true,'SESSION_TRUTH','CATALOG_ONLY',false),
('ipo','CAPITAL_ACTION', '','B','EVENT',true,'IPO_PIPELINE','CATALOG_ONLY',false),
('lendable_stock','RISK_EVENT', '','D','WEEKLY',true,'LENDABLE_CONTEXT','CATALOG_ONLY',false),
('margin_summary','RISK_EVENT', '','D','PERIOD_CHANGE',true,'MARGIN_CONTEXT','CATALOG_ONLY',false),
('index_constituent','INDEX_MEMBERSHIP', '','D','MONTHLY',true,'INDEX_MEMBERSHIP','CATALOG_ONLY',false),
('participants','PARTICIPANT_REFERENCE', '','C','MONTHLY',false,'REFERENCE_ONLY','CATALOG_ONLY',false),
('reference','REFERENCE', '','C','MONTHLY',false,'SECTOR_BOARD_MARKET_TIME','CATALOG_ONLY',false),
('additional_listings','CAPITAL_ACTION', '','B','EVENT',true,'DILUTION','CATALOG_ONLY',false),
('delistings','CAPITAL_ACTION', '','B','EVENT',true,'DELISTING','CATALOG_ONLY',false),
('dividends','CAPITAL_ACTION', '','B','EVENT',true,'DIVIDEND','CATALOG_ONLY',false),
('new_listings','CAPITAL_ACTION', '','B','EVENT',true,'LISTING','CATALOG_ONLY',false),
('rights_offerings','CAPITAL_ACTION', '','B','EVENT',true,'DILUTION','CATALOG_ONLY',false),
('stock_splits','CAPITAL_ACTION', '','B','EVENT',true,'PRICE_ADJUSTMENT','CATALOG_ONLY',false),
('derivatives','OUT_OF_SCOPE', '','G','NONE',false,'NONE','DISABLED',false),
('bonds','OUT_OF_SCOPE', '','G','NONE',false,'NONE','DISABLED',false),
('structured_warrants','OUT_OF_SCOPE', '','G','NONE',false,'NONE','DISABLED',false)
on conflict(endpoint_key) do update set
  family=excluded.family,path=excluded.path,acquisition_class=excluded.acquisition_class,
  cadence=excluded.cadence,history_supported=excluded.history_supported,
  scoring_role=excluded.scoring_role,route_state=excluded.route_state,
  enabled=excluded.enabled,updated_at=now();

create table if not exists public.cak_idx_ingestion_runs (
  run_key text primary key,
  mode text not null,
  requested_from date,
  requested_to date,
  status text not null default 'RUNNING',
  core_sessions_attempted integer not null default 0,
  core_sessions_loaded integer not null default 0,
  rows_persisted bigint not null default 0,
  failures integer not null default 0,
  detail jsonb not null default '{}'::jsonb,
  started_at timestamptz not null default now(),
  completed_at timestamptz,
  producer_version text not null
);

create table if not exists public.cak_idx_payload_manifest (
  provider text not null,
  endpoint_key text not null references public.cak_idx_endpoint_catalog(endpoint_key),
  target_date date not null,
  payload_hash text not null,
  source_url text not null,
  rows_received integer not null default 0,
  rows_accepted integer not null default 0,
  validation_state text not null,
  producer_version text not null,
  fetched_at timestamptz not null default now(),
  primary key(provider,endpoint_key,target_date,payload_hash)
);

create table if not exists public.cak_idx_ingestion_failures (
  endpoint_key text not null,
  target_date date not null,
  failure_class text not null,
  message text not null,
  retryable boolean not null default false,
  first_seen_at timestamptz not null default now(),
  last_seen_at timestamptz not null default now(),
  attempt_count integer not null default 1,
  primary key(endpoint_key,target_date,failure_class,message)
);

create table if not exists public.cak_idx_market_daily (
  trade_date date not null,
  ticker text not null,
  stock_name text,
  previous numeric, open numeric, high numeric, low numeric, close numeric, change numeric,
  volume numeric not null default 0,
  traded_value numeric not null default 0,
  frequency numeric not null default 0,
  foreign_buy numeric not null default 0,
  foreign_sell numeric not null default 0,
  foreign_net numeric not null default 0,
  listed_shares numeric, tradable_shares numeric,
  bid numeric, offer numeric, bid_volume numeric, offer_volume numeric,
  non_regular_volume numeric, non_regular_value numeric,
  source_url text not null, payload_hash text not null,
  source_verified boolean not null default true,
  provenance_state text not null,
  ingested_at timestamptz not null default now(),
  primary key(trade_date,ticker),
  constraint cak_idx_market_nonnegative check(volume>=0 and traded_value>=0 and frequency>=0 and foreign_buy>=0 and foreign_sell>=0)
);
create index if not exists cak_idx_market_ticker_date_idx on public.cak_idx_market_daily(ticker,trade_date desc);
create index if not exists cak_idx_market_date_value_idx on public.cak_idx_market_daily(trade_date desc,traded_value desc);

create table if not exists public.cak_idx_index_daily (
  trade_date date not null,
  index_code text not null,
  previous numeric, highest numeric, lowest numeric, close numeric,
  number_of_stock numeric, change numeric, volume numeric, traded_value numeric,
  frequency numeric, market_capital numeric,
  source_url text not null, payload_hash text not null, source_verified boolean not null default true,
  ingested_at timestamptz not null default now(),
  primary key(trade_date,index_code)
);
create index if not exists cak_idx_index_code_date_idx on public.cak_idx_index_daily(index_code,trade_date desc);

create table if not exists public.cak_idx_broker_market_daily (
  trade_date date not null,
  broker_code text not null,
  broker_name text,
  traded_value numeric not null default 0,
  volume numeric not null default 0,
  frequency numeric not null default 0,
  semantic_scope text not null check(semantic_scope='MARKET_WIDE_NO_TICKER_BUY_SELL_SPLIT'),
  source_url text not null, payload_hash text not null, source_verified boolean not null default true,
  ingested_at timestamptz not null default now(),
  primary key(trade_date,broker_code)
);
comment on table public.cak_idx_broker_market_daily is
'Market-wide exchange member totals only. Never use as ticker broker flow or bandar identity.';

create table if not exists public.cak_idx_events (
  event_family text not null,
  event_type text not null,
  ticker text,
  event_date date not null,
  publication_date date,
  source_ref text not null,
  title text,
  source_url text not null,
  payload_hash text not null,
  source_verified boolean not null default true,
  raw_payload jsonb not null default '{}'::jsonb,
  ingested_at timestamptz not null default now(),
  primary key(event_family,event_type,event_date,source_ref,payload_hash)
);
create index if not exists cak_idx_events_ticker_date_idx on public.cak_idx_events(ticker,event_date desc);
create index if not exists cak_idx_events_family_date_idx on public.cak_idx_events(event_family,event_date desc);

create table if not exists public.cak_idx_company_snapshot (
  ticker text not null,
  observed_on date not null,
  company_name text,
  sector text,
  subsector text,
  listing_date date,
  source_url text not null,
  payload_hash text not null,
  source_verified boolean not null default true,
  raw_payload jsonb not null default '{}'::jsonb,
  ingested_at timestamptz not null default now(),
  primary key(ticker,observed_on)
);

create table if not exists public.cak_idx_fundamental_snapshot (
  ticker text not null,
  period_end date not null,
  observed_on date not null,
  period_type text,
  revenue numeric,
  revenue_growth_yoy_pct numeric,
  net_income numeric,
  earnings_growth_yoy_pct numeric,
  net_margin_pct numeric,
  roe_pct numeric,
  roa_pct numeric,
  ocf numeric,
  fcf numeric,
  cash numeric,
  debt numeric,
  debt_to_equity numeric,
  current_ratio numeric,
  cash_to_debt_ratio numeric,
  source_url text not null,
  coverage_pct numeric,
  payload_hash text not null,
  source_verified boolean not null default true,
  raw_payload jsonb not null default '{}'::jsonb,
  ingested_at timestamptz not null default now(),
  primary key(ticker,period_end,observed_on,payload_hash)
);
create index if not exists cak_idx_fundamental_latest_idx on public.cak_idx_fundamental_snapshot(ticker,period_end desc,observed_on desc);

create table if not exists public.cak_idx_rank_daily (
  rank_date date not null,
  ticker text not null,
  overall_rank integer,
  execution_rank integer,
  emir_score numeric not null,
  execution_score numeric not null,
  fundamental_score numeric not null,
  smart_money_score numeric not null,
  momentum_score numeric not null,
  liquidity_score numeric not null,
  regime_score numeric not null,
  risk_score numeric not null,
  observations integer not null,
  close numeric,
  return_5d_pct numeric,
  return_20d_pct numeric,
  return_60d_pct numeric,
  foreign_net_20d numeric,
  foreign_positive_days_20d integer,
  adtv_20d numeric,
  frequency_20d numeric,
  spread_pct numeric,
  entry_price numeric,
  stop_loss numeric,
  structural_tp1 numeric,
  rr_tp1 numeric,
  official_fundamental_coverage_pct numeric,
  active_suspension boolean not null default false,
  recent_uma boolean not null default false,
  recent_dilution boolean not null default false,
  execution_eligible boolean not null default false,
  blocker text,
  feature_contract text not null default 'EMIR_BLOCK_IDX_FEATURE_V1',
  computed_at timestamptz not null default now(),
  primary key(rank_date,ticker)
);
create index if not exists cak_idx_rank_daily_overall_idx on public.cak_idx_rank_daily(rank_date desc,overall_rank);
create index if not exists cak_idx_rank_daily_execution_idx on public.cak_idx_rank_daily(rank_date desc,execution_rank) where execution_eligible;

create table if not exists public.cak_idx_top3_execution (
  rank_date date not null,
  execution_rank integer not null check(execution_rank between 1 and 3),
  ticker text not null,
  execution_score numeric not null,
  emir_score numeric not null,
  entry_price numeric not null,
  stop_loss numeric not null,
  tp1 numeric not null,
  tp2 numeric not null,
  rr_tp1 numeric not null,
  risk_per_share numeric not null,
  geometry_state text not null,
  decision_state text not null default 'EXECUTION_READY',
  source_state text not null default 'DATABASE_ONLY_OFFICIAL_BLOCK_IDX',
  feature_contract text not null default 'EMIR_BLOCK_IDX_FEATURE_V1',
  computed_at timestamptz not null default now(),
  primary key(rank_date,execution_rank),
  unique(rank_date,ticker)
);

create or replace function public.cak_refresh_idx_ranking_v1(
  p_as_of date default ((now() at time zone 'Asia/Jakarta')::date)
)
returns jsonb
language plpgsql
security definer
set search_path=pg_catalog,public
as $fn$
declare
  v_eod date;
  v_rows integer;
  v_eligible integer;
  v_top3 integer;
begin
  select max(trade_date) into v_eod
  from public.cak_idx_market_daily
  where trade_date<=p_as_of and source_verified;

  if v_eod is null then
    raise exception 'EMIR Block IDX market source not ready for %',p_as_of;
  end if;

  delete from public.cak_idx_rank_daily where rank_date=v_eod;
  delete from public.cak_idx_top3_execution where rank_date=v_eod;

  with sessions as (
    select trade_date,row_number() over(order by trade_date desc) rn
    from (select distinct trade_date from public.cak_idx_market_daily where trade_date<=v_eod and source_verified order by trade_date desc limit 130) s
  ), bars as (
    select m.*,s.rn
    from public.cak_idx_market_daily m join sessions s using(trade_date)
    where m.source_verified and m.close>0
  ), agg as (
    select
      ticker,count(*)::int observations,
      (array_agg(close order by trade_date desc))[1] close,
      (array_agg(close order by trade_date desc))[6] close_5,
      (array_agg(close order by trade_date desc))[21] close_20,
      (array_agg(close order by trade_date desc))[61] close_60,
      avg(close) filter(where rn<=20) ma20,
      avg(close) filter(where rn<=60) ma60,
      max(high) filter(where rn between 2 and 61) resistance60,
      min(low) filter(where rn<=20) support20,
      avg(greatest(high-low,abs(high-previous),abs(low-previous))) filter(where rn<=14) atr14,
      sum(foreign_net) filter(where rn<=20) foreign_net20,
      count(*) filter(where rn<=20 and foreign_net>0)::int foreign_positive20,
      avg(traded_value) filter(where rn<=20) adtv20,
      avg(frequency) filter(where rn<=20) freq20,
      avg(case when listed_shares>0 then volume/listed_shares else null end) filter(where rn<=20) turnover20,
      stddev_samp(case when previous>0 then close/previous-1 else null end) filter(where rn<=20) volatility20,
      (array_agg(bid order by trade_date desc))[1] bid,
      (array_agg(offer order by trade_date desc))[1] offer
    from bars group by ticker
  ), measures as (
    select a.*,
      100*(close/nullif(close_5,0)-1) ret5,
      100*(close/nullif(close_20,0)-1) ret20,
      100*(close/nullif(close_60,0)-1) ret60,
      foreign_net20/nullif(sum(abs(foreign_net20)) over(),0) foreign_share,
      case when close>0 and offer>0 and bid>0 and offer>=bid then 100*(offer-bid)/close end spread,
      case when resistance60>close and atr14>0
        then least(support20*0.985,close-1.25*atr14)
        else null end sl,
      case when resistance60>close then resistance60 end structural_tp1
    from agg a
  ), cross as (
    select m.*,
      100*percent_rank() over(order by coalesce(foreign_share,-1)) foreign_rank,
      100*percent_rank() over(order by foreign_positive20) persistence_rank,
      100*percent_rank() over(order by ln(1+greatest(coalesce(adtv20,0),0))) value_rank,
      100*percent_rank() over(order by coalesce(freq20,0)) frequency_rank,
      100*percent_rank() over(order by coalesce(ret20,-999)) ret20_rank,
      100*percent_rank() over(order by coalesce(ret60,-999)) ret60_rank
    from measures m where observations>=20
  ), latest_f as (
    select distinct on(ticker) *
    from public.cak_idx_fundamental_snapshot
    where source_verified and observed_on<=v_eod
    order by ticker,period_end desc,observed_on desc
  ), event_state as (
    select c.ticker,
      exists(
        select 1 from public.cak_idx_events e
        where e.ticker=c.ticker and e.event_type='SUSPEND' and e.event_date<=v_eod
          and not exists(select 1 from public.cak_idx_events u where u.ticker=e.ticker and u.event_type='UNSUSPEND' and u.event_date>=e.event_date and u.event_date<=v_eod)
      ) active_suspension,
      exists(select 1 from public.cak_idx_events e where e.ticker=c.ticker and e.event_type='UMA' and e.event_date between v_eod-14 and v_eod) recent_uma,
      exists(select 1 from public.cak_idx_events e where e.ticker=c.ticker and e.event_family='ISSUED_HISTORY' and e.event_type in ('HMETD','TANPA_HMETD','PRIVATE_PLACEMENT','RIGHTS_ISSUE','WARAN') and e.event_date between v_eod-45 and v_eod) recent_dilution
    from cross c
  ), ihsg as (
    select
      (array_agg(close order by trade_date desc))[1] current_close,
      (array_agg(close order by trade_date desc))[21] prior20
    from public.cak_idx_index_daily
    where index_code='COMPOSITE' and trade_date<=v_eod and source_verified
  ), scored0 as (
    select c.*,f.coverage_pct,
      least(100,greatest(0,
        0.15*(50+least(100,greatest(-50,coalesce(f.revenue_growth_yoy_pct,0)))/2)+
        0.15*(50+least(100,greatest(-50,coalesce(f.earnings_growth_yoy_pct,0)))/2)+
        0.15*least(100,greatest(0,coalesce(f.roe_pct,0)*4))+
        0.10*least(100,greatest(0,coalesce(f.roa_pct,0)*7))+
        0.10*least(100,greatest(0,coalesce(f.net_margin_pct,0)*4))+
        0.10*(case when f.ocf>0 and f.net_income>0 then 100 when f.ocf>0 then 65 else 15 end)+
        0.10*(case when f.fcf>0 then 100 when f.fcf is null then 40 else 10 end)+
        0.10*(case when f.debt_to_equity is null then 45 when f.debt_to_equity<=0.5 then 100 when f.debt_to_equity<=1 then 70 when f.debt_to_equity<=2 then 35 else 5 end)+
        0.05*(case when f.current_ratio>=1.5 then 100 when f.current_ratio>=1 then 65 when f.current_ratio is null then 45 else 20 end)
      )) fundamental_score,
      least(100,greatest(0,0.45*foreign_rank+0.30*persistence_rank+0.25*least(100,greatest(0,50+5000*coalesce(turnover20,0))))) smart_score,
      least(100,greatest(0,0.35*ret20_rank+0.25*ret60_rank+20*(case when close>ma20 and ma20>ma60 then 1 else 0 end)+20*least(1,close/nullif(resistance60,0)))) momentum_score,
      least(100,greatest(0,0.60*value_rank+0.20*frequency_rank+0.20*least(1,observations/100.0)*100)) liquidity_score,
      case when i.current_close>i.prior20 then 75 when i.current_close is null or i.prior20 is null then 45 else 25 end regime_score,
      least(100,greatest(0,100-35*e.active_suspension::int-15*e.recent_uma::int-15*e.recent_dilution::int-10*greatest(coalesce(c.spread,0)-1,0))) risk_score,
      e.active_suspension,e.recent_uma,e.recent_dilution
    from cross c
    left join latest_f f using(ticker)
    left join event_state e using(ticker)
    cross join ihsg i
  ), scored as (
    select s.*,
      0.30*fundamental_score+0.25*smart_score+0.20*momentum_score+0.10*liquidity_score+0.10*regime_score+0.05*risk_score emir_score,
      0.35*smart_score+0.30*momentum_score+0.20*liquidity_score+0.15*risk_score execution_score,
      case when sl>0 and structural_tp1>close and close>sl then (structural_tp1-close)/(close-sl) end rr1,
      (
        observations>=80 and adtv20>=1000000000 and close>ma20 and ma20>ma60
        and foreign_positive20>=11 and foreign_net20>0
        and not active_suspension and not recent_dilution
        and coalesce(spread,99)<=2 and coverage_pct>=50
        and sl>0 and structural_tp1>close
      ) base_eligible
    from scored0 s
  ), ranked as (
    select s.*,
      row_number() over(order by emir_score desc,ticker)::int overall_rank,
      case when base_eligible and rr1>=1.8
        then row_number() over(partition by (base_eligible and rr1>=1.8) order by execution_score desc,emir_score desc,ticker)::int
      end execution_rank
    from scored s
  )
  insert into public.cak_idx_rank_daily(
    rank_date,ticker,overall_rank,execution_rank,emir_score,execution_score,fundamental_score,smart_money_score,
    momentum_score,liquidity_score,regime_score,risk_score,observations,close,
    return_5d_pct,return_20d_pct,return_60d_pct,foreign_net_20d,foreign_positive_days_20d,
    adtv_20d,frequency_20d,spread_pct,entry_price,stop_loss,structural_tp1,rr_tp1,official_fundamental_coverage_pct,
    active_suspension,recent_uma,recent_dilution,execution_eligible,blocker
  )
  select v_eod,ticker,overall_rank,
    case when base_eligible and rr1>=1.8 then execution_rank end,
    round(emir_score,4),round(execution_score,4),round(fundamental_score,4),round(smart_score,4),round(momentum_score,4),
    round(liquidity_score,4),round(regime_score,4),round(risk_score,4),observations,close,
    round(ret5,4),round(ret20,4),round(ret60,4),foreign_net20,foreign_positive20,
    adtv20,freq20,spread,close,sl,structural_tp1,rr1,coverage_pct,active_suspension,recent_uma,recent_dilution,
    base_eligible and rr1>=1.8,
    case
      when observations<80 then 'INSUFFICIENT_6M_HISTORY'
      when adtv20<1000000000 then 'LIQUIDITY_BELOW_1B'
      when active_suspension then 'ACTIVE_SUSPENSION'
      when recent_dilution then 'RECENT_DILUTION'
      when coalesce(coverage_pct,0)<50 then 'OFFICIAL_FUNDAMENTAL_COVERAGE'
      when not(close>ma20 and ma20>ma60) then 'NO_BULLISH_HTF_STACK'
      when not(foreign_positive20>=11 and foreign_net20>0) then 'NO_SILENT_ACCUMULATION'
      when coalesce(spread,99)>2 then 'SPREAD_TOO_WIDE'
      when sl is null or structural_tp1 is null then 'INVALID_STRUCTURAL_GEOMETRY'
      when coalesce(rr1,0)<1.8 then 'RR_BELOW_1_8'
      else null end
  from ranked;

  insert into public.cak_idx_top3_execution(
    rank_date,execution_rank,ticker,execution_score,emir_score,entry_price,stop_loss,tp1,tp2,
    rr_tp1,risk_per_share,geometry_state
  )
  select v_eod,row_number() over(order by r.execution_score desc,r.emir_score desc,r.ticker)::int,
    r.ticker,r.execution_score,r.emir_score,r.entry_price,r.stop_loss,r.structural_tp1,
    r.entry_price+3*(r.entry_price-r.stop_loss),r.rr_tp1,
    r.entry_price-r.stop_loss,'OBSERVED_SUPPORT_RESISTANCE_ATR'
  from public.cak_idx_rank_daily r
  where r.rank_date=v_eod and r.execution_eligible
  order by r.execution_score desc,r.emir_score desc,r.ticker
  limit 3;

  select count(*) into v_rows from public.cak_idx_rank_daily where rank_date=v_eod;
  select count(*) into v_eligible from public.cak_idx_rank_daily where rank_date=v_eod and execution_eligible;
  select count(*) into v_top3 from public.cak_idx_top3_execution where rank_date=v_eod;

  return jsonb_build_object(
    'status',case when v_top3=3 then 'READY' else 'INSUFFICIENT_EXECUTION_READY' end,
    'rank_date',v_eod,'ranked',v_rows,'execution_eligible',v_eligible,'top3',v_top3,
    'feature_contract','EMIR_BLOCK_IDX_FEATURE_V1','scanner_data_mode','DATABASE_ONLY'
  );
end
$fn$;

create or replace function public.cak_load_latest_idx_top3_v1()
returns setof public.cak_idx_top3_execution
language sql
security definer
set search_path=pg_catalog,public
stable
as $fn$
  select t.* from public.cak_idx_top3_execution t
  where t.rank_date=(select max(rank_date) from public.cak_idx_top3_execution)
  order by t.execution_rank
$fn$;

create or replace function public.cak_load_idx_market_panel_v1(
  p_tickers text[],
  p_sessions integer default 130
)
returns table(
  ticker text,trade_date date,open numeric,high numeric,low numeric,close numeric,
  volume numeric,traded_value numeric,frequency numeric,foreign_buy numeric,
  foreign_sell numeric,foreign_net numeric,bid numeric,offer numeric
)
language sql
security definer
set search_path=pg_catalog,public
stable
as $fn$
  with sessions as (
    select d.trade_date from (
      select distinct m.trade_date from public.cak_idx_market_daily m
      where m.source_verified order by m.trade_date desc limit least(greatest(p_sessions,20),160)
    ) d
  )
  select m.ticker,m.trade_date,m.open,m.high,m.low,m.close,m.volume,m.traded_value,m.frequency,
    m.foreign_buy,m.foreign_sell,m.foreign_net,m.bid,m.offer
  from public.cak_idx_market_daily m join sessions s using(trade_date)
  where m.ticker=any(p_tickers) and m.source_verified
  order by m.ticker,m.trade_date
$fn$;

alter table public.cak_idx_endpoint_catalog enable row level security;
alter table public.cak_idx_ingestion_runs enable row level security;
alter table public.cak_idx_payload_manifest enable row level security;
alter table public.cak_idx_ingestion_failures enable row level security;
alter table public.cak_idx_market_daily enable row level security;
alter table public.cak_idx_index_daily enable row level security;
alter table public.cak_idx_broker_market_daily enable row level security;
alter table public.cak_idx_events enable row level security;
alter table public.cak_idx_company_snapshot enable row level security;
alter table public.cak_idx_fundamental_snapshot enable row level security;
alter table public.cak_idx_rank_daily enable row level security;
alter table public.cak_idx_top3_execution enable row level security;

revoke all on table public.cak_idx_endpoint_catalog,public.cak_idx_ingestion_runs,
public.cak_idx_payload_manifest,public.cak_idx_ingestion_failures,public.cak_idx_market_daily,
public.cak_idx_index_daily,public.cak_idx_broker_market_daily,public.cak_idx_events,
public.cak_idx_company_snapshot,public.cak_idx_fundamental_snapshot,public.cak_idx_rank_daily,
public.cak_idx_top3_execution from public,anon,authenticated;

grant select,insert,update,delete on table public.cak_idx_endpoint_catalog,public.cak_idx_ingestion_runs,
public.cak_idx_payload_manifest,public.cak_idx_ingestion_failures,public.cak_idx_market_daily,
public.cak_idx_index_daily,public.cak_idx_broker_market_daily,public.cak_idx_events,
public.cak_idx_company_snapshot,public.cak_idx_fundamental_snapshot,public.cak_idx_rank_daily,
public.cak_idx_top3_execution to service_role;

revoke all on function public.cak_refresh_idx_ranking_v1(date),
public.cak_load_latest_idx_top3_v1(),public.cak_load_idx_market_panel_v1(text[],integer)
from public,anon,authenticated;
grant execute on function public.cak_refresh_idx_ranking_v1(date),
public.cak_load_latest_idx_top3_v1(),public.cak_load_idx_market_panel_v1(text[],integer)
to service_role;
