-- Emir scanner production hardening 2026-08-13
-- 1) seed top-60 daily radar observations into 5D/20D/60D outcome memory
-- 2) resolve pending outcomes strictly from later OHLCV sessions
-- 3) content-deduplicate durable research memory
-- 4) keep one heavy per-scan snapshot/job on Supabase Free

begin;

create index if not exists idx_cak_outcome_pending_signal
on public.cak_outcome_memory(outcome_verified, signal_date, ticker);

create index if not exists idx_cak_research_memory_content_dedup
on public.cak_research_memory(ticker, family, content_sha256, updated_at desc);

create or replace function public.cak_seed_outcomes_for_scan(p_scan_id text, p_limit integer default 60)
returns integer language plpgsql security invoker set search_path = pg_catalog, public as $$
declare v_rows integer := 0;
begin
  with ranked as (
    select r.scan_id, r.ticker, r.conviction_score, r.public_method_state, r.action,
           r.coverage_pct, c.last_session_date as signal_date,
           nullif(c.payload -> -1 ->> 4, '')::numeric as anchor_price,
           row_number() over (order by r.conviction_score desc nulls last, r.ticker) as rn
      from public.cak_radar_snapshots r
      join public.cak_ohlcv_cache c on c.ticker=r.ticker
     where r.scan_id=p_scan_id and c.last_session_date is not null
       and jsonb_typeof(c.payload)='array' and jsonb_array_length(c.payload)>0
  ), seeds as (
    select ranked.*, h.horizon_days from ranked
    cross join (values (5),(20),(60)) as h(horizon_days)
    where rn <= greatest(1,p_limit) and anchor_price > 0
  )
  insert into public.cak_outcome_memory(outcome_id,scan_id,ticker,signal_date,horizon_days,outcome_verified,payload,created_at,updated_at)
  select md5('EMIR|'||ticker||'|'||signal_date::text||'|'||horizon_days::text),
         scan_id,ticker,signal_date,horizon_days,false,
         jsonb_build_object('ticker',ticker,'signal_date',signal_date,'horizon_days',horizon_days,
           'anchor_price',anchor_price,'emir_conviction_score',conviction_score,
           'emir_decision_state',public_method_state,'action',action,
           'emir_evidence_coverage_pct',coverage_pct,'source','AUTO_RADAR_OUTCOME_SEED_V1'),
         now(),now()
  from seeds
  on conflict(outcome_id) do update set scan_id=excluded.scan_id,payload=excluded.payload,updated_at=now()
  where public.cak_outcome_memory.outcome_verified=false;
  get diagnostics v_rows = row_count;
  return v_rows;
end; $$;

create or replace function public.cak_resolve_outcome_memory(p_limit integer default 5000)
returns integer language plpgsql security invoker set search_path = pg_catalog, public as $$
declare rec record; v_anchor numeric; v_target_close numeric; v_target_date date;
        v_min_close numeric; v_max_close numeric; v_updated integer := 0;
begin
  for rec in
    select o.outcome_id,o.ticker,o.signal_date,o.horizon_days,o.payload,c.payload as bars_payload
    from public.cak_outcome_memory o join public.cak_ohlcv_cache c on c.ticker=o.ticker
    where o.outcome_verified=false and o.signal_date is not null and o.horizon_days in(5,20,60)
      and jsonb_typeof(c.payload)='array'
    order by o.signal_date,o.ticker,o.horizon_days limit greatest(1,p_limit)
  loop
    v_anchor := nullif(rec.payload->>'anchor_price','')::numeric;
    if v_anchor is null or v_anchor <= 0 then continue; end if;
    with bars as (
      select (b.elem->>0)::date bar_date, nullif(b.elem->>4,'')::numeric close_price,
             row_number() over(order by (b.elem->>0)::date) rn
      from jsonb_array_elements(rec.bars_payload) b(elem)
      where (b.elem->>0)::date > rec.signal_date and nullif(b.elem->>4,'') is not null
    ), windowed as (select * from bars where rn <= rec.horizon_days)
    select max(close_price) filter(where rn=rec.horizon_days),
           max(bar_date) filter(where rn=rec.horizon_days), min(close_price), max(close_price)
      into v_target_close,v_target_date,v_min_close,v_max_close from windowed;
    if v_target_close is null or v_target_date is null then continue; end if;
    update public.cak_outcome_memory set outcome_verified=true,
      payload=rec.payload||jsonb_build_object('target_date',v_target_date,'target_close',v_target_close,
        'return_pct',round(100.0*(v_target_close/v_anchor-1.0),4),
        'max_drawdown_pct',round(100.0*(v_min_close/v_anchor-1.0),4),
        'max_favorable_excursion_pct',round(100.0*(v_max_close/v_anchor-1.0),4),
        'resolved_at',now(),'resolver','OHLCV_TRADING_SESSION_RESOLVER_V1'),updated_at=now()
    where outcome_id=rec.outcome_id;
    v_updated := v_updated+1;
  end loop;
  return v_updated;
end; $$;

create or replace function public.cak_free_tier_housekeeping(p_keep_scan_runs integer default 1,p_keep_terminal_jobs integer default 1)
returns jsonb language plpgsql security definer set search_path=pg_catalog,public as $$
declare v_ephemeral integer:=0; v_dup integer:=0; v_memory integer:=0; v_runs integer:=0; v_jobs integer:=0; v_outcomes integer:=0;
begin
  delete from public.cak_research_memory where family in('BROKER_INVENTORY_OHLCV_PROXY','BID_OFFER_EOD_PROXY');
  get diagnostics v_ephemeral=row_count;
  with ranked as (
    select memory_id,row_number() over(partition by ticker,family,content_sha256 order by updated_at desc nulls last,observed_at desc nulls last,memory_id desc) rn
    from public.cak_research_memory where nullif(content_sha256,'') is not null
  ), doomed as(select memory_id from ranked where rn>1)
  delete from public.cak_research_memory m using doomed d where m.memory_id=d.memory_id;
  get diagnostics v_dup=row_count;
  with ranked as (
    select memory_id,row_number() over(partition by ticker,family order by effective_period desc nulls last,observed_at desc nulls last,updated_at desc nulls last,memory_id desc) rn,
      case when family in('NARRATIVE_EVENT','KSEI_CORPORATE_ACTION') then 6 else 4 end keep_n
    from public.cak_research_memory
  ), doomed as(select memory_id from ranked where rn>keep_n)
  delete from public.cak_research_memory m using doomed d where m.memory_id=d.memory_id;
  get diagnostics v_memory=row_count;
  delete from public.cak_outcome_memory where signal_date < current_date-400; get diagnostics v_outcomes=row_count;
  with ranked as(select scan_id,row_number() over(order by created_at desc,scan_id desc) rn from public.cak_scan_runs),doomed as(select scan_id from ranked where rn>greatest(1,p_keep_scan_runs))
  delete from public.cak_scan_runs r using doomed d where r.scan_id=d.scan_id; get diagnostics v_runs=row_count;
  with terminal as(select scan_id,row_number() over(order by updated_at desc,created_at desc,scan_id desc) rn from public.cak_scan_jobs where status in('COMPLETED','COMPLETED_PARTIAL_PERSISTENCE','CANCELLED','FAILED')),doomed as(select scan_id from terminal where rn>greatest(1,p_keep_terminal_jobs))
  delete from public.cak_scan_jobs j using doomed d where j.scan_id=d.scan_id; get diagnostics v_jobs=row_count;
  return jsonb_build_object('state','HOUSEKEEPING_COMPLETE','ephemeral_deleted',v_ephemeral,'content_duplicates_deleted',v_dup,'research_memory_deleted',v_memory,'outcomes_deleted',v_outcomes,'scan_runs_deleted',v_runs,'jobs_deleted',v_jobs);
end; $$;

create or replace function public.cak_free_tier_housekeeping_trigger()
returns trigger language plpgsql security definer set search_path=pg_catalog,public as $$
begin
  if new.status in('COMPLETED','COMPLETED_PARTIAL_PERSISTENCE','CANCELLED','FAILED') and new.status is distinct from old.status then
    perform public.cak_resolve_outcome_memory(5000);
    if new.status in('COMPLETED','COMPLETED_PARTIAL_PERSISTENCE') then perform public.cak_seed_outcomes_for_scan(new.scan_id,60); end if;
    perform public.cak_free_tier_housekeeping(1,1);
  end if;
  return new;
exception when others then return new;
end; $$;

revoke all on function public.cak_seed_outcomes_for_scan(text,integer) from public,anon,authenticated;
revoke all on function public.cak_resolve_outcome_memory(integer) from public,anon,authenticated;
revoke all on function public.cak_free_tier_housekeeping(integer,integer) from public,anon,authenticated;
revoke all on function public.cak_free_tier_housekeeping_trigger() from public,anon,authenticated;
grant execute on function public.cak_seed_outcomes_for_scan(text,integer) to service_role;
grant execute on function public.cak_resolve_outcome_memory(integer) to service_role;
grant execute on function public.cak_free_tier_housekeeping(integer,integer) to service_role;
grant execute on function public.cak_free_tier_housekeeping_trigger() to service_role;

commit;
