create or replace function public.cak_refresh_idx_ranking_v2(
  p_as_of date default ((now() at time zone 'Asia/Jakarta')::date)
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
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
    from (select distinct trade_date from public.cak_idx_market_daily
          where trade_date<=v_eod and source_verified order by trade_date desc limit 130) s
  ), bars as (
    select m.*,s.rn
    from public.cak_idx_market_daily m join sessions s using(trade_date)
    where m.source_verified and m.close>0
  ), agg as (
    select ticker,count(*)::int observations,
      (array_agg(close order by trade_date desc))[1] close,
      (array_agg(close order by trade_date desc))[6] close_5,
      (array_agg(close order by trade_date desc))[21] close_20,
      (array_agg(close order by trade_date desc))[61] close_60,
      avg(close) filter(where rn<=20) ma20,
      avg(close) filter(where rn<=50) ma50,
      avg(close) filter(where rn<=100) ma100,
      max(high) filter(where rn between 2 and 21) resistance20,
      max(high) filter(where rn between 2 and 61) resistance60,
      min(low) filter(where rn between 2 and 11) support10,
      avg(greatest(high-low,abs(high-previous),abs(low-previous))) filter(where rn<=14) atr14,
      sum(foreign_net) filter(where rn<=20) foreign_net20,
      count(*) filter(where rn<=20 and foreign_net>0)::int foreign_positive20,
      avg(traded_value) filter(where rn<=20) adtv20,
      avg(frequency) filter(where rn<=20) freq20,
      avg(case when listed_shares>0 then volume/listed_shares end) filter(where rn<=20) turnover20,
      avg(volume) filter(where rn<=20 and close>=previous) /
        nullif(avg(volume) filter(where rn<=20 and close<previous),0) up_down_volume_ratio,
      (array_agg(bid order by trade_date desc))[1] bid,
      (array_agg(offer order by trade_date desc))[1] offer
    from bars group by ticker
  ), measures as (
    select a.*,
      100*(close/nullif(close_5,0)-1) ret5,
      100*(close/nullif(close_20,0)-1) ret20,
      100*(close/nullif(close_60,0)-1) ret60,
      case when close>0 and offer>0 and bid>0 and offer>=bid then 100*(offer-bid)/close end spread,
      close>resistance20 bos20,
      close>ma20+2*atr14 overextended,
      case when atr14>0 then greatest(support10*0.99,close-1.2*atr14) end sl,
      case when atr14>0 then greatest(resistance60*1.01,close+2.2*atr14) end tp1,
      case when atr14>0 then greatest(resistance60*1.01,close+3.5*atr14) end tp2
    from agg a
  ), cross_ranked as (
    select m.*,
      100*percent_rank() over(order by coalesce(foreign_net20,-1)) foreign_rank,
      100*percent_rank() over(order by foreign_positive20) persistence_rank,
      100*percent_rank() over(order by coalesce(up_down_volume_ratio,0)) absorption_rank,
      100*percent_rank() over(order by ln(1+greatest(coalesce(adtv20,0),0))) value_rank,
      100*percent_rank() over(order by coalesce(freq20,0)) frequency_rank,
      100*percent_rank() over(order by coalesce(ret20,-999)) ret20_rank,
      100*percent_rank() over(order by coalesce(ret60,-999)) ret60_rank
    from measures m where observations>=20
  ), latest_f as (
    select distinct on(ticker) * from public.cak_idx_fundamental_snapshot
    where source_verified and observed_on<=v_eod
    order by ticker,period_end desc,observed_on desc
  ), event_state as (
    select c.ticker,
      exists(select 1 from public.cak_idx_events e
        where e.ticker=c.ticker and e.event_type='SUSPEND' and e.event_date<=v_eod
          and not exists(select 1 from public.cak_idx_events u
            where u.ticker=e.ticker and u.event_type='UNSUSPEND'
              and u.event_date>=e.event_date and u.event_date<=v_eod)) active_suspension,
      exists(select 1 from public.cak_idx_events e where e.ticker=c.ticker
        and e.event_type='UMA' and e.event_date between v_eod-14 and v_eod) recent_uma,
      exists(select 1 from public.cak_idx_events e where e.ticker=c.ticker
        and e.event_family='ISSUED_HISTORY'
        and e.event_type in ('HMETD','TANPA_HMETD','PRIVATE_PLACEMENT','RIGHTS_ISSUE','WARAN')
        and e.event_date between v_eod-45 and v_eod) recent_dilution
    from cross_ranked c
  ), ihsg as (
    select (array_agg(close order by trade_date desc))[1] current_close,
      (array_agg(close order by trade_date desc))[21] prior20
    from public.cak_idx_index_daily
    where index_code='COMPOSITE' and trade_date<=v_eod and source_verified
  ), components as (
    select c.*,f.coverage_pct,
      least(100,greatest(0,50+0.7*coalesce(f.revenue_growth_yoy_pct,0)+
        0.3*coalesce(f.earnings_growth_yoy_pct,0))) growth_score,
      least(100,greatest(0,0.45*least(100,greatest(0,coalesce(f.roe_pct,0)*4))+
        0.25*least(100,greatest(0,coalesce(f.roa_pct,0)*7))+
        0.30*least(100,greatest(0,coalesce(f.net_margin_pct,0)*4)))) profitability_score,
      case when f.debt_to_equity is null then 40 when f.debt_to_equity<=0.5 then 100
        when f.debt_to_equity<=1 then 75 when f.debt_to_equity<=2 then 40 else 10 end * 0.65 +
      case when f.current_ratio>=1.5 then 100 when f.current_ratio>=1 then 65
        when f.current_ratio is null then 40 else 20 end * 0.20 +
      case when f.cash_to_debt_ratio>=1 then 100 when f.cash_to_debt_ratio>=0.5 then 70
        when f.cash_to_debt_ratio is null then 40 else 25 end * 0.15 balance_score,
      case when f.ocf>0 and f.net_income>0 and f.ocf>=f.net_income then 100
        when f.ocf>0 then 70 else 10 end * 0.60 +
      case when f.fcf>0 then 100 when f.fcf is null then 40 else 10 end * 0.40 cashflow_score,
      least(100,greatest(0,0.40*foreign_rank+0.25*persistence_rank+
        0.20*absorption_rank+0.15*least(100,greatest(0,50+5000*coalesce(turnover20,0))))) smart_score,
      least(100,greatest(0,25*(close>ma20)::int+25*(ma20>ma50)::int+
        15*(ma50>coalesce(ma100,ma50-1))::int+20*bos20::int+
        15*least(1,greatest(0,1-abs(close-ma20)/nullif(2*atr14,0))))) structure_score,
      least(100,greatest(0,0.50*ret20_rank+0.30*ret60_rank+
        20*least(1,greatest(0,1-abs(coalesce(ret5,0)-3)/12)))) momentum_score,
      least(100,greatest(0,0.65*value_rank+0.25*frequency_rank+
        0.10*least(1,observations/100.0)*100)) liquidity_score,
      case when i.current_close>i.prior20 then 75 when i.current_close is null or i.prior20 is null then 45 else 25 end regime_score,
      least(100,greatest(0,100-40*e.active_suspension::int-15*e.recent_uma::int-
        20*e.recent_dilution::int-10*greatest(coalesce(c.spread,0)-1,0))) risk_score,
      e.active_suspension,e.recent_uma,e.recent_dilution
    from cross_ranked c left join latest_f f using(ticker)
    left join event_state e using(ticker) cross join ihsg i
  ), scored0 as (
    select x.*,
      0.30*growth_score+0.25*profitability_score+0.20*balance_score+0.25*cashflow_score fundamental_score,
      case when sl>0 and tp1>close and close>sl then (tp1-close)/(close-sl) end rr1,
      100*(
        (coverage_pct is not null)::int+(observations>=80)::int+(adtv20 is not null)::int+
        (foreign_net20 is not null)::int+(spread is not null)::int
      )/5.0 data_completeness
    from components x
  ), scored as (
    select s.*,
      0.30*fundamental_score+0.25*smart_score+0.20*structure_score+
        0.10*momentum_score+0.10*liquidity_score+0.05*risk_score emir_score,
      0.25*fundamental_score+0.30*smart_score+0.25*structure_score+
        0.10*liquidity_score+0.10*risk_score execution_score,
      observations>=80 and adtv20>=1000000000 and close>ma20 and ma20>ma50
        and foreign_positive20>=10 and foreign_net20>0 and not overextended
        and not active_suspension and not recent_dilution and coalesce(spread,99)<=2
        and coalesce(coverage_pct,0)>=50 and fundamental_score>=50
        and sl>0 and tp1>close as base_eligible
    from scored0 s
  ), ranked as (
    select s.*,row_number() over(order by emir_score desc,ticker)::int overall_rank,
      case when base_eligible and rr1>=1.8 and execution_score>=65
        then row_number() over(partition by (base_eligible and rr1>=1.8 and execution_score>=65)
          order by execution_score desc,emir_score desc,ticker)::int end execution_rank
    from scored s
  )
  insert into public.cak_idx_rank_daily(
    rank_date,ticker,overall_rank,execution_rank,emir_score,execution_score,
    fundamental_score,smart_money_score,momentum_score,liquidity_score,regime_score,risk_score,
    observations,close,return_5d_pct,return_20d_pct,return_60d_pct,foreign_net_20d,
    foreign_positive_days_20d,adtv_20d,frequency_20d,spread_pct,entry_price,stop_loss,
    structural_tp1,rr_tp1,official_fundamental_coverage_pct,active_suspension,recent_uma,
    recent_dilution,execution_eligible,blocker,feature_contract,growth_score,balance_score,
    cashflow_score,structure_score,bos_20d,overextended,tp2,setup_state,data_completeness_pct)
  select v_eod,ticker,overall_rank,
    case when base_eligible and rr1>=1.8 and execution_score>=65 then execution_rank end,
    round(emir_score::numeric,4),round(execution_score::numeric,4),
    round(fundamental_score::numeric,4),round(smart_score::numeric,4),
    round(momentum_score::numeric,4),round(liquidity_score::numeric,4),
    round(regime_score::numeric,4),round(risk_score::numeric,4),observations,close,
    round(ret5::numeric,4),round(ret20::numeric,4),round(ret60::numeric,4),foreign_net20,
    foreign_positive20,adtv20,freq20,spread,close,sl,tp1,rr1,coverage_pct,
    active_suspension,recent_uma,recent_dilution,
    base_eligible and rr1>=1.8 and execution_score>=65,
    case when observations<80 then 'INSUFFICIENT_6M_HISTORY'
      when adtv20<1000000000 then 'LIQUIDITY_BELOW_1B'
      when active_suspension then 'ACTIVE_SUSPENSION'
      when recent_dilution then 'RECENT_DILUTION'
      when coalesce(coverage_pct,0)<50 then 'OFFICIAL_FUNDAMENTAL_COVERAGE'
      when fundamental_score<50 then 'FUNDAMENTAL_SCORE_BELOW_50'
      when not(close>ma20 and ma20>ma50) then 'NO_BULLISH_HTF_STACK'
      when not(foreign_positive20>=10 and foreign_net20>0) then 'NO_SILENT_ACCUMULATION'
      when overextended then 'OVEREXTENDED_ABOVE_VALUE'
      when coalesce(spread,99)>2 then 'SPREAD_TOO_WIDE'
      when sl is null or tp1 is null or not(sl<close and tp1>close) then 'INVALID_STRUCTURAL_GEOMETRY'
      when coalesce(rr1,0)<1.8 then 'RR_BELOW_1_8'
      when execution_score<65 then 'EXECUTION_SCORE_BELOW_65' else null end,
    'EMIR_BLOCK_IDX_FEATURE_V2',round(growth_score::numeric,4),round(balance_score::numeric,4),
    round(cashflow_score::numeric,4),round(structure_score::numeric,4),coalesce(bos20,false),
    coalesce(overextended,false),tp2,
    case when coalesce(bos20,false) then 'BOS_ATR_EXPANSION' else 'TREND_PULLBACK_TO_VALUE' end,
    data_completeness
  from ranked;

  insert into public.cak_idx_top3_execution(
    rank_date,execution_rank,ticker,execution_score,emir_score,entry_price,stop_loss,tp1,tp2,
    rr_tp1,risk_per_share,geometry_state,decision_state,source_state,feature_contract)
  select v_eod,row_number() over(order by r.execution_score desc,r.emir_score desc,r.ticker)::int,
    r.ticker,r.execution_score,r.emir_score,r.entry_price,r.stop_loss,r.structural_tp1,r.tp2,
    r.rr_tp1,r.entry_price-r.stop_loss,
    case when r.bos_20d then 'BOS_STRUCTURAL_ATR_EXPANSION' else 'TREND_PULLBACK_STRUCTURAL_ATR' end,
    'EXECUTION_READY','VERIFIED_OFFICIAL_BLOCK_IDX_DATABASE_ONLY','EMIR_BLOCK_IDX_FEATURE_V2'
  from public.cak_idx_rank_daily r
  where r.rank_date=v_eod and r.execution_eligible
  order by r.execution_score desc,r.emir_score desc,r.ticker limit 3;

  select count(*) into v_rows from public.cak_idx_rank_daily where rank_date=v_eod;
  select count(*) into v_eligible from public.cak_idx_rank_daily where rank_date=v_eod and execution_eligible;
  select count(*) into v_top3 from public.cak_idx_top3_execution where rank_date=v_eod;
  return jsonb_build_object('status',case when v_top3=3 then 'READY' else 'INSUFFICIENT_EXECUTION_READY' end,
    'rank_date',v_eod,'ranked',v_rows,'execution_eligible',v_eligible,'top3',v_top3,
    'feature_contract','EMIR_BLOCK_IDX_FEATURE_V2','scanner_data_mode','DATABASE_ONLY');
end;
$$;


