-- Convert continuous model geometry into executable IDX price fractions.
create or replace function public.cak_idx_tick_size_v1(p_price numeric)
returns numeric
language sql
immutable
strict
set search_path = pg_catalog
as $$
  select case
    when p_price < 200 then 1
    when p_price < 500 then 2
    when p_price < 2000 then 5
    when p_price < 5000 then 10
    else 25
  end::numeric
$$;

create or replace function public.cak_idx_floor_to_tick_v1(p_price numeric)
returns numeric
language sql
immutable
strict
set search_path = pg_catalog, public
as $$
  select floor(round(p_price,6)/public.cak_idx_tick_size_v1(round(p_price,6)))
    *public.cak_idx_tick_size_v1(round(p_price,6))
$$;

create or replace function public.cak_load_latest_idx_top3_v3()
returns jsonb
language sql
stable
security definer
set search_path = pg_catalog, public
as $$
  with latest as (
    select max(rank_date) rank_date from public.cak_idx_top3_execution
  )
  select coalesce(jsonb_agg(
    to_jsonb(t) || jsonb_build_object(
      'raw_model_stop_loss',t.stop_loss,
      'raw_model_tp1',t.tp1,
      'raw_model_tp2',t.tp2,
      'stop_loss',public.cak_idx_floor_to_tick_v1(t.stop_loss),
      'tp1',public.cak_idx_floor_to_tick_v1(t.tp1),
      'tp2',public.cak_idx_floor_to_tick_v1(t.tp2),
      'risk_per_share',t.entry_price-public.cak_idx_floor_to_tick_v1(t.stop_loss),
      'rr_tp1',round((public.cak_idx_floor_to_tick_v1(t.tp1)-t.entry_price)/
        nullif(t.entry_price-public.cak_idx_floor_to_tick_v1(t.stop_loss),0),4),
      'price_fraction_state','IDX_TICK_EXECUTABLE',
      'company_name',c.company_name,
      'sector',c.sector,
      'subsector',c.subsector,
      'ownership',coalesce(o.ownership,'{}'::jsonb),
      'recent_events',coalesce(e.events,'[]'::jsonb)
    ) order by t.execution_rank
  ),'[]'::jsonb)
  from public.cak_idx_top3_execution t
  join latest l on l.rank_date=t.rank_date
  left join lateral (
    select x.company_name,x.sector,x.subsector
    from public.cak_idx_company_snapshot x
    where x.ticker=t.ticker and x.source_verified
    order by x.observed_on desc,x.ingested_at desc limit 1
  ) c on true
  left join lateral (
    select jsonb_strip_nulls(jsonb_build_object(
      'observed_on',x.observed_on,'controller_pct',x.controller_pct,
      'public_pct',x.public_pct,'treasury_pct',x.treasury_pct,
      'holder_count',x.holder_count,'source_url',x.source_url,
      'source_verified',x.source_verified
    )) ownership
    from public.cak_idx_ownership_snapshot x
    where x.ticker=t.ticker and x.source_verified
    order by x.observed_on desc,x.ingested_at desc limit 1
  ) o on true
  left join lateral (
    select jsonb_agg(jsonb_strip_nulls(jsonb_build_object(
      'event_family',z.event_family,'event_type',z.event_type,
      'event_date',z.event_date,'publication_date',z.publication_date,
      'title',z.title,'source_url',z.source_url
    )) order by coalesce(z.publication_date,z.event_date) desc) events
    from (
      select x.event_family,x.event_type,x.event_date,x.publication_date,x.title,x.source_url
      from public.cak_idx_events x
      where x.ticker=t.ticker and x.source_verified
        and coalesce(x.publication_date,x.event_date)>=t.rank_date-30
      order by coalesce(x.publication_date,x.event_date) desc limit 10
    ) z
  ) e on true;
$$;

revoke all on function public.cak_idx_tick_size_v1(numeric) from public,anon,authenticated;
revoke all on function public.cak_idx_floor_to_tick_v1(numeric) from public,anon,authenticated;
revoke all on function public.cak_load_latest_idx_top3_v3() from public,anon,authenticated;
grant execute on function public.cak_idx_tick_size_v1(numeric) to service_role;
grant execute on function public.cak_idx_floor_to_tick_v1(numeric) to service_role;
grant execute on function public.cak_load_latest_idx_top3_v3() to service_role;
