-- Fundamental evidence can change the rank boundary. Reconcile the queue to
-- the post-fundamental Top 900, then ingest any newly admitted issuers before
-- declaring the ranking final.
create or replace function public.cak_idx_reconcile_fundamental_queue_v1(
  p_rank_date date
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog,public
as $$
declare
  v_seed jsonb;
  v_deleted integer;
begin
  delete from public.cak_idx_fundamental_ingestion_queue q
  where q.rank_date=p_rank_date
    and not exists (
      select 1
      from (
        select r.ticker
        from public.cak_idx_rank_daily r
        where r.rank_date=p_rank_date
        order by r.overall_rank
        limit 900
      ) selected
      where selected.ticker=q.ticker
    );
  get diagnostics v_deleted=row_count;

  v_seed:=public.cak_idx_seed_fundamental_queue_v1(p_rank_date);
  return v_seed||jsonb_build_object('removed_outside_top900',v_deleted);
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
  v_seed jsonb;
  v_worker_request bigint;
begin
  select coalesce(p_rank_date,max(rank_date)) into v_rank_date
  from public.cak_idx_fundamental_ingestion_queue;
  if v_rank_date is null then return jsonb_build_object('state','QUEUE_EMPTY'); end if;

  update public.cak_idx_fundamental_ingestion_queue
  set status=case when attempts>=3 then 'PARSE_FAILED' else 'RETRY' end,
      next_retry_at=case when attempts>=3 then null else now() end,
      completed_at=case when attempts>=3 then now() else null end,
      error_code='STALE_WORKER_RECLAIMED',updated_at=now()
  where rank_date=v_rank_date and status='RUNNING'
    and claimed_at<now()-interval '5 minutes';

  select count(*) into v_open
  from public.cak_idx_fundamental_ingestion_queue
  where rank_date=v_rank_date and status in ('PENDING','RUNNING','RETRY');
  if v_open>0 then
    v_worker_request:=public.cak_idx_kick_xbrl_worker_v1();
    return jsonb_build_object(
      'state','EVIDENCE_PENDING','rank_date',v_rank_date,
      'open_jobs',v_open,'worker_request_id',v_worker_request
    );
  end if;

  v_rank:=public.cak_refresh_idx_ranking_v2(v_rank_date);
  v_seed:=public.cak_idx_reconcile_fundamental_queue_v1(v_rank_date);
  select count(*) into v_open
  from public.cak_idx_fundamental_ingestion_queue
  where rank_date=v_rank_date and status in ('PENDING','RUNNING','RETRY');
  if v_open>0 then
    v_worker_request:=public.cak_idx_kick_xbrl_worker_v1();
    return jsonb_build_object(
      'state','TOP900_RECONCILIATION_PENDING','rank_date',v_rank_date,
      'open_jobs',v_open,'worker_request_id',v_worker_request,
      'ranking',v_rank,'seed',v_seed
    );
  end if;

  return jsonb_build_object(
    'state','FINALIZED','rank_date',v_rank_date,'ranking',v_rank,'seed',v_seed,
    'complete',(select count(*) from public.cak_idx_fundamental_ingestion_queue where rank_date=v_rank_date and status='COMPLETE'),
    'parse_failed',(select count(*) from public.cak_idx_fundamental_ingestion_queue where rank_date=v_rank_date and status='PARSE_FAILED'),
    'officially_unavailable',(select count(*) from public.cak_idx_fundamental_ingestion_queue where rank_date=v_rank_date and status='OFFICIAL_FILING_UNAVAILABLE'),
    'special_instruments',(select count(*) from public.cak_idx_fundamental_ingestion_queue where rank_date=v_rank_date and status='SPECIAL_INSTRUMENT')
  );
end
$$;

revoke all on function public.cak_idx_reconcile_fundamental_queue_v1(date) from public,anon,authenticated;
revoke all on function public.cak_idx_finalize_fundamental_evidence_v1(date) from public,anon,authenticated;
grant execute on function public.cak_idx_reconcile_fundamental_queue_v1(date) to service_role;
grant execute on function public.cak_idx_finalize_fundamental_evidence_v1(date) to service_role;
