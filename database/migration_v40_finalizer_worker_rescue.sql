-- If finalization finds unfinished or reclaimed jobs, restart the bounded worker chain.
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
  v_worker_request bigint;
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
    v_worker_request:=public.cak_idx_kick_xbrl_worker_v1();
    return jsonb_build_object(
      'state','EVIDENCE_PENDING','rank_date',v_rank_date,
      'open_jobs',v_open,'worker_request_id',v_worker_request
    );
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

revoke all on function public.cak_idx_finalize_fundamental_evidence_v1(date) from public,anon,authenticated;
grant execute on function public.cak_idx_finalize_fundamental_evidence_v1(date) to service_role;

