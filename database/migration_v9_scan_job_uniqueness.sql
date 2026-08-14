-- v9 scan-job integrity hardening
-- 1) retire stale duplicate active jobs that already have a newer terminal sibling
-- 2) atomically prevent two active jobs for the same universe + scanner version

with stale_active as (
    select a.scan_id
    from public.cak_scan_jobs a
    where a.status in ('CREATED', 'RUNNING', 'PAUSED', 'FINALIZE_RETRY_REQUIRED')
      and exists (
          select 1
          from public.cak_scan_jobs t
          where t.universe_hash = a.universe_hash
            and t.scanner_version = a.scanner_version
            and t.status in ('COMPLETED', 'COMPLETED_PARTIAL_PERSISTENCE', 'CANCELLED', 'FAILED')
            and t.created_at >= a.created_at
            and t.updated_at > a.updated_at
      )
)
update public.cak_scan_jobs j
set status = 'CANCELLED',
    result_status = 'AUTO_CANCELLED_DUPLICATE_ACTIVE_JOB',
    last_error = 'Retired by scan-job uniqueness migration after a newer terminal sibling completed.',
    heartbeat_at = now(),
    updated_at = now()
where j.scan_id in (select scan_id from stale_active);

-- If historical duplicate active rows still remain, keep only the most recently updated.
with ranked_active as (
    select
        scan_id,
        row_number() over (
            partition by universe_hash, scanner_version
            order by updated_at desc nulls last, created_at desc, scan_id desc
        ) as rn
    from public.cak_scan_jobs
    where status in ('CREATED', 'RUNNING', 'PAUSED', 'FINALIZE_RETRY_REQUIRED')
)
update public.cak_scan_jobs j
set status = 'CANCELLED',
    result_status = 'AUTO_CANCELLED_DUPLICATE_ACTIVE_JOB',
    last_error = 'Retired by scan-job uniqueness migration as a duplicate active job.',
    heartbeat_at = now(),
    updated_at = now()
where j.scan_id in (select scan_id from ranked_active where rn > 1);

create unique index if not exists uq_cak_scan_jobs_active_universe_version
    on public.cak_scan_jobs (universe_hash, scanner_version)
    where status in ('CREATED', 'RUNNING', 'PAUSED', 'FINALIZE_RETRY_REQUIRED');
