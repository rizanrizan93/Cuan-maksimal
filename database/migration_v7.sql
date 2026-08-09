-- IDX Emir Autonomous Scanner v1.6.1
-- Resumable chunked scan jobs. Additive and idempotent. Run after migrations v1-v6.

create table if not exists public.cak_scan_jobs (
    scan_id text primary key,
    universe_hash text not null,
    scanner_version text not null,
    status text not null default 'CREATED',
    current_stage text not null default 'BENCHMARK',
    current_offset integer not null default 0,
    current_chunk integer not null default 0,
    chunk_size integer not null default 20,
    total_tickers integer not null default 0,
    processed_tickers integer not null default 0,
    failed_tickers integer not null default 0,
    progress_pct numeric not null default 0,
    scan_mode text,
    result_status text,
    universe jsonb not null default '[]'::jsonb,
    settings jsonb not null default '{}'::jsonb,
    shortlist jsonb not null default '[]'::jsonb,
    failures jsonb not null default '{}'::jsonb,
    result_summary jsonb not null default '{}'::jsonb,
    last_error text,
    heartbeat_at timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_cak_scan_jobs_universe_status
    on public.cak_scan_jobs(universe_hash, status, updated_at desc);
create index if not exists idx_cak_scan_jobs_heartbeat
    on public.cak_scan_jobs(heartbeat_at desc);

create table if not exists public.cak_scan_job_chunks (
    chunk_id text primary key,
    scan_id text not null references public.cak_scan_jobs(scan_id) on delete cascade,
    stage text not null,
    chunk_no integer not null,
    ticker_count integer not null default 0,
    processed_count integer not null default 0,
    failed_count integer not null default 0,
    status text not null,
    started_at timestamptz,
    completed_at timestamptz,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique(scan_id, stage, chunk_no)
);

create index if not exists idx_cak_scan_job_chunks_scan_stage
    on public.cak_scan_job_chunks(scan_id, stage, chunk_no);
create index if not exists idx_cak_scan_job_chunks_completed
    on public.cak_scan_job_chunks(completed_at desc);

create or replace function public.cak_touch_updated_at()
returns trigger language plpgsql as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_cak_scan_jobs_touch on public.cak_scan_jobs;
create trigger trg_cak_scan_jobs_touch
before update on public.cak_scan_jobs
for each row execute function public.cak_touch_updated_at();

grant usage on schema public to service_role;
grant select, insert, update, delete on table
    public.cak_scan_jobs,
    public.cak_scan_job_chunks
    to service_role;

alter default privileges for role postgres in schema public
    grant select, insert, update, delete on tables to service_role;
