-- IDX Emir Public Framework Scanner v1.3.0
-- Additive and idempotent. Run after migration_v1.sql, migration_v2.sql, and migration_v3.sql.

create table if not exists public.cak_outcome_memory (
    outcome_id text primary key,
    scan_id text references public.cak_scan_runs(scan_id) on delete set null,
    ticker text not null,
    signal_date date,
    horizon_days integer,
    outcome_verified boolean not null default false,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_cak_outcome_ticker_date
    on public.cak_outcome_memory(ticker, signal_date desc);

create index if not exists idx_cak_outcome_scan
    on public.cak_outcome_memory(scan_id, outcome_verified);

grant usage on schema public to service_role;
grant select, insert, update, delete on table
    public.cak_outcome_memory
    to service_role;

-- Keep future scanner tables reachable by the backend Data API when they are
-- created by the postgres role. Explicit per-table grants remain preferred.
alter default privileges for role postgres in schema public
    grant select, insert, update, delete on tables to service_role;
