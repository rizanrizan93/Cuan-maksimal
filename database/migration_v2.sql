-- IDX Emir Public Framework Scanner v1.1.0
-- Additive and idempotent. Run after migration_v1.sql (or run both in order).

create table if not exists public.cak_provider_audit (
    audit_id text primary key,
    scan_id text not null references public.cak_scan_runs(scan_id) on delete cascade,
    ticker text,
    provider text not null,
    status text,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_cak_provider_scan
    on public.cak_provider_audit(scan_id, provider, status);

create index if not exists idx_cak_provider_ticker
    on public.cak_provider_audit(ticker, created_at desc);

grant usage on schema public to service_role;
grant select, insert, update, delete on table
    public.cak_provider_audit
    to service_role;
