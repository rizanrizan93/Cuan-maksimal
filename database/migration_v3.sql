-- IDX Emir Public Framework Scanner v1.2.0
-- Additive and idempotent. Run after migration_v1.sql and migration_v2.sql.

create table if not exists public.cak_direct_evidence (
    evidence_id text primary key,
    scan_id text not null references public.cak_scan_runs(scan_id) on delete cascade,
    ticker text not null,
    evidence_type text not null,
    observed_at timestamptz,
    source_verified boolean not null default false,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_cak_direct_evidence_scan
    on public.cak_direct_evidence(scan_id, evidence_type, ticker);

create index if not exists idx_cak_direct_evidence_ticker_date
    on public.cak_direct_evidence(ticker, observed_at desc);

grant usage on schema public to service_role;
grant select, insert, update, delete on table
    public.cak_direct_evidence
    to service_role;
