-- IDX Emir Autonomous Scanner v1.4.0
-- Additive and idempotent. Run after migrations v1-v4.

create table if not exists public.cak_autonomous_evidence (
    evidence_id text primary key,
    scan_id text references public.cak_scan_runs(scan_id) on delete cascade,
    ticker text not null,
    evidence_type text not null,
    observed_at timestamptz,
    source_verified boolean not null default false,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_cak_autonomous_evidence_scan
    on public.cak_autonomous_evidence(scan_id, evidence_type);

create index if not exists idx_cak_autonomous_evidence_ticker_date
    on public.cak_autonomous_evidence(ticker, observed_at desc);

grant usage on schema public to service_role;
grant select, insert, update, delete on table
    public.cak_autonomous_evidence
    to service_role;

alter default privileges for role postgres in schema public
    grant select, insert, update, delete on tables to service_role;
