-- IDX Emir Autonomous Scanner v1.8.0
-- Durable research memory. Additive and idempotent. Run after migrations v1-v7.

create table if not exists public.cak_research_memory (
    memory_id text primary key,
    ticker text not null,
    family text not null,
    effective_period date,
    observed_at timestamptz,
    provider text,
    source_url text,
    source_verified boolean not null default false,
    official_source boolean not null default false,
    content_sha256 text not null,
    last_scan_id text,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);
create index if not exists idx_cak_research_memory_ticker_family_period on public.cak_research_memory(ticker, family, effective_period desc);
create index if not exists idx_cak_research_memory_family_observed on public.cak_research_memory(family, observed_at desc);
create index if not exists idx_cak_research_memory_scan on public.cak_research_memory(last_scan_id);
drop trigger if exists trg_cak_research_memory_touch on public.cak_research_memory;
create trigger trg_cak_research_memory_touch before update on public.cak_research_memory for each row execute function public.cak_touch_updated_at();
grant usage on schema public to service_role;
grant select, insert, update, delete on table public.cak_research_memory to service_role;
