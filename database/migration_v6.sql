-- IDX Emir Autonomous Scanner v1.5.0
-- Persistent cache + incremental refresh. Additive and idempotent.
-- Run after migrations v1-v5.

create table if not exists public.cak_ohlcv_cache (
    ticker text primary key,
    period text not null default '5y',
    first_session_date date,
    last_session_date date,
    bars integer not null default 0,
    provider text,
    quality_state text,
    checked_at timestamptz not null,
    last_scan_id text,
    content_sha256 text not null,
    payload jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_cak_ohlcv_cache_last_session
    on public.cak_ohlcv_cache(last_session_date desc);
create index if not exists idx_cak_ohlcv_cache_checked
    on public.cak_ohlcv_cache(checked_at desc);

create table if not exists public.cak_source_cache (
    cache_key text primary key,
    ticker text not null,
    family text not null,
    provider text,
    status text,
    checked_at timestamptz not null,
    valid_until timestamptz not null,
    latest_observed_at timestamptz,
    last_scan_id text,
    content_sha256 text not null,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_cak_source_cache_ticker_family
    on public.cak_source_cache(ticker, family);
create index if not exists idx_cak_source_cache_valid_until
    on public.cak_source_cache(family, valid_until desc);

create or replace function public.cak_touch_updated_at()
returns trigger language plpgsql as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_cak_ohlcv_cache_touch on public.cak_ohlcv_cache;
create trigger trg_cak_ohlcv_cache_touch
before update on public.cak_ohlcv_cache
for each row execute function public.cak_touch_updated_at();

drop trigger if exists trg_cak_source_cache_touch on public.cak_source_cache;
create trigger trg_cak_source_cache_touch
before update on public.cak_source_cache
for each row execute function public.cak_touch_updated_at();

grant usage on schema public to service_role;
grant select, insert, update, delete on table
    public.cak_ohlcv_cache,
    public.cak_source_cache
    to service_role;

alter default privileges for role postgres in schema public
    grant select, insert, update, delete on tables to service_role;
