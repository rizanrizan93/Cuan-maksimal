create table if not exists public.cak_scan_runs (
    scan_id text primary key,
    as_of timestamptz not null,
    scanner_version text not null,
    scan_mode text,
    ticker_count integer not null default 0,
    production_ready_count integer not null default 0,
    status text not null,
    created_at timestamptz not null default now()
);

create table if not exists public.cak_radar_snapshots (
    scan_id text not null references public.cak_scan_runs(scan_id) on delete cascade,
    ticker text not null,
    as_of timestamptz not null,
    public_method_state text,
    action text,
    conviction_score double precision,
    coverage_pct double precision,
    production_ready boolean not null default false,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    primary key (scan_id, ticker)
);

create table if not exists public.cak_narrative_events (
    event_id text primary key,
    scan_id text not null references public.cak_scan_runs(scan_id) on delete cascade,
    ticker text not null,
    published_at timestamptz,
    title text,
    publisher text,
    source_url text,
    payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_cak_radar_ticker_asof on public.cak_radar_snapshots(ticker, as_of desc);
create index if not exists idx_cak_events_ticker_date on public.cak_narrative_events(ticker, published_at desc);

-- Supabase Data API permissions.
-- Required for both sb_secret_* keys and legacy service_role JWTs because
-- elevated API keys bypass RLS but still require PostgreSQL table privileges.
grant usage on schema public to service_role;
grant select, insert, update, delete on table
    public.cak_scan_runs,
    public.cak_radar_snapshots,
    public.cak_narrative_events
    to service_role;
