-- v1.9.23 evidence governance / OOS calibration / provider negative cache
-- Applied to production on 2026-08-15. Idempotent for reproducible environments.

create table if not exists public.cak_source_documents (
  document_id text primary key,
  ticker text not null,
  document_type text not null,
  source_family text not null,
  source_url text not null,
  published_at timestamptz,
  observed_at timestamptz not null default now(),
  source_https_verified boolean not null default false,
  entity_match_verified boolean not null default false,
  entity_match_method text,
  source_verified boolean not null default false,
  official_source boolean not null default false,
  content_sha256 text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint cak_source_documents_https_ck check (not source_https_verified or source_url like 'https://%')
);

create table if not exists public.cak_forward_evidence (
  evidence_id text primary key,
  ticker text not null,
  evidence_type text not null,
  evidence_date date,
  observed_at timestamptz not null default now(),
  title text,
  value_numeric numeric,
  unit text,
  horizon text,
  source_document_id text references public.cak_source_documents(document_id) on delete set null,
  source_url text not null,
  source_family text not null,
  source_quorum_count integer not null default 0,
  source_quorum_verified boolean not null default false,
  entity_match_verified boolean not null default false,
  source_verified boolean not null default false,
  evidence_confidence numeric,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  constraint cak_forward_evidence_https_ck check (not source_verified or source_url like 'https://%'),
  constraint cak_forward_evidence_quorum_ck check (not source_quorum_verified or source_quorum_count >= 2)
);

create table if not exists public.cak_management_capital_evidence (
  evidence_id text primary key,
  ticker text not null,
  evidence_type text not null,
  evidence_date date,
  observed_at timestamptz not null default now(),
  person_or_holder text,
  role_or_action text,
  ownership_pct numeric,
  source_document_id text references public.cak_source_documents(document_id) on delete set null,
  source_url text not null,
  source_family text not null,
  source_quorum_count integer not null default 0,
  source_quorum_verified boolean not null default false,
  entity_match_verified boolean not null default false,
  source_verified boolean not null default false,
  evidence_confidence numeric,
  payload jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  constraint cak_mgmt_capital_https_ck check (not source_verified or source_url like 'https://%'),
  constraint cak_mgmt_capital_quorum_ck check (not source_quorum_verified or source_quorum_count >= 2)
);

create table if not exists public.cak_provider_negative_cache (
  provider text not null,
  request_family text not null,
  cache_key text not null,
  failure_class text not null,
  http_status integer,
  retry_after timestamptz not null,
  hit_count integer not null default 1,
  last_error text,
  last_checked_at timestamptz not null default now(),
  last_success_at timestamptz,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key(provider, request_family, cache_key)
);

create table if not exists public.cak_guardrail_calibrations (
  calibration_id text primary key,
  strategy text not null,
  model_version text not null,
  calibration_state text not null,
  trained_through date,
  evaluation_start date,
  evaluation_end date,
  sample_count integer not null default 0,
  distinct_signal_dates integer not null default 0,
  fold_count integer not null default 0,
  objective_name text not null default 'RETURN_DRAWDOWN_STABILITY',
  objective_value numeric,
  parameters jsonb not null default '{}'::jsonb,
  metrics jsonb not null default '{}'::jsonb,
  active boolean not null default false,
  produced_at timestamptz not null default now(),
  created_at timestamptz not null default now()
);

create index if not exists idx_cak_source_documents_ticker on public.cak_source_documents(ticker,published_at desc);
create index if not exists idx_cak_forward_evidence_ticker on public.cak_forward_evidence(ticker,evidence_date desc);
create index if not exists idx_cak_management_capital_ticker on public.cak_management_capital_evidence(ticker,evidence_date desc);
create index if not exists idx_cak_negative_cache_retry on public.cak_provider_negative_cache(provider,request_family,retry_after);
create index if not exists idx_cak_guardrail_active on public.cak_guardrail_calibrations(strategy,active,produced_at desc);

alter table public.cak_source_documents enable row level security;
alter table public.cak_forward_evidence enable row level security;
alter table public.cak_management_capital_evidence enable row level security;
alter table public.cak_provider_negative_cache enable row level security;
alter table public.cak_guardrail_calibrations enable row level security;
revoke all on public.cak_source_documents from anon, authenticated;
revoke all on public.cak_forward_evidence from anon, authenticated;
revoke all on public.cak_management_capital_evidence from anon, authenticated;
revoke all on public.cak_provider_negative_cache from anon, authenticated;
revoke all on public.cak_guardrail_calibrations from anon, authenticated;
