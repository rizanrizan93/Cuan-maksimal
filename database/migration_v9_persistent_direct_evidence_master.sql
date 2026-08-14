-- Durable direct-evidence master store for Emir Scanner.
-- Evidence lifetime is independent from scan snapshot retention/pruning.

create table if not exists public.cak_persistent_direct_evidence (
  evidence_key text primary key,
  ticker text not null,
  evidence_type text not null,
  observed_at timestamptz not null,
  source_verified boolean not null default false,
  source_url text not null default '',
  payload jsonb not null default '{}'::jsonb,
  content_hash text not null,
  source_scan_id text,
  freshness_policy_days integer not null,
  revoked boolean not null default false,
  revoked_at timestamptz,
  superseded_by text,
  first_seen_at timestamptz not null default now(),
  last_seen_at timestamptz not null default now(),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint cak_persistent_direct_evidence_type_chk check (
    evidence_type in (
      'BROKER_INVENTORY','OWNERSHIP_FREE_FLOAT','ORDERBOOK_BID_OFFER',
      'IDX_INTEGRITY_REGULATORY','OFFICIAL_FORWARD_EVENT'
    )
  ),
  constraint cak_persistent_direct_evidence_freshness_chk check (freshness_policy_days > 0)
);

create index if not exists idx_cak_persistent_direct_evidence_ticker_type
  on public.cak_persistent_direct_evidence(ticker,evidence_type,observed_at desc);
create index if not exists idx_cak_persistent_direct_evidence_active
  on public.cak_persistent_direct_evidence(evidence_type,observed_at desc)
  where source_verified and not revoked;

alter table public.cak_persistent_direct_evidence enable row level security;

create or replace function public.cak_evidence_freshness_days(p_type text)
returns integer language sql immutable as $$
  select case upper(coalesce(p_type,''))
    when 'BROKER_INVENTORY' then 35
    when 'OWNERSHIP_FREE_FLOAT' then 180
    when 'ORDERBOOK_BID_OFFER' then 5
    when 'IDX_INTEGRITY_REGULATORY' then 60
    when 'OFFICIAL_FORWARD_EVENT' then 540
    else 30 end;
$$;

create or replace function public.promote_verified_direct_evidence()
returns trigger language plpgsql security definer set search_path=public as $$
declare
  v_url text;
  v_key text;
  v_hash text;
begin
  if new.source_verified is not true then
    return new;
  end if;
  if upper(coalesce(new.evidence_type,'')) not in (
    'BROKER_INVENTORY','OWNERSHIP_FREE_FLOAT','ORDERBOOK_BID_OFFER',
    'IDX_INTEGRITY_REGULATORY','OFFICIAL_FORWARD_EVENT'
  ) then
    return new;
  end if;

  v_url := coalesce(new.payload->>'source_url', new.payload->>'url', '');
  v_hash := md5(coalesce(new.payload::text,''));
  v_key := md5(
    upper(coalesce(new.ticker,'')) || '|' || upper(coalesce(new.evidence_type,'')) || '|' ||
    coalesce(v_url,'') || '|' || coalesce(new.observed_at::text,'')
  );

  insert into public.cak_persistent_direct_evidence(
    evidence_key,ticker,evidence_type,observed_at,source_verified,source_url,payload,
    content_hash,source_scan_id,freshness_policy_days,first_seen_at,last_seen_at,updated_at
  ) values (
    v_key,upper(new.ticker),upper(new.evidence_type),new.observed_at,true,v_url,new.payload,
    v_hash,new.scan_id,public.cak_evidence_freshness_days(new.evidence_type),now(),now(),now()
  )
  on conflict (evidence_key) do update set
    payload=excluded.payload,
    content_hash=excluded.content_hash,
    source_scan_id=excluded.source_scan_id,
    last_seen_at=now(),
    updated_at=now(),
    source_verified=true;
  return new;
end;
$$;

drop trigger if exists trg_promote_verified_direct_evidence on public.cak_direct_evidence;
create trigger trg_promote_verified_direct_evidence
after insert or update of source_verified,payload,observed_at,evidence_type
on public.cak_direct_evidence
for each row execute function public.promote_verified_direct_evidence();

-- Backfill currently retained verified direct evidence. Historical issuer forward
-- events that were already pruned should be restored separately from their
-- surviving audited narrative rows before the next production scan.
insert into public.cak_persistent_direct_evidence(
 evidence_key,ticker,evidence_type,observed_at,source_verified,source_url,payload,
 content_hash,source_scan_id,freshness_policy_days,first_seen_at,last_seen_at,updated_at
)
select
 md5(upper(d.ticker)||'|'||upper(d.evidence_type)||'|'||coalesce(d.payload->>'source_url',d.payload->>'url','')||'|'||coalesce(d.observed_at::text,'')),
 upper(d.ticker),upper(d.evidence_type),d.observed_at,true,
 coalesce(d.payload->>'source_url',d.payload->>'url',''),d.payload,
 md5(coalesce(d.payload::text,'')),d.scan_id,public.cak_evidence_freshness_days(d.evidence_type),
 coalesce(d.created_at,now()),now(),now()
from public.cak_direct_evidence d
where d.source_verified is true
  and upper(coalesce(d.evidence_type,'')) in (
    'BROKER_INVENTORY','OWNERSHIP_FREE_FLOAT','ORDERBOOK_BID_OFFER',
    'IDX_INTEGRITY_REGULATORY','OFFICIAL_FORWARD_EVENT'
  )
on conflict (evidence_key) do update set
  payload=excluded.payload,
  content_hash=excluded.content_hash,
  last_seen_at=now(),
  updated_at=now();
