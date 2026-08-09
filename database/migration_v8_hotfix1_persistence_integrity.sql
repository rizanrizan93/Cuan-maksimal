-- IDX Emir Autonomous Scanner v1.9.8 persistence/security integrity hotfix.
-- The runtime table contract remains schema v8. Idempotent and non-destructive.

begin;

create or replace function public.cak_touch_updated_at()
returns trigger
language plpgsql
security invoker
set search_path = pg_catalog
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

create index if not exists idx_cak_narrative_events_scan_id
    on public.cak_narrative_events (scan_id);

revoke execute on function public.cak_touch_updated_at() from public, anon, authenticated;

do $$
begin
    if to_regprocedure('public.set_scanner_updated_at()') is not null then
        execute 'alter function public.set_scanner_updated_at() set search_path = pg_catalog';
        execute 'revoke execute on function public.set_scanner_updated_at() from public, anon, authenticated';
    end if;
    if to_regprocedure('public.rls_auto_enable()') is not null then
        execute 'revoke execute on function public.rls_auto_enable() from public, anon, authenticated';
    end if;
end;
$$;

commit;
