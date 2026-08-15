-- Emir database security hardening. Idempotent and schema-compatible with v9.
begin;

alter function public.cak_evidence_freshness_days(text) set search_path = '';
alter function public.guard_ksei_source_cache_profile() set search_path = '';
alter function public.promote_verified_direct_evidence() set search_path = '';

revoke all on function public.cak_evidence_freshness_days(text) from public, anon, authenticated;
revoke all on function public.guard_ksei_source_cache_profile() from public, anon, authenticated;
revoke all on function public.promote_verified_direct_evidence() from public, anon, authenticated;
grant execute on function public.cak_evidence_freshness_days(text) to service_role;
grant execute on function public.guard_ksei_source_cache_profile() to service_role;
grant execute on function public.promote_verified_direct_evidence() to service_role;

revoke all on table public.cak_persistent_direct_evidence from anon, authenticated;
grant select, insert, update, delete on table public.cak_persistent_direct_evidence to service_role;

commit;
