-- Close the legacy privileged helper and cover the manifest foreign key.

revoke all on function public.rls_auto_enable() from public,anon,authenticated;
grant execute on function public.rls_auto_enable() to service_role;

create index if not exists cak_idx_payload_manifest_endpoint_idx
  on public.cak_idx_payload_manifest(endpoint_key);
