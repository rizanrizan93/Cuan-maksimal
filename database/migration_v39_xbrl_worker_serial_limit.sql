-- Keep the free-plan Edge worker within CPU and memory limits after bounded live validation.
create or replace function public.cak_idx_kick_xbrl_worker_v1()
returns bigint
language plpgsql
security definer
set search_path = pg_catalog,public,vault,net
as $$
declare
  v_url text;
  v_key text;
  v_token text;
begin
  select decrypted_secret into v_url from vault.decrypted_secrets where name='emir_project_url' order by created_at desc limit 1;
  select decrypted_secret into v_key from vault.decrypted_secrets where name='emir_worker_anon_key' order by created_at desc limit 1;
  select decrypted_secret into v_token from vault.decrypted_secrets where name='emir_xbrl_worker_token' order by created_at desc limit 1;
  if v_url is null or v_key is null or v_token is null then return null; end if;
  return net.http_post(
    url:=v_url||'/functions/v1/emir-xbrl-worker',
    headers:=jsonb_build_object(
      'Content-Type','application/json','apikey',v_key,
      'Authorization','Bearer '||v_key,'x-emir-worker-token',v_token
    ),
    body:=jsonb_build_object('chain',true,'limit',3),
    timeout_milliseconds:=10000
  );
end
$$;

revoke all on function public.cak_idx_kick_xbrl_worker_v1() from public,anon,authenticated;
grant execute on function public.cak_idx_kick_xbrl_worker_v1() to service_role;

