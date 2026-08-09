-- IDX Emir Autonomous Scanner v1.4.0 permission repair.
-- Idempotent and non-destructive.

begin;
grant usage on schema public to service_role;
grant select, insert, update, delete on table
    public.cak_scan_runs,
    public.cak_radar_snapshots,
    public.cak_narrative_events,
    public.cak_provider_audit,
    public.cak_direct_evidence,
    public.cak_autonomous_evidence,
    public.cak_outcome_memory
    to service_role;
alter default privileges for role postgres in schema public
    grant select, insert, update, delete on tables to service_role;
commit;
