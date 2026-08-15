-- Emir v1.9.21: keep terminal job transitions constant-time.
--
-- The legacy trigger synchronously resolved up to 5,000 outcomes, seeded a new
-- outcome cohort, deduplicated research memory and cascaded deletion of older
-- scan snapshots inside the PATCH that marks a job terminal.  On Supabase Free
-- this exceeded statement_timeout and rolled back valid COMPLETED transitions.
-- Outcome and storage maintenance now run from the application only after the
-- terminal checkpoint has committed, with bounded best-effort RPC calls.

begin;

drop trigger if exists trg_cak_free_tier_housekeeping
on public.cak_scan_jobs;

comment on function public.cak_free_tier_housekeeping_trigger() is
  'Legacy synchronous terminal hook retained for migration compatibility only; not attached to cak_scan_jobs as of Emir v1.9.21.';

commit;
