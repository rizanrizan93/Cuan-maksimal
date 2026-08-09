grant usage on schema public to service_role;
grant select, insert, update, delete on table
    public.cak_scan_jobs,
    public.cak_scan_job_chunks
    to service_role;
