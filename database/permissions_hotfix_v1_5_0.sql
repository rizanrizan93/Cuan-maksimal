grant usage on schema public to service_role;
grant select, insert, update, delete on table
    public.cak_ohlcv_cache,
    public.cak_source_cache
    to service_role;
