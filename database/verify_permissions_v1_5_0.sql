select
    has_table_privilege('service_role', 'public.cak_ohlcv_cache', 'select') as ohlcv_select,
    has_table_privilege('service_role', 'public.cak_ohlcv_cache', 'insert') as ohlcv_insert,
    has_table_privilege('service_role', 'public.cak_ohlcv_cache', 'update') as ohlcv_update,
    has_table_privilege('service_role', 'public.cak_source_cache', 'select') as source_select,
    has_table_privilege('service_role', 'public.cak_source_cache', 'insert') as source_insert,
    has_table_privilege('service_role', 'public.cak_source_cache', 'update') as source_update;
