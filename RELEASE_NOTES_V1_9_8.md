# IDX Emir Autonomous Scanner v1.9.8 — Persistence Integrity

## Root causes repaired

1. Blank or `NaT` fundamental periods were preserved as strings and posted to
   `cak_research_memory.effective_period` (`date`), causing entire chunks to fail.
2. Final scan tables could be exact-count verified while research memory was
   partial, yet the job still reported `SCAN_COMPLETED_FULL_PERSISTENCE`.
3. Trigger functions used a mutable search path and privileged database helper
   functions were executable by API roles.

## Fix

- Typed dates/timestamps normalise to ISO values or SQL `NULL` before writes.
- Full persistence now requires exact verification of both final scan tables and
  durable research memory.
- The schema-v8 security hotfix fixes trigger search paths, revokes API execution,
  and adds the missing narrative `scan_id` foreign-key index.

The v1.9.6 scoring calibration and v1.9.7 universe metadata/hash contract remain
unchanged.
