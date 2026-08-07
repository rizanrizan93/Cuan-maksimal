# Files to Replace — v1.7.0

For an existing v1.6.4 deployment, the analytical/runtime files changed in v1.7.0 are:

```text
VERSION
autonomous_enrichment.py
data_providers.py
narrative_flow_engine.py
persistence.py
persistent_cache.py
resumable_scan.py
scan_jobs.py
top3_dashboard.py
```

For safest deployment, replace the repository root using the full root-ready ZIP rather than mixing versions.

Test files changed/added to validate the new semantics:

```text
tests/test_autonomous.py
tests/test_engine.py
tests/test_persistence.py
tests/test_top3_dashboard.py
validation_artifacts_v1_7_0/*
tests/test_release_entrypoint_v164.py
```

No database migration is required. The version bump intentionally invalidates analytical cache keys computed under v1.6.4 semantics.

Deployment guide: `DEPLOYMENT_SAFE_UPDATE_V1_7_0.md`.
