# Validation Audit — v1.9.8

- Pytest: 177 passed.
- Resumable synthetic 300: 300 rows, 66 checkpoints, disconnect recovery: PASS.
- Synthetic 400 hierarchy/gate validation: PASS.
- Realistic 20, guarded-real-money 400, database transfer, and dashboard: PASS.
- Blank/`NaT` effective-period regression: PASS.
- False full-persistence regression: PASS.
- Production schema-v8 hotfix: applied successfully.
- Supabase security advisor after migration: 0 WARN/ERROR findings.

Existing historical scans are retained. A new v1.9.8 scan is required to rebuild
the previously rejected fundamental-memory chunks with the corrected contract.
