# Emir v9 runtime hotfix validation

This marker records the 2026-08-14 production correction:

- `app.py` now accepts `HEALTHY_EMIR_DATABASE_V9`, matching `persistence.py`.
- The misleading schema-v8 readiness copy was updated to schema v9.
- Smart Money Cost Basis v1.0.0 is exposed in Emir Radar, Inventory & Smart Money, chart levels/diagnostics, and Top 3 runtime labelling.
- Estimated smart-money cost remains evidence-only and does not alter conviction/ranking weights.
- Direct verified broker cost evidence takes precedence; otherwise the scanner labels the value as a proxy.

This commit intentionally triggers the existing production-hardening workflow and Streamlit runtime smoke on `main`.
