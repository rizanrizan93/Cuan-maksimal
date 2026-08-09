# IDX Emir Scanner v1.9.9 — Evidence Integrity

- Recalculates cached fundamental coverage from finite, observed fields after database-first reconciliation.
- Preserves official IDX evidence separately from public financial-history proxies; proxy data is never promoted to official.
- Fills only missing OCF/FCF and related fields, retaining the stronger existing value and field-level provenance.
- Recognizes current `idx.id` regulator URLs and tries the current static host before the legacy `idx.co.id` fallback.
- Keeps business momentum in the Next Leader projection and keeps real-money execution fail-closed without direct bid/offer and official/current evidence.
- Bumps durable job/persistence contracts to `1.9.9-evidence-integrity`.

Validation: `validation_v1_9_9_evidence_integrity.py` plus guarded-real-money, resumable-300, and Top-3 suites.
