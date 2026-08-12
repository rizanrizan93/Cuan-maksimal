# Test Report — Emir Scanner v1.9.15 Free Tier Storage Safety

- compileall: PASS
- free-tier retention tests: 4/4 PASS
- production-relevant full pytest from working tree: 204 PASS; the only prior non-production failure was Streamlit startup in an audit container without Streamlit installed
- synthetic 400 core validation: PASS; 400/400 feature state OK, 400/400 valid execution geometry, 0 hierarchy violations, 0 production-gate bypass
- research memory: semantic dedupe + compact payload + bounded per ticker/family
- scan history: bounded to two recent published scans and two recent terminal resumable jobs
- checkpoint audit payload: compact projection instead of full provider payload
- database outage classification: resource exhaustion/ECONNREFUSED is not mislabeled as schema missing

No scoring/SMC/inventory/Guarded Real Money change.
