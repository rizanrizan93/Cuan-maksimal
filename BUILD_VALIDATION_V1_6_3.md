# Build Validation — v1.6.3

## Static and regression

```text
python compileall: PASS
pytest: 106 passed, 0 failed
```

Coverage includes the pre-existing engine/provider/cache/job/database tests plus:

- All-eligible progressive deep-review selection;
- Fast Top 30, Balanced Top 60, and Custom scopes;
- one-click scan/rescan control presence;
- database transfer reconciliation;
- dashboard factors included in radar database payloads.

## Controlled 20-ticker resumable integration

```text
input: 20
ticker radar rows: 20
OHLCV eligible deep target: 18
OHLCV unavailable: FINN.JK, MFIN.JK
KSEI processed: 18
news processed: 18
fundamentals processed: 18
transient cache failure retry: PASS
resume after JSON/session reconstruction: PASS
mean-of-empty warnings: 0
date-parser warnings: 0
```

The two missing-price tickers remained fail-closed. This is controlled fixture data, not a live trading scan.

## Progressive 300-ticker resumable integration

```text
tickers: 300
chunk size: 20
checkpoints to completion: 63
OHLCV calls: 16 (benchmark + 15 chunks)
KSEI target processed: 300
news target processed: 300
fundamental target processed: 300
deep reviewed: 300
resume after simulated disconnect: PASS
final radar rows: 300
```

The Top-3 summary field was persisted in job `result_summary`; the fixture generated no valid non-blocked Top-3 and therefore stored an empty list rather than forcing rejected candidates.

## 400-ticker engine acceptance

```text
tickers: 400
bars per ticker: 760
feature_state OK: 400/400
finite feature rows: 400/400
valid execution plans: 400
invalid entry–SL–TP hierarchy: 0
production-gate bypass: 0
engine elapsed: approximately 5.14 seconds
peak RSS: approximately 127.34 MB
```

The 40 deep-reviewed rows in this isolated engine benchmark are controlled fixtures used to exercise deep-profile logic; progressive breadth is validated separately by the 300-ticker job test.

## Database result-transfer validation

```text
result tables reconciled: 7
expected rows: 6
written rows: 6
verified rows: 6
state: DATABASE_RESULT_TRANSFER_VERIFIED
```

The captured `cak_radar_snapshots.payload` contained:

```text
emir_final_score
dashboard_flow_score
dashboard_silent_accum_score
dashboard_recommendation
```

## Top-3 dashboard validation

```text
source rows: 20
Top-3 rows: 3
blocked/reject included: false
Final Score alias: verified
OHLCV proxy disclosure: present
mobile CSS: present
HTML escaping: covered by pytest
```

## Fuzz

```text
random profiles: 2,000
crashes: 0
production-gate bypass: 0
invalid execution hierarchy: 0
```

## Live-data boundary

The build environment did not certify live Yahoo, KSEI, Google News, or Supabase HTTP. Deployment acceptance still requires a Streamlit Cloud smoke scan and inspection of the database transfer reconciliation table. No claim of zero live-provider failure is made.

## Extracted-package validation

The root-ready ZIP was extracted into a clean directory and validated:

```text
manifest entries checked: 95
checksum/size errors: 0
compileall: PASS
pytest: 106 passed, 0 failed
20-ticker integration: PASS
progressive 300-ticker integration: PASS
database-transfer validation: PASS
400-ticker engine acceptance: PASS
2,000-profile fuzz: PASS
Top-3 dashboard validation: PASS
```
