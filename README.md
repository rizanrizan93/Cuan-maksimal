# IDX Emir Autonomous Scanner v1.6.3

## Progressive deep review + dashboard scan control + database transfer audit

v1.6.3 removes the fixed default Top-30 deep-review ceiling. The scanner now supports four scopes:

- **All eligible (progressive)** — default; every ticker with valid OHLCV enters deep review.
- **Balanced Top 60** — shorter daily workflow.
- **Fast Top 30** — fastest research workflow.
- **Custom limit** — user-selected cap.

The target list is processed in resumable chunks. KSEI, public news, and fundamentals are fetched/cache-read per chunk, then checkpointed to `cak_scan_jobs` and `cak_scan_job_chunks`. A disconnected mobile/browser session can resume from the last committed stage and offset.

## One-click scan controls

After uploading a ticker CSV, the main dashboard provides:

```text
🚀 Mulai Scan
🔄 Scan Ulang Universe Ini
Proses 1 checkpoint
Lanjut otomatis
Jeda
```

The **Mulai Scan** button creates the database job and immediately enables automatic checkpoint processing. The **Top 3 Report Card** tab also contains **Scan Ulang dari Dashboard**.

## Data moved to Supabase

Final scan results are written by `scan_id` to:

```text
cak_scan_runs
cak_radar_snapshots
cak_narrative_events
cak_provider_audit
cak_direct_evidence
cak_autonomous_evidence
cak_outcome_memory
```

Source/cache and resumable state remain in:

```text
cak_ohlcv_cache
cak_source_cache
cak_scan_jobs
cak_scan_job_chunks
```

Dashboard-derived fields such as `emir_final_score`, Flow, Silent Accum, Momentum, and recommendation labels are included inside each radar snapshot payload. The derived Top-3 summary is also stored in `cak_scan_jobs.result_summary`.

The dashboard displays a **Final result transfer reconciliation** table with:

```text
expected_rows
written_rows
verified_rows
transfer_state
```

Possible transfer states include:

```text
VERIFIED_IN_DATABASE
VERIFIED_EMPTY
WRITTEN_NOT_FULLY_VERIFIED
PARTIAL_DATABASE_TRANSFER
NOT_CONFIRMED
```

Database persistence is best-effort and never fabricates missing analytical evidence. Partial persistence does not turn a failed provider or missing field into a valid score.

## Top 3 Report Card

The Top-3 report uses the scanner's existing data:

- Final Score = `emir_conviction_score` / `emir_final_score`;
- Narrative, Flow, Silent Accum, Smart Money, Structure, Momentum, Future Fundamental, Liquidity;
- entry zone, trigger, stop, TP1/TP2, risk-reward;
- explicit OHLCV-proxy disclosure when broker/order-book data are not direct;
- blocked/reject states are excluded rather than forced into the Top 3.

## Database

No new migration is required from v1.6.2. Schema remains:

```text
emir_autonomous_schema_v7
HEALTHY_EMIR_DATABASE_V7
11/11 tables readable
```

Existing Supabase secrets remain unchanged:

```toml
CAK_DATABASE_ENABLED = "true"
CAK_DATABASE_SCHEMA = "public"
SUPABASE_URL = "https://PROJECT.supabase.co"
SUPABASE_SECRET_KEY = "sb_secret_..."
```

## Deployment

1. Replace the repository contents with the root-ready ZIP contents.
2. Ensure `app.py`, `resumable_scan.py`, `dashboard_persistence.py`, and `top3_dashboard.py` are at repository root.
3. Commit to `main`.
4. Reboot Streamlit.
5. Upload the ticker CSV.
6. Select the deep-review scope.
7. Press **Mulai Scan**.
8. If the session disconnects, reopen the same CSV and press **Lanjut otomatis**.
9. After completion, inspect the database-transfer metrics and the **Database** tab.

## Validated boundaries

- The scanner does not run a background worker; a checkpoint only runs while a Streamlit session is active.
- Resume continues from the last database checkpoint.
- Live Yahoo/KSEI/Google/Supabase connectivity cannot be certified from the build container; deployment still requires a live smoke scan.
- Broker inventory and bid-offer remain explicitly labelled proxies unless direct evidence is supplied.
- Output is decision support, not automated trade execution.
