# IDX Emir Autonomous Scanner v1.9.8 — Persistence Integrity

Clean-room public-framework implementation. It is not affiliated with Emir Parengkuan and does not claim to reproduce a proprietary CAK formula.

## Production path

Macro/regime → sector opportunity → business/future fundamental → official filing reconciliation → narrative–money flow → inventory/smart-money behaviour → SMC/ICT execution → real-money authorization gate.

## What v1.9 changes

### 1. Official-first fundamental deep review
For the top deep-review shortlist (default 60), the scanner attempts the official IDX XBRL `instance.zip` for the newest plausible reporting period. A verified same/newer-period IDX filing is authoritative over Yahoo/public statement proxies. The proxy remains a cross-check and TTM helper, never an “official” source by inference.

Official evidence records period, URL, source verification, revenue/net income, comparable YoY growth, OCF, capex/FCF proxy, balance-sheet fields and solvency ratios where available. Missing fields remain missing.

### 2. Durable official research memory
Official filing payloads are cached in `cak_source_cache` under `IDX_FUNDAMENTAL:*` for fast reuse and stored as versioned `IDX_OFFICIAL_FUNDAMENTAL` evidence in `cak_research_memory`. The original period/source/hash remains available across scans.

### 3. Blended market regime
Direct IHSG context remains primary (60%) and universe breadth contributes 40%. The result is fail-closed when both are weak, but a single soft benchmark metric no longer forces every stock into RISK_OFF when breadth remains healthy.

### 4. GUARDED_REAL_MONEY
Default UI capital mode is `GUARDED_REAL_MONEY`.

- risk budget is clamped to **0.75% per idea**;
- manual-confirmation position cap is **max 7.5%**, and **max 5% in RISK_OFF**;
- missing/stale official fundamental, missing cash-flow evidence, high leverage, poor liquidity, distribution, crowding, weak structure or inadequate RR blocks authorization;
- an **EOD bid/offer proxy can never make `production_ready=True`**;
- scanner-side `DIRECT_VERIFIED_READY` additionally requires direct IDX integrity evidence and direct/live bid-offer evidence;
- without direct evidence, a fully qualified idea becomes `REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED`, not an automatic order;
- no averaging down below thesis/price invalidation. Adds are permitted only after confirmation.

## Two leaderboards remain separate

- **The Next Leader**: business/future-fundamental quality and runway.
- **Execution Top 3**: current inventory/flow/structure/liquidity/entry geometry.

The new **Real Money Gate** tab is the final authorization layer.

## Persistence layers

1. `cak_ohlcv_cache` — reusable OHLCV.
2. `cak_source_cache` — latest KSEI/news/Yahoo fundamental/IDX official fundamental payloads.
3. `cak_research_memory` — durable content-hashed evidence history across scans.
4. `cak_scan_jobs` / `cak_scan_job_chunks` — resumable checkpoints.
5. final point-in-time radar/evidence tables — auditable scan decisions.

## Database

v1.9 continues to use **schema v8**. No destructive migration and no new DB migration are required if v8 is already healthy. Run `database/verify_v8.sql` before the first real-money scan.

## Validation in the build environment

- pytest: **175 passed**
- v1.9-specific tests: XBRL parse, official/proxy reconciliation, OCF/FCF extraction, market blend, durable official memory, EOD-proxy real-money block, direct-verified risk/cap bounds
- synthetic 400 core engine: **400/400 valid**, 0 hierarchy errors, 0 gate bypass
- guarded real-money 400: **399 EOD-proxy rows → 0 production-ready**, 1 direct-verified control → 1 ready; max risk **0.75%**, max position cap **7.0%** in fixture
- fuzz: **2,000 scenarios, 0 crashes, 0 gate bypasses**
- resumable 300 with official-XBRL stage: **PASS**, official stage limited to 60 tickers
- database final-result transfer: **exact verified**
- realistic transient-cache retry: **PASS**

The local build environment could not connect to live IDX static files, so live IDX XBRL connectivity must be smoke-tested in the deployed Streamlit environment. Failure remains fail-closed: it blocks real-money authorization rather than substituting fabricated values.

See `RELEASE_NOTES_V1_9_7.md`, `VALIDATION_AUDIT_V1_9_7.md`,
`REAL_MONEY_GUARDRAILS_V1_9_0.md`, and `OFFICIAL_IDX_XBRL_CONTRACT_V1_9_0.md`.


## v1.9.3 Next Leader regression fix

- `The Next Leader` is now a business/future-fundamental leaderboard and no longer uses execution/real-money decision states as eligibility blockers.
- `EMIR_DATA_INTEGRITY_BLOCK`, smart-money distribution, retail euphoria, or other timing states remain visible as execution overlays but do not erase a fundamentally qualified issuer from the long-horizon ranking.
- IDX official XBRL is optional by default. Yahoo/public financial statements remain the normal scoring path; official evidence upgrades confidence when available.
- If the leaderboard is empty, the UI now reports counts of finite fundamental scores, minimum coverage, and data-quality so the cause is diagnosable.
- No database migration and no persistent-cache schema change.

## v1.9.1+ proxy-tolerant policy
Official IDX XBRL is a confidence upgrade, not a prerequisite for scoring. Current/aligned Yahoo/public financial statements may score and become manual-confirmation candidates. Proxy-only guarded risk is capped at 0.50% and 3% position cap; direct-ready remains strict.


## v1.9.3 Fundamental join integrity fix
- Preserves ticker keys through proxy/official fundamental reconciliation.
- Prevents silent empty Next Leader when source fundamentals are valid.
- Adds `FUNDAMENTAL_JOIN_INTEGRITY_FAILURE` hard gate.
- Adds chunk-level research-memory retry and truthful 0/N verification UI.
- No database migration; schema remains v8.

## v1.9.4 Real Money calibration
- strong diversified public narrative is a MANUAL-confirmation condition rather than an automatic hard block
- direct/official evidence remains mandatory for DIRECT_VERIFIED_READY
- candidate quality is separated from entry timing via `real_money_entry_candidate`
- Real Money Gate includes a separate actionable `Real Money Candidate Top 3`
- Execution Top 3 is explicitly labeled `Execution Research Top 3`
- proxy/manual risk remains <=0.50% and cap <=3%


## v1.9.5 — Fundamental freshness & YTD consistency

- Fundamental cache schema v4 adds YTD growth/consistency evidence.
- Absolute freshness is tighter and cross-sectional period lag is calibrated against the active shortlist.
- A strong standalone quarter with still-negative YTD becomes `TURNAROUND_INFLECTION_UNCONFIRMED`, not fully confirmed growth.
- Next Leader applies evidence penalties; guarded real-money remains stricter.
- Database schema remains v8.


## v1.9.6 — Sector-aware business-quality calibration

- Next Leader is now business-heavy: fundamental 30%, confirmed business momentum 20%, data quality 5%, future-fundamental narrative/conversion/alignment 27%, sector 9%, execution/flow overlays only 9%.
- Bank issuers no longer gain a false quality advantage from generic corporate OCF/FCF, current-ratio, cash/debt or DER proxies. Until NPL/NIM/CAR/LDR/BOPO coverage exists, the bank model is `BANK_GENERIC_PROXY_LIMITED` and research-only for Real Money Top 3.
- Confirmed broad YTD growth receives a small +3 quality adjustment; loss-making growth is capped/penalized; earnings-led low-topline cases receive review flags.
- Moderate distribution no longer directly penalizes Next Leader; extreme distribution still receives a modest haircut. Execution/Real Money gates remain strict.
- Manual proxy-backed Real Money candidates display risk budget <=0.50% even when the parent scan is launched in RESEARCH mode.
- No database migration. Fundamental cache schema remains v4.

## v1.9.7 — Resumable universe integrity

- The complete allow-listed universe metadata now survives CSV parsing, job
  creation, Supabase JSONB checkpoints, resume, final issuer context, and radar.
- `rank_universe`, `universe_role`, `priority`, and `yahoo_ticker` from the
  supplied 300-ticker CSV are retained.
- Universe hashes are order-independent but metadata-sensitive, preventing a
  stale resumable job from being reused after sector/theme/catalyst changes.
- Blank ticker cells no longer become the false symbol `NAN.JK`.
- Database/checkpoint failures stop safely in Streamlit and keep committed
  progress resumable.
- Schema v8 verification now checks all 12 runtime tables and service-role
  SELECT/INSERT/UPDATE privileges.
- v1.9.6 analytical weights and cache schema v4 remain unchanged.

## v1.9.8 — Persistence integrity

- Blank, `NaT`, or invalid fundamental periods are stored as SQL `NULL`, never
  sent as invalid values to `cak_research_memory.effective_period`.
- A scan is reported as full persistence only when both final scan tables and
  durable research memory are exact-count verified.
- Database trigger functions use a fixed safe search path and privileged helper
  functions are no longer executable by API roles.
- v1.9.6 scoring weights, v1.9.7 universe metadata, and schema-v8 table contracts
  remain unchanged.
