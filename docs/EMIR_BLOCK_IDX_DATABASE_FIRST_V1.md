# EMIR Block IDX database-first contract v1

## Outcome

The interactive EMIR scanner is a database consumer. It must not call Yahoo, Google, KSEI, IDX, ZAPI, or another market provider while a user is scanning. Official acquisition runs in the scheduled GitHub producer and writes only to the dedicated EMIR v2 Supabase project.

- Dedicated project ref: `nredhspgvrqapycpakay` (new EMIR/Cuan-maksimal target)
- Official primary API host: `https://block.idx.id` (official XBRL attachment bytes may be served by IDX-owned `idx.id`/`idx.co.id` static hosts)
- Historical bootstrap window as of 2026-09-15: `2026-03-15` through `2026-09-15`
- Interactive data mode: `CAK_SCAN_DATABASE_ONLY=1`
- Feature contract: `EMIR_BLOCK_IDX_FEATURE_V1`

The Shared Evidence Hub, PASTICUAN tables, and IDX Flow `flow_*` tables are not ranking inputs for this contract. The Block IDX endpoint supplies official discovery/metadata; downloaded financial attachment bytes retain their original official IDX URL.

## Acquisition tiers

| Class | Treatment | Examples |
|---|---|---|
| A | Mandatory bulk, once per completed session | stock summary, index summary |
| B | Incremental/event range; deduplicate by source reference and SHA-256 | announcements, UMA, suspension, issued history, financial report index |
| C | Slow reference or rotating snapshot | companies, company profile |
| D | Conditional/finalist or market diagnostic | trading info, company announcement, market-wide broker summary |
| E/F | Do not spend normal runtime calls or do not infer ticker facts | top movers, duplicate news, market-wide broker totals |
| G | Outside common-stock scanner scope | bonds and derivatives |

Only direct routes that have a verified contract are enabled. Other known IDX dataset families remain `CATALOG_ONLY`; the producer does not guess a URL. This is how “use every endpoint” remains auditable without calling redundant, unsafe, or out-of-scope routes.

## Six-month load

The backfill is date-major:

1. For each weekday from 2026-03-15 through 2026-09-15, request StockSummary, IndexSummary, and BrokerSummary once.
2. Validate requested date, payload completeness, non-negative market quantities, official host, and no redirects.
3. Upsert normalized facts by natural key. A rerun is idempotent.
4. Fetch event families for the whole bounded range and deduplicate by event key plus payload hash.
5. Refresh company reference and financial-report index.
6. Compute the preliminary all-market rank.
7. Download/parse official XBRL for the leading 60 names, then recompute the final rank.
8. Keep non-session weekdays as explicit no-data/failure telemetry; never synthesize or forward-fill them.

Manual workflow input:

```text
mode: backfill
from_date: 2026-03-15
to_date: 2026-09-15
fundamental_limit: 60
```

Daily workflow runs at 17:30 WIB and repeats idempotently at 18:30 WIB to tolerate delayed publication. The database also owns a private core-market fallback at 17:45 and 18:45 WIB via pg_cron; it loads the same three verified bulk endpoints, reruns ranking, and prunes storage without exposing an HTTP webhook.

## Ranking

`emir_score` is a cross-sectional decision-priority score, not a promised return:

| Component | Weight | Official inputs |
|---|---:|---|
| Fundamental growth/quality/solvency | 30% | revenue and earnings growth, ROE, ROA, NPM, OCF, FCF, debt/equity, current ratio |
| Smart-money proxy | 25% | official foreign net flow, positive-flow persistence, turnover |
| Momentum/structure | 20% | 5D/20D/60D return, close–MA20–MA60 stack, prior 60-session resistance |
| Liquidity | 10% | ADTV20, frequency, observation completeness |
| Market regime | 10% | official COMPOSITE 20-session direction |
| Risk/catalyst state | 5% | suspension, UMA, dilution and bid/offer spread |

BrokerSummary is market-wide only. It is persisted for market/member diagnostics and never treated as stock-level broker accumulation or bandar identity.

## Execution Top 3

A name enters the eligible pool only when all conditions are true:

- at least 80 official observations;
- ADTV20 at least IDR 1 billion;
- bullish close > MA20 > MA60;
- positive 20-session foreign net and at least 11 positive days;
- official fundamental coverage at least 50%;
- no unresolved suspension and no recent dilution event;
- bid/offer spread no more than 2%;
- stop and TP1 come from observed support/resistance plus ATR geometry;
- RR to TP1 is at least 1.8.

Top 3 is selected by 35% smart money, 30% momentum, 20% liquidity, and 15% risk quality. If fewer than three names pass, the database returns fewer than three; it never fills the list with a blocked setup.

## Storage and speed

The normalized six-month market panel is the reusable fact layer. Daily rank and Top-3 tables are denormalized serving layers. The dashboard normally performs two small reads: latest Top‑50 and latest Top‑3. Full chart/scoring loads read bounded five-ticker chunks from the market table, with no provider fallback.

Raw payload bodies are not duplicated indefinitely. The manifest retains endpoint, date, URL, row counts, payload SHA‑256, validation state, producer version, and fetch time. Event rows preserve their raw record for forensic review.

The hard quota is 500 MiB (524,288,000 bytes), with warning at 420 MiB and an ingestion hard-stop at 470 MiB. Every producer run prunes before and after acquisition: market/rank/manifest data keep six calendar months, endpoint audit data 30–90 days, events six calendar months, only two company snapshots and eight fundamental snapshots per ticker, and only the five latest legacy radar runs.

The supplied legacy gzip was valid as a compressed stream but its logical SQL ended inside the `cak_research_memory` COPY block. It therefore cannot be restored blindly. Migration v31 creates compatibility tables and records a SHA-256 salvage manifest; only completed COPY blocks and compact, validated recent/latest rows are imported.

## Activation sequence

1. Apply `database/migration_v30_emir_block_idx_database_first.sql`.
2. Apply `database/migration_v31_emir_backup_salvage_storage_guard.sql`.
3. Apply `database/migration_v32_emir_block_idx_server_eod.sql`.
4. Run `database/verify_v30_emir_block_idx_database_first.sql`.
5. Point GitHub secrets `SUPABASE_URL` and `SUPABASE_SECRET_KEY` to project `nredhspgvrqapycpakay`.
6. Dispatch the EOD workflow once in backfill mode with the exact dates above.
7. Verify session coverage, endpoint manifests, storage state, rank count, zero guardrail bypasses, and Top‑3.
8. Merge/deploy the application with `CAK_SCAN_DATABASE_ONLY=1`.

The target project is active and its compact legacy salvage is validated. A completed six-month official Block IDX backfill is still a distinct operational checkpoint and must be verified from ingestion manifests before it is claimed complete.
