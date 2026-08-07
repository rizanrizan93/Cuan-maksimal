# IDX Emir Autonomous Scanner v1.7.0

## Emir-methodology hardening: business conversion → narrative/flow → inventory → execution

v1.7.0 is a methodology-hardening release built from the public concepts used in the project: macro/sector context, business and future-fundamental conversion, narrative, money flow, smart-money / inventory behaviour, ownership/liquidity, and disciplined execution. It is a **clean-room public synthesis plus empirical IDX proxies**; it is not affiliated with Emir Parengkuan and does not claim to reproduce a proprietary CAK formula.

The release fixes semantic problems found while independently reviewing the v1.6.4 Top-3 output and the 400-ticker small/mid-cap universe.

## What changed

### 1. Fundamental engine: QoQ is no longer mislabeled as general growth

The financial snapshot now separates:

- revenue growth YoY and QoQ;
- earnings growth YoY and QoQ;
- TTM revenue / net income / operating income;
- TTM OCF and FCF proxy;
- current and TTM net / operating margins;
- TTM ROE and ROA.

Backward-compatible `revenue_growth_pct` and `earnings_growth_pct` prefer same-quarter YoY and explicitly fall back to QoQ only when YoY is unavailable. `growth_basis_state` records the basis.

### 2. Solvency definitions are explicit

The old generic DER interpretation is no longer allowed to hide ratio-definition differences. The snapshot separates:

- interest-bearing debt / equity;
- total liabilities / equity;
- net debt / equity;
- current ratio;
- cash / debt.

`der_ratio` remains for compatibility but is explicitly defined as interest-bearing debt / equity.

### 3. Business/future fundamental is an independent conviction pillar

Fundamental conversion now contributes **16%** directly to the empirical conviction score and is also a production-readiness gate. Missing business evidence produces `EMIR_FUNDAMENTAL_EVIDENCE_PENDING`; weak conversion produces `EMIR_WAIT_FUNDAMENTAL_CONVERSION`.

Current empirical fixed-denominator pillars:

| Pillar | Weight |
|---|---:|
| Flow / inventory | 18% |
| Fundamental conversion | 16% |
| Narrative runway | 14% |
| Market structure | 13% |
| Sector + macro context | 10% |
| Financial/narrative conversion blend | 8% |
| Issuer alignment + ownership | 7% |
| IDX integrity | 5% |
| Order-book / EOD microstructure evidence | 4% |
| Trend | 3% |
| Liquidity + float | 2% |

These weights are an internal empirical implementation, not an official CAK formula.

### 4. Inventory proxy is multi-horizon

When direct broker data are unavailable, the OHLCV fallback now measures 20/60/120/252/504/756-bar behaviour. It distinguishes persistent collection, inventory release, bottoming-to-collection, and mixed cycles. It remains clearly labelled as a behavioural proxy and never identifies a broker or beneficial owner.

### 5. Markup is not silently re-labelled as early accumulation

A stock already in `MARKUP` with strong recent flow but insufficient story-flow convergence can become `MARKUP_REACCUMULATION_REQUIRED` / `EMIR_WAIT_REACCUMULATION`. The prescribed action is to wait for a healthy pullback/base or an accepted breakout-retest rather than chase.

### 6. Macro-first without a blanket market blocker

`RISK_OFF` remains a penalty and gate. A strict exception is available only for a `LEADING` sector with positive relative strength and positive sector-strength momentum. If it becomes executable under this exception, position size is capped at 5%.

### 7. Dual execution geometry

Accumulation and breakout/retest are separate plans. The scanner reports RR at entry-low and entry-high and builds a distinct breakout entry, stop, targets, and RR. The Top-3 dashboard no longer presents a midpoint `1:3` as though it applied to the entire entry zone.

### 8. Top-3 ranking follows decision-state priority

The dashboard sort now respects deep-review state, decision-state priority, final score, evidence coverage, silent accumulation / flow, then liquidity. A `NO_EDGE` ticker can no longer leapfrog a stronger actionable/watch state merely because of a small score difference.

## Progressive deep review

The resumable all-eligible architecture remains intact. Every eligible ticker can enter deep review, while balanced/fast/custom scopes remain available for shorter workflows. Scan checkpoints and database transfer behaviour are preserved.

## Database

No new database migration is required from v1.6.4. Schema remains `emir_autonomous_schema_v7`. Version/cache identifiers were raised to v1.7.0 so cached results computed under the old growth/inventory semantics are not silently reused as current analytical output.

## Validation

Final local validation for this release:

- `pytest`: **117 passed**;
- synthetic 400-ticker feature validation: **400/400 valid**;
- invalid production-gate bypass: **0**;
- valid execution plans: **400/400**;
- invalid execution hierarchy: **0**;
- Top-3 dashboard validation: **PASS**;
- database transfer validation: **PASS**;
- resumable 300-ticker validation: **PASS**.

The build environment cannot certify live Yahoo/KSEI/Google/Supabase network connectivity. A live smoke scan on the deployment environment is still required before trusting production output.

## Deployment

1. Replace the repository root with the root-ready ZIP contents.
2. Keep the existing Streamlit/Supabase secrets.
3. Commit to `main` and reboot Streamlit.
4. Upload the same 400-ticker CSV.
5. Run **All eligible (progressive)** for the benchmark scan.
6. Compare the v1.7 Top-20 / Top-3 with the independent benchmark report included in this release.
7. Treat only evidence-complete states as candidates for real-money execution; proxy-only states remain decision support.

See `RELEASE_NOTES_V1_7_0.md`, `METHODOLOGY_AUDIT_V1_7_0.md`, `FILES_TO_REPLACE_V1_7_0.md`, and `INDEPENDENT_400_BENCHMARK_V1_7_0.md`.
