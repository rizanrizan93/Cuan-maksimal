# Release Notes — v1.7.0 Emir Methodology Hardening

## Why this release exists

An independent review of the v1.6.4 Top-3 output exposed several cases where the scanner's technical/flow detection was useful but the analytical semantics were not yet strong enough for real-money ranking. The most material findings were: QoQ growth presented as generic growth; ambiguous DER semantics; short-horizon inventory being allowed to resemble early accumulation even in markup; a dashboard ranking order that could conflict with the engine state; and a single midpoint RR shown for two materially different execution styles.

## P0/P1 fixes

1. **Growth basis hardening** — same-quarter YoY, QoQ, and TTM are separated. Backward-compatible growth fields prefer YoY and record the basis.
2. **Solvency semantics** — interest-bearing debt/equity, liabilities/equity and net debt/equity are separate fields.
3. **Fundamental pillar** — business/future-fundamental conversion is a direct 16% conviction pillar and production gate.
4. **Cash-quality expansion** — TTM OCF/FCF, OCF conversion and profitability sub-scores feed fundamental conversion when available.
5. **Multi-horizon inventory** — 20/60/120/252/504/756-bar context distinguishes collection, release and re-accumulation.
6. **Lifecycle hardening** — advanced markup without convergence becomes `WAIT_REACCUMULATION`, not automatic inventory collection.
7. **Macro/sector hierarchy** — risk-off remains restrictive but a strict sector-leader exception is permitted with capped sizing.
8. **Execution geometry** — accumulation and breakout/retest plans now carry separate risk/reward calculations.
9. **Dashboard ranking integrity** — decision-state priority precedes small score differences.
10. **Proxy coverage fix** — OBV slope and pullback-volume-contraction fields used by the broker fallback are now emitted by the market feature engine.

## New/updated decision states

- `EMIR_FUNDAMENTAL_EVIDENCE_PENDING`
- `EMIR_WAIT_FUNDAMENTAL_CONVERSION`
- `EMIR_WAIT_REACCUMULATION`

Existing production gates remain intentionally strict. This release is designed to reduce false positives, not manufacture more `BUY` signals.

## Public-method registry additions

- `EP32_MULTIYEAR_INVENTORY_CONTEXT`
- `EP33_MACRO_FIRST_SECTOR_LEADER_EXCEPTION`
- `EP34_GROWTH_BASIS_AND_SOLVENCY_SEMANTICS`
- `EP35_DUAL_EXECUTION_GEOMETRY`
- `EP36_BUSINESS_FUNDAMENTAL_PILLAR`

All remain either `PUBLIC_SYNTHESIS` or `EMPIRICAL_PROXY`. No proprietary formula is claimed.

## Validation

- 117/117 tests passed.
- 400/400 synthetic feature rows valid.
- 0 production-gate bypasses.
- 400/400 execution plans satisfy hierarchy checks.
- Top-3, database-transfer and resumable-scan validations pass.

## Compatibility

No Supabase migration is required from v1.6.4. Cache/scanner versions are bumped so old semantic outputs do not masquerade as v1.7 results.
