# Emir v1.9.15 — Live Provider Smoke

Date: 2026-08-12 (Asia/Jakarta market session)
Environment: GitHub-hosted Ubuntu runner, Python 3.13, repository `main`.

## Result

PASS.

The smoke used real public provider calls from the current production repository, not synthetic fixtures.

- OHLCV requested: 20 IDX tickers + `^JKSE` = 21 symbols
- OHLCV ready: 21/21 (`OK`)
- IDX ticker OHLCV ready: 20/20
- Benchmark `^JKSE`: ready
- Market-feature computation: 20/20
- Public fundamental snapshots: 10/10 (`OK`)
- IDX official fundamental/XBRL checks: 3/3 (`OK`)
- Latest OHLCV session returned: 2026-08-12

Sample live feature states:
- `OMED.JK`: feature_state `OK`, last 234, ADTV20 ~Rp7.04B, smart-money score 80.2, inventory-cycle score 74.7, structure score 83.1.
- `ELSA.JK`: feature_state `OK`, last 705, ADTV20 ~Rp18.21B, smart-money score 70.3, structure score 82.1.
- `MARK.JK`: feature_state `OK`, last 1115, ADTV20 ~Rp18.41B, inventory-cycle score 76.1, structure score 81.5.

## Cold 400-ticker production verification

The first production scan against the replacement Supabase project completed with `SCAN_COMPLETED_FULL_PERSISTENCE`: 400 processed, 399 usable, 1 OHLCV failure (`SCPI.JK`), 399 official fundamentals verified, 400 future-fundamental snapshots, and exact research-memory verification.

A live storage audit found that a deployment path still wrote ephemeral broker/orderbook proxies and excessive historical memory. Supabase now has an independent server-side free-tier housekeeping trigger, while current application code already excludes the ephemeral proxy families from durable research memory. This commit intentionally refreshes the Streamlit deployment so the next warm scan tests both protections together.

## Dependency note

The first temporary smoke used Python 3.11 and stopped before scanner execution because the repository pins `numpy==2.5.1`, which requires Python >=3.12. The corrected run used Python 3.13 and passed. Streamlit deployment must therefore use a compatible Python version (>=3.12).

The temporary GitHub Actions workflow used for this check was removed after the successful run.
