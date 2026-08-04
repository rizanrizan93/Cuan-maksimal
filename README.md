# IDX Emir Autonomous Scanner v1.4.0

A ticker-only, clean-room implementation of publicly described Emir Parengkuan/CAK concepts, with independent IDX-specific data integrity, execution capacity, and outcome controls.

It is not affiliated with CAK Investment Club, does not contain membership material, and does not claim to reproduce an official CAK formula.

## Required user input

Only one CSV is required:

```csv
ticker
ADMR
MDKA
ELSA
RAJA
```

Optional direct-evidence CSVs remain available under **Advanced direct-evidence overrides**. They are not required for the autonomous EOD tier.

## Autonomous pipeline

```text
Ticker CSV
→ completed-session OHLCV
→ KSEI issuer profile, status, sector and corporate actions
→ public news and narrative events
→ public financial-statement proxy
→ IHSG market regime and sector leadership
→ market structure and OHLCV flow
→ broker-inventory behavioural proxy
→ EOD bid-offer/microstructure proxy
→ IDX integrity/regulatory-event overlay
→ thesis, invalidation, execution capacity and position sizing
→ Supabase persistence and exact readback
```

## Provider order

### OHLCV

1. Yahoo chart REST;
2. yfinance fallback;
3. KSEI public price-history fallback.

KSEI volumes are converted from lots to shares before feature calculation.

### Issuer and corporate action

KSEI public registered-security pages provide issuer name, activity sector, listing date, security status, number of securities, local/foreign registration composition, and corporate-action history.

### Narrative

- Yahoo public news;
- Google News RSS;
- KSEI corporate-action events;
- optional direct issuer/IDX evidence override.

Syndicated stories are deduplicated. Regulator-domain events can create a conservative integrity block; media-only regulatory mentions create caution pending confirmation.

### Fundamentals

The autonomous collector reads public quarterly statements exposed through yfinance and normalizes revenue, earnings, margins, debt/equity, operating cash flow, free-cash-flow proxy, and conversion quality. Missing data remains missing.

## Proxy boundaries

The following are explicit proxies, not direct market data:

```text
broker_inventory_evidence_type = OHLCV_PROXY
orderbook_evidence_type        = OHLCV_EOD_PROXY
```

The broker proxy uses accumulation persistence, absorption, close acceptance, CMF, OBV slope, up-value behaviour, and pullback-volume contraction. It never identifies a broker or beneficial owner.

The bid-offer proxy uses EOD acceptance, absorption, volume confirmation, breakout proximity, gaps, and execution friction. It is not live market depth.

## Readiness tiers

### `EMIR_AUTO_EOD_READY`

Autonomous public/proxy tier. Requires a supported thesis, flow, structure, public fundamental conversion, partial automatic IDX integrity, EOD microstructure proxy, and executable capacity. Position cap is limited to 8%.

### `EMIR_READY_WITH_PRECISE_TRIGGER`

Higher evidence tier. Still requires directly verified IDX integrity and direct bid-offer evidence.

No provider failure or missing evidence is converted into a neutral score of 50.

## Recommended modes

- `EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP`: full-universe OHLCV radar, then automatic deep enrichment for the top 20–40.
- `EMIR_AUTONOMOUS_DEEP_REVIEW`: automatic deep enrichment for all uploaded tickers; recommended for at most 100.
- `EMIR_FLOW_RADAR_ONLY`: OHLCV flow and structure discovery only.

## Database v5

New table:

```text
cak_autonomous_evidence
```

New installation: run migrations v1 through v5. Existing v1.3.1 installation: run only:

```text
database/migration_v5.sql
database/verify_v5.sql
```

If REST returns HTTP 403/SQLSTATE 42501, run:

```text
database/permissions_hotfix_v1_4_0.sql
database/verify_permissions_v1_4_0.sql
```

Target preflight:

```text
HEALTHY_EMIR_DATABASE_V5
7/7 tables readable
```

Streamlit Secrets:

```toml
CAK_DATABASE_ENABLED = "true"
CAK_DATABASE_SCHEMA = "public"
SUPABASE_URL = "https://PROJECT-REF.supabase.co"
SUPABASE_SECRET_KEY = "sb_secret_..."
```

Never commit secrets to GitHub.

## Validation

- 55 tests passed, 0 failed.
- 400 tickers × 760 bars: 400/400 feature states valid.
- 8 autonomous positive controls reached `EMIR_AUTO_EOD_READY`.
- 0 production-gate bypass.
- 0 invalid entry–SL–TP hierarchy.
- 2,000 randomized profiles: 0 crashes, 0 gate bypass, 0 invalid hierarchy.
- Provider success fixtures passed for Yahoo chart, KSEI profile/corporate action/price history, Google News RSS, public financial statements, and both proxies.
- Database tests cover secret-key headers, all-table write/readback, and visible partial-write failure.

The build container cannot resolve external DNS, so live Yahoo/KSEI/Google HTTP acceptance is not claimed. Live failure was fail-closed and recorded in `validation_artifacts/LIVE_PROVIDER_SMOKE_V1_4_0.json`. Run a 10-ticker deployment smoke scan before a 400-ticker production scan.
