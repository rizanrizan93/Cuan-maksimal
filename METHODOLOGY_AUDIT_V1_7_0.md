# Methodology Audit — v1.7.0

## Objective

The production sequence is now explicitly:

**macro/regime → sector opportunity map → business quality & future-fundamental conversion → narrative → flow/inventory/ownership → market structure → execution**.

The scanner remains clean-room decision support. It uses public ideas associated with narrative play, smart-money behaviour, sector rotation and disciplined risk management, combined with independently designed empirical proxies. It does not claim to recreate an undisclosed proprietary CAK algorithm.

## Audit conclusions from v1.6.4

### Growth

The prior implementation compared the latest quarter with the immediately previous quarter for generic growth fields. That is inappropriate for seasonal businesses and can invert the interpretation of a company whose same-quarter YoY performance is improving. v1.7 stores both bases and prefers YoY for the legacy fields.

### Solvency

`Total Debt / Equity` and `Total Liabilities / Equity` answer different questions. They are no longer treated as interchangeable. Net debt and liquidity coverage are also separated so a company with large cash is not assessed the same way as a similarly levered company with weak liquidity.

### Future-fundamental conversion

Narrative and price flow cannot substitute for business evidence. Fundamental conversion now has a direct 16% weight and minimum coverage/quality thresholds for thesis readiness. Revenue, earnings, margin, OCF/FCF and solvency are explicit proof requirements.

### Inventory / smart money

A 20-day OHLCV pattern is useful for local absorption but insufficient to infer whether the market is collecting from a multi-year bottom, distributing old inventory, or re-accumulating after markup. The fallback therefore spans up to 756 bars when history exists. Direct broker/ownership evidence remains superior.

### Macro and sector

Weak broad-market conditions should reduce risk, but a blanket blocker can discard a genuine relative-strength leader. v1.7 permits only a narrowly defined leading-sector exception and caps position size.

### Execution

An accumulation entry and breakout entry have different cost bases and invalidation points. They now receive separate plans; the scanner no longer advertises a midpoint RR as the RR of the whole entry zone.

## Conviction score

The v1.7 fixed-denominator empirical score is:

- flow/inventory 18%
- fundamental conversion 16%
- narrative runway 14%
- structure 13%
- macro/sector 10%
- financial/narrative conversion blend 8%
- issuer alignment/ownership 7%
- IDX integrity 5%
- order-book/EOD microstructure 4%
- trend 3%
- liquidity/float 2%

Distribution, crowding, cannibalisation, execution friction and negative narrative remain penalties. Evidence coverage is tracked separately; a high numerical score cannot by itself promote sparse evidence to production.

## Real-money doctrine

A high score is a research priority, not an order. Production readiness requires business evidence, narrative evidence, flow, structure, liquidity, integrity and execution conditions to converge. `WAIT_REACCUMULATION`, `WAIT_FUNDAMENTAL_CONVERSION`, and `FUNDAMENTAL_EVIDENCE_PENDING` are expected outcomes and are preferable to false precision.
