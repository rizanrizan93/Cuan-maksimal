from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from narrative_flow_engine import build_emir_profile


def maybe(rng, value, p=0.82):
    return value if rng.random() < p else np.nan


def main() -> None:
    Path("validation_artifacts").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1400)
    crashes = 0
    invalid_hierarchy = 0
    gate_bypass = 0
    state_counts: dict[str, int] = {}
    for i in range(2000):
        last = float(rng.uniform(50, 20000))
        atr = float(last * rng.uniform(0.008, 0.08))
        low20 = float(last * rng.uniform(0.72, 0.98))
        high20 = float(last * rng.uniform(1.01, 1.35))
        ema20 = float(last * rng.uniform(0.88, 1.08))
        features = {
            "feature_state": "OK" if rng.random() > 0.04 else "ERROR",
            "last_price": last,
            "last_date": "2026-07-31",
            "atr14": atr,
            "low20": low20,
            "high20": high20,
            "ema20": ema20,
            "smart_money_score": maybe(rng, rng.uniform(0, 100)),
            "smart_money_coverage_pct": rng.uniform(0, 100),
            "market_structure_score": maybe(rng, rng.uniform(0, 100)),
            "market_structure_mode": rng.choice(["REVERSAL_SETUP", "CONTINUATION_SETUP", "SIDEWAYS_ACCUMULATION", "NO_CLEAR_STRUCTURE"]),
            "trend_score": maybe(rng, rng.uniform(0, 100)),
            "liquidity_score": maybe(rng, rng.uniform(0, 100)),
            "distribution_score": rng.uniform(0, 100),
            "crowding_score": rng.uniform(0, 100),
            "price_stage": rng.choice(["MARKUP", "SILENT_ACCUMULATION", "BASE_TRANSITION", "MARKDOWN"]),
            "execution_friction_score": rng.uniform(0, 100),
            "gap_risk_score": rng.uniform(0, 100),
            "adtv20_idr": max(0.0, rng.lognormal(19, 2)),
            "ohlcv_integrity_state": rng.choice(["VALID", "VALID", "VALID", "STALE_DATA_BLOCK", "INSUFFICIENT_HISTORY"]),
            "corporate_action_anomaly_flag": bool(rng.random() < 0.06),
        }
        narrative = {
            "narrative_score": maybe(rng, rng.uniform(0, 100)),
            "narrative_coverage_pct": rng.uniform(0, 100),
            "narrative_state": rng.choice(["MATERIAL_THESIS_CONFIRMED", "NO_ACTIVE_PUBLIC_NARRATIVE", "CONTRADICTED_OR_NEGATIVE"]),
            "financial_conversion_score": maybe(rng, rng.uniform(0, 100)),
            "issuer_alignment_score": maybe(rng, rng.uniform(0, 100)),
            "issuer_alignment_coverage_pct": rng.uniform(0, 100),
            "story_runway_score": maybe(rng, rng.uniform(0, 100)),
            "top_down_catalyst_score": maybe(rng, rng.uniform(0, 100)),
            "industry_translation_score": maybe(rng, rng.uniform(0, 100)),
            "retail_adoption_stage": rng.choice(["PRE_RETAIL", "EARLY_PUBLIC", "EUPHORIA"]),
            "narrative_verified_source_count": int(rng.integers(0, 3)),
            "narrative_independent_story_count": int(rng.integers(0, 5)),
        }
        broker_prov = rng.choice(["OHLCV_BEHAVIOURAL_PROXY_NOT_BROKER_DATA", "DIRECT_SOURCE_VERIFIED", "UNAVAILABLE"])
        broker = {
            "broker_inventory_score": maybe(rng, rng.uniform(0, 100)),
            "broker_inventory_coverage_pct": rng.uniform(0, 100),
            "broker_inventory_shift_state": rng.choice(["COLLECTION_PERSISTING_PROXY", "NO_CLEAR_INVENTORY_PROXY", "DISTRIBUTION_DOMINANT"]),
            "broker_summary_provenance_state": broker_prov,
        }
        orderbook_prov = rng.choice(["OHLCV_EOD_MICROSTRUCTURE_PROXY_NOT_LIVE_DEPTH", "DIRECT_SOURCE_VERIFIED", "UNAVAILABLE"])
        orderbook = {
            "orderbook_trigger_score": maybe(rng, rng.uniform(0, 100)),
            "orderbook_coverage_pct": rng.uniform(0, 100),
            "precise_trigger_price": high20,
            "orderbook_provenance_state": orderbook_prov,
        }
        integrity_prov = rng.choice(["AUTO_PUBLIC_KSEI_PARTIAL_PROXY", "AUTO_PUBLIC_KSEI_AND_REGULATORY_NEWS", "DIRECT_SOURCE_VERIFIED", "PROVIDER_FAILED"])
        integrity = {
            "idx_integrity_score": maybe(rng, rng.uniform(0, 100)),
            "idx_integrity_coverage_pct": rng.uniform(0, 100),
            "idx_integrity_hard_block": bool(rng.random() < 0.08),
            "idx_integrity_provenance_state": integrity_prov,
            "idx_integrity_unknown_critical_count": int(rng.integers(0, 9)) if integrity_prov != "DIRECT_SOURCE_VERIFIED" else 0,
            "corporate_action_review_cleared": bool(rng.random() > 0.2),
        }
        ownership = {
            "ownership_score": maybe(rng, rng.uniform(0, 100)),
            "ownership_coverage_pct": rng.uniform(0, 100),
            "effective_free_float_pct": maybe(rng, rng.uniform(2, 65), 0.4),
        }
        market = {
            "market_regime": rng.choice(["RISK_ON", "SELECTIVE", "RISK_OFF"]),
            "market_context_score": rng.uniform(0, 100),
            "market_context_coverage_pct": 100,
        }
        sector = {"sector_leadership_score": maybe(rng, rng.uniform(0, 100)), "sector_context_coverage_pct": rng.uniform(0, 100)}
        fundamental = {"fundamental_conversion_score": maybe(rng, rng.uniform(0, 100)), "fundamental_coverage_pct": rng.uniform(0, 100)}
        try:
            profile = build_emir_profile(
                ticker=f"F{i:04d}.JK", features=features, narrative=narrative, broker=broker,
                ownership=ownership, orderbook=orderbook, market=market, sector=sector,
                integrity=integrity, fundamental=fundamental, deep_reviewed=bool(rng.random() > 0.15),
                capital_idr=float(rng.choice([1_000_000, 5_000_000, 50_000_000, 500_000_000])),
                risk_budget_pct=float(rng.uniform(0.25, 2.0)), calibration_mode=rng.choice(["SHADOW_ONLY", "GUARDED"]),
            )
        except Exception:
            crashes += 1
            continue
        state = str(profile.get("emir_decision_state"))
        state_counts[state] = state_counts.get(state, 0) + 1
        values = [profile.get(key) for key in ("stop_loss", "entry_low", "entry_high", "tp1", "tp2")]
        if all(value is not None and np.isfinite(float(value)) for value in values):
            stop, entry_low, entry_high, tp1, tp2 = map(float, values)
            if not stop < entry_low <= entry_high < tp1 < tp2:
                invalid_hierarchy += 1
        if profile.get("production_ready"):
            tier = profile.get("production_tier")
            if tier == "DIRECT_PRECISE" and (integrity_prov != "DIRECT_SOURCE_VERIFIED" or orderbook_prov != "DIRECT_SOURCE_VERIFIED"):
                gate_bypass += 1
            if tier == "AUTO_EOD_PROXY" and (integrity_prov not in {"AUTO_PUBLIC_KSEI_PARTIAL_PROXY", "AUTO_PUBLIC_KSEI_AND_REGULATORY_NEWS"} or int(integrity.get("idx_integrity_unknown_critical_count", 99)) != 0 or orderbook_prov != "OHLCV_EOD_MICROSTRUCTURE_PROXY_NOT_LIVE_DEPTH"):
                gate_bypass += 1
    result = {
        "rows": 2000,
        "crashes": crashes,
        "invalid_execution_hierarchy": invalid_hierarchy,
        "production_gate_bypass": gate_bypass,
        "state_counts": state_counts,
    }
    print(json.dumps(result, indent=2))
    Path("validation_artifacts/FUZZ_VALIDATION_V1_6_0.json").write_text(json.dumps(result, indent=2) + "\n")
    if crashes or invalid_hierarchy or gate_bypass:
        raise SystemExit("fuzz validation failed")


if __name__ == "__main__":
    main()
