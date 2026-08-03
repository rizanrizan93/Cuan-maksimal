from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from narrative_flow_engine import (
    build_emir_profile,
    calculate_market_context,
    calculate_market_features,
    calculate_sector_context,
    score_narrative_events,
)


def synthetic_frame(n: int, seed: int, trend: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-02", periods=n)
    returns = rng.normal(trend, 0.014, n)
    close = 600 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.002, 0.018, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.002, 0.018, n))
    volume = rng.integers(400_000, 8_000_000, n).astype(float)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=index)


def main() -> None:
    start = time.perf_counter()
    n_tickers = 400
    n_bars = 760
    benchmark = synthetic_frame(n_bars, 999, 0.00035)
    as_of = benchmark.index[-1]
    market = calculate_market_context(benchmark)
    tickers = [f"T{i:03d}.JK" for i in range(n_tickers)]
    sectors = [f"SECTOR_{i % 10}" for i in range(n_tickers)]
    universe = pd.DataFrame({"ticker": tickers, "sector": sectors})

    frames: dict[str, pd.DataFrame] = {}
    features_rows = []
    for i, ticker in enumerate(tickers):
        trend = -0.0004 + (i % 20) * 0.00008
        frame = synthetic_frame(n_bars, i + 1, trend)
        frames[ticker] = frame
        features_rows.append({"ticker": ticker, **calculate_market_features(frame, benchmark, as_of=as_of)})
    fast = pd.DataFrame(features_rows)
    sector_map = calculate_sector_context(fast, universe)

    profiles = []
    for i, row in fast.iterrows():
        deep = i < 40
        narrative = score_narrative_events(pd.DataFrame())
        broker = {}
        ownership = {}
        orderbook = {}
        integrity = {}
        if deep:
            events = pd.DataFrame([
                {
                    "ticker": row["ticker"],
                    "published_at": pd.Timestamp(as_of, tz="Asia/Jakarta").tz_convert("UTC") - pd.Timedelta(days=3),
                    "title": "Issuer expansion project reaches commercial operation",
                    "summary": "Capacity and order book support revenue margin earnings and cash flow",
                    "publisher": "Issuer Investor Relations",
                    "url": "https://issuer.example/press-release",
                    "source_tier": "OFFICIAL", "source_verified": True,
                    "top_down_catalyst_score": 80,
                    "industry_translation_score": 78,
                },
                {
                    "ticker": row["ticker"],
                    "published_at": pd.Timestamp(as_of, tz="Asia/Jakarta").tz_convert("UTC") - pd.Timedelta(days=7),
                    "title": "Industry demand supports project ramp-up",
                    "summary": "Revenue and margin conversion expected from utilization",
                    "publisher": "Independent News",
                    "url": "https://news.example/article",
                    "source_tier": "PUBLIC_NEWS",
                },
            ])
            narrative = score_narrative_events(
                events,
                as_of=pd.Timestamp(as_of, tz="Asia/Jakarta"),
                issuer_context={"company_name": row["ticker"], "sector": sectors[i], "theme": "expansion project", "catalyst": "commercial operation"},
            )
            broker = {
                "broker_inventory_score": 78.0,
                "broker_inventory_coverage_pct": 90.0,
                "broker_inventory_shift_state": "COLLECTION_PERSISTING",
                "retail_cannibalisation_risk": 10.0,
            }
            ownership = {"ownership_score": 75.0, "ownership_coverage_pct": 90.0, "effective_free_float_pct": 20.0}
            integrity = {
                "idx_integrity_score": 92.0,
                "idx_integrity_coverage_pct": 100.0,
                "idx_integrity_state": "IDX_INTEGRITY_CLEAR",
                "idx_integrity_hard_block": False,
                "idx_integrity_provenance_state": "DIRECT_SOURCE_VERIFIED",
                "corporate_action_review_cleared": True,
            }
            if i < 20:
                orderbook = {
                    "orderbook_trigger_score": 76.0,
                    "orderbook_coverage_pct": 100.0,
                    "orderbook_provenance_state": "DIRECT_SOURCE_VERIFIED",
                    "precise_trigger_price": float(row["high20"]),
                }
        profile_features = row.to_dict()
        profile_sector = sector_map.get(row["ticker"], {})
        if i < 10:
            profile_features.update({
                "smart_money_score": 84.0, "smart_money_coverage_pct": 100.0,
                "trend_score": 86.0, "liquidity_score": 82.0,
                "distribution_score": 8.0, "crowding_score": 35.0,
                "price_stage": "MARKUP", "absorption_score": 80.0,
                "market_structure_score": 82.0, "market_structure_mode": "CONTINUATION_SETUP",
                "continuation_price_flow_score": 82.0,
                "ohlcv_integrity_state": "VALID", "ohlcv_integrity_score": 95.0,
                "corporate_action_anomaly_flag": False, "execution_friction_score": 18.0,
            })
            narrative.update({
                "narrative_score": 82.0, "narrative_coverage_pct": 90.0,
                "narrative_state": "MATERIAL_THESIS_CONFIRMED",
                "financial_conversion_score": 80.0, "issuer_alignment_score": 82.0,
                "issuer_alignment_coverage_pct": 90.0, "story_runway_score": 84.0,
                "top_down_catalyst_score": 82.0, "industry_translation_score": 80.0,
                "retail_adoption_stage": "PRE_RETAIL",
                "narrative_verified_source_count": 1, "narrative_official_source_count": 1,
            })
            profile_sector = {
                "sector_leadership_score": 80.0, "sector_context_coverage_pct": 100.0,
                "sector_state": "LEADING", "sector_rrg_state": "LEADING",
            }
        profile = build_emir_profile(
            ticker=row["ticker"],
            features=profile_features,
            narrative=narrative,
            broker=broker,
            ownership=ownership,
            orderbook=orderbook,
            market=market,
            sector=profile_sector,
            integrity=integrity,
            deep_reviewed=deep,
            capital_idr=5_000_000,
            risk_budget_pct=1.0,
            calibration_mode="SHADOW_ONLY",
        )
        profiles.append(profile)

    radar = pd.DataFrame(profiles)
    hierarchy = radar.dropna(subset=["entry_low", "entry_high", "stop_loss", "tp1", "tp2"])
    invalid_hierarchy = hierarchy[
        ~(
            (hierarchy["stop_loss"] < hierarchy["entry_low"])
            & (hierarchy["entry_low"] <= hierarchy["entry_high"])
            & (hierarchy["entry_high"] < hierarchy["tp1"])
            & (hierarchy["tp1"] < hierarchy["tp2"])
        )
    ]
    production = radar[radar["production_ready"].fillna(False)]
    invalid_production = production[
        production["idx_integrity_ready"].ne(True)
        | production["orderbook_provenance_state"].ne("DIRECT_SOURCE_VERIFIED")
        | production["execution_capacity_state"].eq("EXECUTION_CAPACITY_BLOCK")
    ]
    feature_keys = [
        "smart_money_score", "market_structure_score", "ohlcv_integrity_score",
        "execution_friction_score", "liquidity_score", "trend_score",
    ]
    finite_feature_rows = int(np.isfinite(fast[feature_keys].to_numpy(dtype=float)).all(axis=1).sum())
    elapsed = time.perf_counter() - start
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    result = {
        "scanner_version": "1.3.0-idx-integrity-capacity-calibration",
        "ticker_count": int(len(radar)),
        "bars_per_ticker": n_bars,
        "feature_state_ok": int(fast["feature_state"].eq("OK").sum()),
        "finite_feature_rows": finite_feature_rows,
        "deep_reviewed": int(radar["deep_review_state"].eq("DEEP_REVIEWED").sum()),
        "radar_only": int(radar["deep_review_state"].eq("RADAR_ONLY").sum()),
        "production_ready": int(radar["production_ready"].fillna(False).sum()),
        "invalid_production_without_direct_gates": int(len(invalid_production)),
        "valid_execution_plans": int(len(hierarchy)),
        "invalid_execution_hierarchy": int(len(invalid_hierarchy)),
        "decision_state_counts": {str(k): int(v) for k, v in radar["emir_decision_state"].value_counts().to_dict().items()},
        "elapsed_seconds": round(elapsed, 3),
        "peak_rss_mb": round(rss_mb, 2),
    }
    print(json.dumps(result, indent=2))
    Path("VALIDATION_400_SYNTHETIC_V1_3_0.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["ticker_count"] != 400 or result["feature_state_ok"] != 400:
        raise SystemExit("400-ticker feature acceptance failed")
    if result["invalid_production_without_direct_gates"] != 0:
        raise SystemExit("production gate acceptance failed")
    if result["invalid_execution_hierarchy"] != 0:
        raise SystemExit("execution hierarchy acceptance failed")


if __name__ == "__main__":
    main()
