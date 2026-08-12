from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import persistent_cache as pc
import resumable_scan as rs
from future_fundamental import calculate_future_fundamental
from narrative_flow_engine import calculate_market_features
from persistence import DatabaseConfig


def synthetic_frame(seed: int, bars: int = 800) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-07-10", periods=bars)
    ret = rng.normal(0.0005, 0.018, bars)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.004, bars))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.015, bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.015, bars))
    volume = rng.integers(500_000, 8_000_000, bars)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


def main() -> None:
    tickers = [f"T{i:03d}.JK" for i in range(400)]
    benchmark = synthetic_frame(999)
    start = time.perf_counter()
    feature_records = []
    full_bytes = 0
    feature_bytes = 0
    for i, ticker in enumerate(tickers):
        frame = synthetic_frame(i + 1)
        features = calculate_market_features(frame, benchmark, as_of=frame.index[-1])
        assert features.get("feature_state") == "OK"
        feature_records.append({"ticker": ticker, **features})
        full_bytes += len(json.dumps(pc.frame_to_payload(frame), separators=(",", ":")))
        row = pc.build_market_feature_cache_row(ticker, features, checked_at="2026-08-12T00:00:00Z")
        feature_bytes += len(json.dumps(row["payload"], separators=(",", ":"), default=str))
    cold_seconds = time.perf_counter() - start
    cached = pd.DataFrame(feature_records)
    universe = pd.DataFrame({"ticker": tickers})

    original_feature_loader = rs.load_cached_market_features
    original_ohlcv_loader = rs.load_cached_ohlcv_frames
    rs.load_cached_market_features = lambda *a, **k: (cached, pd.DataFrame())
    rs.load_cached_ohlcv_frames = lambda *a, **k: (_ for _ in ()).throw(AssertionError("warm path loaded full OHLCV"))
    try:
        warm_start = time.perf_counter()
        warm, frames, audit, rows = rs._load_fast_features_cache_first(
            DatabaseConfig(True, "https://fixture.supabase.co", "secret"), universe, benchmark,
            period="5y", now="2026-08-12", completed_only=True,
        )
        warm_seconds = time.perf_counter() - warm_start
    finally:
        rs.load_cached_market_features = original_feature_loader
        rs.load_cached_ohlcv_frames = original_ohlcv_loader
    assert len(warm) == 400 and not frames and not rows

    events = pd.DataFrame([{
        "ticker":"TEST.JK", "published_at":"2026-08-01", "title":"Issuer wins contract and expands plant capacity",
        "summary":"capex project enters commercial operation", "source_tier":"OFFICIAL", "source_verified":True,
        "materiality_score":90, "financial_bridge_score":90, "category":"PROJECT_CAPACITY", "url":"https://issuer.example/project",
    }])
    ff = calculate_future_fundamental(
        ticker="TEST.JK", events=events,
        narrative={"top_down_catalyst_score":70,"industry_translation_score":75,"narrative_coverage_pct":85,"issuer_alignment_score":80,"issuer_alignment_coverage_pct":80,"narrative_state":"THESIS_FORMING"},
        fundamental={"fundamental_coverage_pct":90,"fundamental_conversion_score":80,"fundamental_cashflow_quality_state":"CASHFLOW_POSITIVE_CONVERTING","fundamental_leverage_risk_state":"BALANCE_SHEET_CAPACITY_OK","current_ratio":2,"cash_to_debt_ratio":1.5,"fundamental_growth_consistency_state":"QUARTER_AND_YTD_CONFIRMED","fundamental_period_freshness_state":"CURRENT_QUARTERLY_PERIOD"},
        ownership={"ownership_score":70,"ownership_coverage_pct":70}, sector={"sector_leadership_score":70,"sector_context_coverage_pct":100}, as_of="2026-08-12",
    )
    assert ff["future_fundamental_state"] == "FUTURE_FUNDAMENTAL_HIGH_VISIBILITY"
    report = {
        "state": "PASS",
        "scanner_version": rs.PIPELINE_VERSION,
        "ticker_count": 400,
        "bars_per_ticker": 800,
        "cold_feature_compute_seconds": round(cold_seconds, 3),
        "warm_feature_cache_seconds": round(warm_seconds, 5),
        "warm_full_ohlcv_loads": 0,
        "full_ohlcv_payload_mb": round(full_bytes / 1024 / 1024, 2),
        "market_feature_payload_mb": round(feature_bytes / 1024 / 1024, 2),
        "feature_payload_vs_ohlcv_pct": round(100 * feature_bytes / max(full_bytes, 1), 2),
        "future_fundamental_state": ff["future_fundamental_state"],
        "future_fundamental_score": ff["future_fundamental_score"],
        "future_fundamental_coverage_pct": ff["future_fundamental_coverage_pct"],
    }
    Path("VALIDATION_V1_9_14.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
