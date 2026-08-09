from __future__ import annotations

from typing import Any, Mapping
import math

import numpy as np
import pandas as pd

from autonomous_enrichment import (
    apply_regulatory_event_overlay,
    autonomous_evidence_frame,
    build_broker_inventory_proxy,
    build_orderbook_proxy,
    ksei_actions_to_events,
    ksei_profiles_to_maps,
    reconcile_fundamental_snapshot,
)
from data_providers import assess_benchmark_freshness, normalize_ticker
from narrative_flow_engine import (
    aggregate_broker_summary,
    build_emir_profile,
    build_outcome_calibration,
    calculate_market_context,
    calculate_market_context_from_universe,
    blend_market_context,
    calculate_market_features,
    calculate_sector_context,
    parse_idx_integrity,
    parse_orderbook_evidence,
    parse_ownership,
    score_narrative_events,
)
from persistence import (
    DatabaseConfig,
    _request,
    persist_verify_scan_best_effort,
)
from persistent_cache import (
    cache_commit_succeeded,
    fetch_fundamental_cache_first,
    fetch_ksei_cache_first,
    fetch_idx_official_fundamental_cache_first,
    fetch_news_cache_first,
    fetch_ohlcv_cache_first,
    load_cached_fundamentals,
    load_cached_ksei,
    load_cached_idx_official_fundamentals,
    load_cached_news,
    load_cached_ohlcv_frames,
    persist_verify_cache_bundle,
)
from research_memory import build_research_memory_rows, persist_verify_research_memory

from top3_dashboard import enrich_dashboard_scores, select_top3, select_next_leaders
from scan_jobs import (
    get_scan_job,
    load_job_chunks,
    next_chunk,
    record_job_chunk,
    stage_progress,
    update_scan_job,
)

PIPELINE_VERSION = "1.9.3-fundamental-join-integrity-fix"


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value if isinstance(value, (str, int, bool, dict, list)) or value is None else str(value)


def _records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    return [{key: _json_safe(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]


def _frame(records: Any) -> pd.DataFrame:
    return pd.DataFrame(records if isinstance(records, list) else [])


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "verified"}


def normalize_manual_events(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker", "published_at", "title", "summary", "publisher", "url", "source_tier",
        "materiality_score", "financial_bridge_score", "top_down_catalyst_score",
        "industry_translation_score", "issuer_alignment_score", "category", "collection_provider",
        "source_verified",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    local = frame.copy()
    local.columns = [str(column).strip().lower() for column in local.columns]
    if "ticker" not in local.columns:
        return pd.DataFrame(columns=columns)
    local["ticker"] = local["ticker"].map(normalize_ticker)
    if "published_at" not in local.columns:
        local["published_at"] = local.get("event_date")
    local["published_at"] = pd.to_datetime(local["published_at"], errors="coerce", utc=True)
    if "source_url" in local.columns and "url" not in local.columns:
        local["url"] = local["source_url"]
    for column in ("title", "summary", "publisher", "url", "source_tier", "category"):
        if column not in local.columns:
            local[column] = ""
    for column in (
        "materiality_score", "financial_bridge_score", "top_down_catalyst_score",
        "industry_translation_score", "issuer_alignment_score",
    ):
        if column not in local.columns:
            local[column] = np.nan
    local["collection_provider"] = "MANUAL_EVIDENCE_UPLOAD"
    if "source_verified" not in local.columns:
        local["source_verified"] = False
    local["source_verified"] = local["source_verified"].map(_truthy)
    return local[columns]


def direct_evidence_frame(frame: pd.DataFrame, evidence_type: str, date_candidates: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "evidence_type", "observed_at", "source_verified"])
    local = frame.copy()
    local.columns = [str(column).strip().lower() for column in local.columns]
    if "ticker" not in local.columns:
        return pd.DataFrame(columns=["ticker", "evidence_type", "observed_at", "source_verified"])
    local["ticker"] = local["ticker"].map(normalize_ticker)
    observed = pd.Series(pd.NaT, index=local.index, dtype="datetime64[ns]")
    for candidate in date_candidates:
        if candidate in local.columns:
            parsed = pd.to_datetime(local[candidate], errors="coerce", utc=True)
            observed = observed.where(observed.notna(), parsed.dt.tz_convert(None))
    local["observed_at"] = observed
    local["evidence_type"] = evidence_type
    if "source_verified" not in local.columns:
        local["source_verified"] = False
    local["source_verified"] = local["source_verified"].map(_truthy)
    return local


def combine_direct_evidence(
    broker: pd.DataFrame, ownership: pd.DataFrame, orderbook: pd.DataFrame, idx_integrity: pd.DataFrame
) -> pd.DataFrame:
    frames = [
        direct_evidence_frame(broker, "BROKER_INVENTORY", ("date", "observed_at")),
        direct_evidence_frame(ownership, "OWNERSHIP_FREE_FLOAT", ("observed_at", "date")),
        direct_evidence_frame(orderbook, "ORDERBOOK_BID_OFFER", ("observed_at", "date")),
        direct_evidence_frame(idx_integrity, "IDX_INTEGRITY_REGULATORY", ("observed_at", "date")),
    ]
    non_empty = [frame for frame in frames if not frame.empty]
    return pd.concat(non_empty, ignore_index=True, sort=False) if non_empty else pd.DataFrame()


def position_builder(row: pd.Series, capital: float, risk_pct: float) -> dict[str, Any]:
    numbers = {
        key: pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
        for key in ("entry_low", "entry_high", "stop_loss", "position_cap_pct")
    }
    if not all(np.isfinite(value) for value in numbers.values()) or numbers["position_cap_pct"] <= 0:
        return {"lot": 0, "position_value": 0.0, "risk_idr": 0.0, "position_state": "NO_EXECUTABLE_POSITION"}
    entry = (numbers["entry_low"] + numbers["entry_high"]) / 2
    per_share_risk = entry - numbers["stop_loss"]
    if per_share_risk <= 0:
        return {"lot": 0, "position_value": 0.0, "risk_idr": 0.0, "position_state": "INVALID_RISK"}
    risk_budget = capital * risk_pct / 100
    max_value = capital * numbers["position_cap_pct"] / 100
    shares_by_risk = np.floor(risk_budget / per_share_risk / 100) * 100
    shares_by_cap = np.floor(max_value / entry / 100) * 100
    shares = max(0, min(shares_by_risk, shares_by_cap))
    return {
        "lot": int(shares / 100),
        "position_value": round(shares * entry, 2),
        "risk_idr": round(shares * per_share_risk, 2),
        "position_state": "POSITION_READY" if shares >= 100 else "CAPITAL_OR_RISK_TOO_SMALL",
    }


def radar_sort(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    local = frame.copy()
    state_order = {
        "EMIR_READY_WITH_PRECISE_TRIGGER": 0,
        "EMIR_AUTO_EOD_READY": 1,
        "EMIR_CORE_THESIS_READY_WAIT_IDX_INTEGRITY": 2,
        "EMIR_THESIS_READY_WAIT_BID_OFFER": 2,
        "EMIR_WATCH_INVENTORY_COLLECTION": 3,
        "EMIR_WAIT_NARRATIVE": 4,
        "EMIR_WAIT_MONEY_FLOW": 5,
        "EMIR_EVIDENCE_PENDING": 6,
        "EMIR_NO_EDGE_YET": 7,
        "EMIR_RADAR_ONLY_NOT_DEEP_REVIEWED": 8,
        "EMIR_AVOID_RETAIL_EUPHORIA": 9,
        "EMIR_REJECT_SMART_MONEY_DISTRIBUTION": 10,
        "EMIR_DATA_INTEGRITY_BLOCK": 11,
        "EMIR_REJECT_IDX_INTEGRITY": 12,
        "EMIR_CALIBRATION_REJECTED": 13,
    }
    local["_state_order"] = local["emir_decision_state"].map(state_order).fillna(99)
    columns = [column for column in ("_state_order", "emir_conviction_score", "broker_inventory_score", "smart_money_score", "liquidity_score") if column in local.columns]
    ascending = [True] + [False] * (len(columns) - 1)
    return local.sort_values(columns, ascending=ascending, na_position="last").drop(columns="_state_order").reset_index(drop=True)


def compute_fast_context(
    universe: pd.DataFrame,
    frames: Mapping[str, pd.DataFrame],
    benchmark: pd.DataFrame,
    *,
    as_of: Any,
) -> dict[str, Any]:
    tickers = universe["ticker"].tolist()
    # Small universes must not require every ticker to have valid OHLCV before benchmark
    # freshness can be evaluated. Require 70% coverage (minimum 5 when possible), capped
    # at 20 for large scans. This remains fail-closed when too little reference data exists.
    ticker_count = len(tickers)
    required_reference_count = max(
        1,
        min(20, ticker_count, max(5, int(math.ceil(0.70 * ticker_count)))) if ticker_count else 1,
    )
    benchmark_freshness = assess_benchmark_freshness(
        benchmark,
        frames,
        min_universe_count=required_reference_count,
    )
    benchmark_for_features = benchmark if benchmark_freshness.get("benchmark_usable") else pd.DataFrame()
    market_context = calculate_market_context(benchmark_for_features)
    fast = pd.DataFrame([
        {"ticker": ticker, **calculate_market_features(frames.get(ticker, pd.DataFrame()), benchmark_for_features, as_of=as_of)}
        for ticker in tickers
    ])
    universe_market_context = calculate_market_context_from_universe(fast)
    if str(market_context.get("market_regime")) == "MARKET_CONTEXT_UNAVAILABLE":
        market_context = universe_market_context
    else:
        market_context = blend_market_context(market_context, universe_market_context)
    sector_map = calculate_sector_context(fast, universe)
    return {
        "fast": fast,
        "market_context": market_context,
        "sector_map": sector_map,
        "benchmark_freshness": benchmark_freshness,
        "benchmark_for_features": benchmark_for_features,
    }


DEEP_REVIEW_SCOPES = {
    "FAST_TOP_30": 30,
    "BALANCED_TOP_60": 60,
    "ALL_ELIGIBLE": None,
    "CUSTOM_LIMIT": -1,
}


def choose_shortlist(
    fast: pd.DataFrame,
    sector_map: Mapping[str, Mapping[str, Any]],
    scan_mode: str,
    deep_limit: int,
    deep_review_scope: str = "ALL_ELIGIBLE",
) -> list[str]:
    """Return the ordered progressive deep-review universe.

    All candidates first pass the cheap OHLCV integrity gate. ``ALL_ELIGIBLE`` keeps
    every valid ticker and lets KSEI/news/fundamental stages process them in resumable
    chunks. Fast/balanced/custom scopes remain available for shorter daily scans.
    """
    if fast.empty or "feature_state" not in fast.columns or scan_mode == "EMIR_FLOW_RADAR_ONLY":
        return []
    eligible = fast[fast["feature_state"].eq("OK")].copy()
    if eligible.empty:
        return []
    eligible["emir_discovery_score"] = (
        0.27 * pd.to_numeric(eligible.get("smart_money_score"), errors="coerce")
        + 0.21 * pd.to_numeric(eligible.get("market_structure_score"), errors="coerce")
        + 0.13 * pd.to_numeric(eligible.get("seller_exhaustion_score"), errors="coerce")
        + 0.12 * pd.to_numeric(eligible.get("absorption_score"), errors="coerce")
        + 0.09 * pd.to_numeric(eligible.get("relative_strength60_pct"), errors="coerce").clip(-20, 20).add(20).mul(2.5)
        + 0.08 * pd.to_numeric(eligible.get("trend_score"), errors="coerce")
        + 0.05 * pd.to_numeric(eligible.get("liquidity_score"), errors="coerce")
        - 0.03 * pd.to_numeric(eligible.get("distribution_score"), errors="coerce")
        - 0.02 * pd.to_numeric(eligible.get("crowding_score"), errors="coerce")
    )
    eligible["sector_overlay"] = eligible["ticker"].map(
        lambda ticker: float(sector_map.get(ticker, {}).get("sector_leadership_score", 50))
    )
    eligible["emir_discovery_score"] += 0.05 * (eligible["sector_overlay"] - 50)
    ordered = eligible.sort_values(
        ["emir_discovery_score", "liquidity_score", "smart_money_score"],
        ascending=[False, False, False],
        na_position="last",
    )["ticker"].tolist()

    scope = str(deep_review_scope or "ALL_ELIGIBLE").upper()
    if scan_mode == "EMIR_AUTONOMOUS_DEEP_REVIEW" or scope == "ALL_ELIGIBLE":
        return ordered
    if scope == "FAST_TOP_30":
        return ordered[: min(30, len(ordered))]
    if scope == "BALANCED_TOP_60":
        return ordered[: min(60, len(ordered))]
    limit = max(1, int(deep_limit or 30))
    return ordered[: min(limit, len(ordered))]


def _next_deep_stage(settings: Mapping[str, Any], after: str, shortlist: list[str]) -> str:
    stages = []
    if bool(settings.get("auto_ksei", True)) and shortlist:
        stages.append("KSEI_SHORTLIST")
    if shortlist and (bool(settings.get("use_google_news", True)) or bool(settings.get("use_yahoo_news", True))):
        stages.append("NEWS_SHORTLIST")
    if bool(settings.get("auto_fundamental", True)) and shortlist:
        stages.append("FUNDAMENTAL_SHORTLIST")
    if bool(settings.get("auto_idx_official_fundamental", False)) and shortlist:
        stages.append("IDX_FUNDAMENTAL_SHORTLIST")
    stages.append("FINALIZE")
    if after == "FAST_RANKING":
        return stages[0]
    try:
        index = stages.index(after)
        return stages[index + 1]
    except (ValueError, IndexError):
        return "FINALIZE"


def _append_failures(job: Mapping[str, Any], stage: str, failed: list[str]) -> dict[str, Any]:
    failures = dict(job.get("failures") or {})
    current = list(failures.get(stage) or [])
    failures[stage] = list(dict.fromkeys([*current, *failed]))
    return failures


def _process_cache_stage(
    config: DatabaseConfig,
    job: Mapping[str, Any],
    *,
    stage: str,
    tickers: list[str],
    now: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = dict(job.get("settings") or {})
    offset = int(job.get("current_offset") or 0)
    chunk_size = int(job.get("chunk_size") or 20)
    chunk, next_offset, done = next_chunk(tickers, offset, chunk_size)
    chunk_no = int(job.get("current_chunk") or 0) + 1
    started_at = pd.Timestamp.now(tz="UTC")
    audit = pd.DataFrame()
    cache_rows_ohlcv: list[dict[str, Any]] = []
    cache_rows_source: list[dict[str, Any]] = []
    success_tickers: set[str] = set()

    if stage == "BENCHMARK":
        frames, audit, cache_rows_ohlcv = fetch_ohlcv_cache_first(
            config, ("^JKSE",), period=str(settings.get("period") or "5y"),
            max_workers=min(int(settings.get("workers") or 3), 3),
            completed_only=bool(settings.get("completed_only", True)), now=now,
            force_refresh=bool(settings.get("force_cache_refresh", False)), last_scan_id=str(job.get("scan_id")),
        )
        success_tickers = {ticker for ticker, frame in frames.items() if not frame.empty}
        chunk = ["^JKSE"]; next_offset = 1; done = True
    elif stage == "OHLCV":
        frames, audit, cache_rows_ohlcv = fetch_ohlcv_cache_first(
            config, chunk, period=str(settings.get("period") or "5y"),
            max_workers=int(settings.get("workers") or 3), completed_only=bool(settings.get("completed_only", True)),
            now=now, force_refresh=bool(settings.get("force_cache_refresh", False)), last_scan_id=str(job.get("scan_id")),
        )
        success_tickers = {ticker for ticker, frame in frames.items() if not frame.empty}
    elif stage == "KSEI_SHORTLIST":
        profiles, actions, audit, cache_rows_source = fetch_ksei_cache_first(
            config, chunk, max_workers=int(settings.get("workers") or 3), now=now,
            force_refresh=bool(settings.get("force_cache_refresh", False)), last_scan_id=str(job.get("scan_id")),
        )
        success_tickers = set(profiles.get("ticker", pd.Series(dtype=str)).astype(str)) | set(actions.get("ticker", pd.Series(dtype=str)).astype(str))
    elif stage == "NEWS_SHORTLIST":
        universe = pd.DataFrame(job.get("universe") or [])
        if universe.empty:
            universe = pd.DataFrame({"ticker": chunk, "company_name": ""})
        for column in ("company_name",):
            if column not in universe.columns:
                universe[column] = ""
        local = universe[universe["ticker"].isin(chunk)][["ticker", "company_name"]]
        events, audit, cache_rows_source = fetch_news_cache_first(
            config, local, limit=int(settings.get("news_per_ticker") or 6),
            max_workers=int(settings.get("workers") or 3),
            use_yahoo=bool(settings.get("use_yahoo_news", True)), use_google=bool(settings.get("use_google_news", True)),
            now=now, force_refresh=bool(settings.get("force_cache_refresh", False)), last_scan_id=str(job.get("scan_id")),
        )
        success_tickers = set(audit.loc[audit.get("status", pd.Series(dtype=str)).astype(str).isin(["CACHE_HIT", "COLD_REFRESH", "REFRESHED", "NO_ITEMS", "OK"]), "ticker"].astype(str)) if not audit.empty and "ticker" in audit.columns else set()
        success_tickers |= set(events.get("ticker", pd.Series(dtype=str)).astype(str))
    elif stage == "FUNDAMENTAL_SHORTLIST":
        snapshots, audit, cache_rows_source = fetch_fundamental_cache_first(
            config, chunk, max_workers=min(int(settings.get("workers") or 3), 3), now=now,
            force_refresh=bool(settings.get("force_cache_refresh", False)), last_scan_id=str(job.get("scan_id")),
        )
        success_tickers = set(snapshots.get("ticker", pd.Series(dtype=str)).astype(str))
    elif stage == "IDX_FUNDAMENTAL_SHORTLIST":
        snapshots, audit, cache_rows_source = fetch_idx_official_fundamental_cache_first(
            config, chunk, max_workers=min(int(settings.get("workers") or 2), 2), now=now,
            force_refresh=bool(settings.get("force_cache_refresh", False)), last_scan_id=str(job.get("scan_id")),
        )
        # NO_ITEMS is a completed official-source check and must advance the checkpoint.
        if not audit.empty and "ticker" in audit.columns:
            success_tickers = set(audit.loc[audit.get("status", pd.Series(dtype=str)).astype(str).isin(["CACHE_HIT","COLD_REFRESH","REFRESHED","NO_ITEMS","OK","PARTIAL","RESEARCH_MEMORY_FALLBACK"]), "ticker"].astype(str))
        success_tickers |= set(snapshots.get("ticker", pd.Series(dtype=str)).astype(str))
    else:
        raise ValueError(f"Unsupported cache stage {stage}")

    write, verify = persist_verify_cache_bundle(
        config,
        scan_id=str(job.get("scan_id")),
        ohlcv_rows=cache_rows_ohlcv,
        source_rows=cache_rows_source,
    )
    rows_expected = len(cache_rows_ohlcv) + len(cache_rows_source)
    cache_ok = rows_expected == 0 or cache_commit_succeeded(verify)
    failed = [ticker for ticker in chunk if ticker not in success_tickers]
    status = "CHUNK_COMMITTED" if cache_ok else "CHUNK_CACHE_COMMIT_FAILED"
    record_job_chunk(
        config,
        scan_id=str(job.get("scan_id")), stage=stage, chunk_no=chunk_no, tickers=chunk,
        processed_count=len(chunk) - len(failed), failed_count=len(failed), status=status,
        payload={
            "tickers": chunk,
            "failed_tickers": failed,
            "audit_records": _records(audit),
            "cache_write_summary": _records(write.head(1)),
            "cache_verify_summary": _records(verify.head(1)),
        },
        started_at=started_at,
    )
    if not cache_ok:
        updated = update_scan_job(config, str(job.get("scan_id")), {
            "status": "PAUSED",
            "last_error": "Cache checkpoint failed; same chunk will be retried.",
            "progress_pct": stage_progress(stage, offset, max(len(tickers), 1)),
        })
        return updated, {"state": status, "audit": audit, "failed": failed}

    failures = _append_failures(job, stage, failed)
    if stage == "BENCHMARK":
        next_stage = "OHLCV"
        next_stage_offset = 0
    elif done:
        next_stage = "FAST_RANKING" if stage == "OHLCV" else _next_deep_stage(settings, stage, list(job.get("shortlist") or []))
        next_stage_offset = 0
    else:
        next_stage = stage
        next_stage_offset = next_offset
    overall_processed = min(int(job.get("total_tickers") or 0), int(job.get("processed_tickers") or 0) + (len(chunk) if stage == "OHLCV" else 0))
    progress_total = 1 if stage == "BENCHMARK" else max(len(tickers), 1)
    progress_offset = progress_total if done else next_offset
    updated = update_scan_job(config, str(job.get("scan_id")), {
        "status": "RUNNING",
        "current_stage": next_stage,
        "current_offset": next_stage_offset,
        "current_chunk": chunk_no,
        "processed_tickers": overall_processed,
        "failed_tickers": len(set(sum((list(items) for items in failures.values()), []))),
        "progress_pct": stage_progress(stage, progress_offset, progress_total),
        "failures": failures,
        "last_error": "",
    })
    return updated, {"state": status, "audit": audit, "failed": failed}


def process_next_job_step(config: DatabaseConfig, job: Mapping[str, Any], *, now: Any = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Process one bounded checkpoint. Reconnect can resume from the stored stage/offset.

    It does not create a background worker. The page must be open while a chunk runs; after a disconnect,
    the next session resumes from the last committed checkpoint.
    """
    now = pd.Timestamp.now(tz="Asia/Jakarta") if now is None else pd.Timestamp(now)
    stage = str(job.get("current_stage") or "BENCHMARK")
    settings = dict(job.get("settings") or {})
    universe = pd.DataFrame(job.get("universe") or [])
    tickers = universe.get("ticker", pd.Series(dtype=str)).astype(str).tolist()
    shortlist = list(job.get("shortlist") or [])

    # Legacy stage-routing contract: stage_tickers = ["^JKSE"] if stage == "BENCHMARK" else tickers if stage == "OHLCV" else shortlist
    if stage in {"BENCHMARK", "OHLCV", "KSEI_SHORTLIST", "NEWS_SHORTLIST", "FUNDAMENTAL_SHORTLIST", "IDX_FUNDAMENTAL_SHORTLIST"}:
        if stage == "BENCHMARK": stage_tickers = ["^JKSE"]
        elif stage == "OHLCV": stage_tickers = tickers
        elif stage == "IDX_FUNDAMENTAL_SHORTLIST": stage_tickers = shortlist[: max(1, int(settings.get("official_fundamental_limit") or 60))]
        else: stage_tickers = shortlist
        updated, report = _process_cache_stage(config, job, stage=stage, tickers=stage_tickers, now=now)
        return updated, report, None

    if stage == "FAST_RANKING":
        frames, load_audit = load_cached_ohlcv_frames(
            config, tickers, period=str(settings.get("period") or "5y"), now=now,
            completed_only=bool(settings.get("completed_only", True)),
        )
        benchmark_frames, benchmark_audit = load_cached_ohlcv_frames(
            config, ("^JKSE",), period=str(settings.get("period") or "5y"), now=now,
            completed_only=bool(settings.get("completed_only", True)),
        )
        context = compute_fast_context(universe, frames, benchmark_frames.get("^JKSE", pd.DataFrame()), as_of=now)
        shortlist = choose_shortlist(
            context["fast"],
            context["sector_map"],
            str(settings.get("scan_mode")),
            int(settings.get("deep_limit") or 30),
            str(settings.get("deep_review_scope") or "ALL_ELIGIBLE"),
        )
        chunk_no = int(job.get("current_chunk") or 0) + 1
        record_job_chunk(
            config, scan_id=str(job.get("scan_id")), stage=stage, chunk_no=chunk_no, tickers=tickers,
            processed_count=int(context["fast"].get("feature_state", pd.Series(dtype=str)).eq("OK").sum()),
            failed_count=int(context["fast"].get("feature_state", pd.Series(dtype=str)).ne("OK").sum()),
            status="SHORTLIST_COMPUTED",
            payload={
                "shortlist": shortlist,
                "shortlist_count": len(shortlist),
                "deep_review_scope": str(settings.get("deep_review_scope") or "ALL_ELIGIBLE"),
                "eligible_count": int(context["fast"].get("feature_state", pd.Series(dtype=str)).eq("OK").sum()),
                "audit_records": [*_records(load_audit), *_records(benchmark_audit)],
            },
        )
        next_stage = _next_deep_stage(settings, "FAST_RANKING", shortlist)
        updated = update_scan_job(config, str(job.get("scan_id")), {
            "status": "RUNNING",
            "current_stage": next_stage,
            "current_offset": 0,
            "current_chunk": chunk_no,
            "shortlist": shortlist,
            "progress_pct": stage_progress("FAST_RANKING", 1, 1),
            "last_error": "",
        })
        return updated, {"state": "SHORTLIST_COMPUTED", "shortlist": shortlist}, None

    if stage == "FINALIZE":
        result, updated = finalize_job(config, job, now=now)
        return updated, {"state": str(updated.get("result_status") or "FINALIZED")}, result

    if stage == "COMPLETED":
        return dict(job), {"state": "ALREADY_COMPLETED"}, load_persisted_scan_result(config, str(job.get("scan_id")))

    updated = update_scan_job(config, str(job.get("scan_id")), {
        "status": "FAILED", "result_status": "FAILED_UNKNOWN_STAGE", "last_error": f"Unknown stage: {stage}",
    })
    return updated, {"state": "FAILED_UNKNOWN_STAGE"}, None


def _chunk_audits(config: DatabaseConfig, scan_id: str) -> pd.DataFrame:
    chunks = load_job_chunks(config, scan_id)
    records: list[dict[str, Any]] = []
    if chunks.empty:
        return pd.DataFrame()
    for payload in chunks.get("payload", pd.Series(dtype=object)):
        if isinstance(payload, str):
            try:
                import json
                payload = json.loads(payload)
            except Exception:
                payload = {}
        if isinstance(payload, dict):
            records.extend(payload.get("audit_records") or [])
    return pd.DataFrame(records)


def finalize_job(config: DatabaseConfig, job: Mapping[str, Any], *, now: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    scan_id = str(job.get("scan_id"))
    settings = dict(job.get("settings") or {})
    universe = pd.DataFrame(job.get("universe") or [])
    for column in ("ticker", "company_name", "sector"):
        if column not in universe.columns:
            universe[column] = ""
    tickers = universe["ticker"].astype(str).tolist()
    shortlist = list(job.get("shortlist") or [])
    period = str(settings.get("period") or "5y")
    completed_only = bool(settings.get("completed_only", True))

    frames, ohlcv_load_audit = load_cached_ohlcv_frames(config, tickers, period=period, now=now, completed_only=completed_only)
    benchmark_frames, benchmark_load_audit = load_cached_ohlcv_frames(config, ("^JKSE",), period=period, now=now, completed_only=completed_only)
    benchmark = benchmark_frames.get("^JKSE", pd.DataFrame())
    ksei_profiles, ksei_actions, ksei_load_audit = load_cached_ksei(config, shortlist)
    online_events, news_load_audit = load_cached_news(config, shortlist)
    fundamental_proxy_frame, fundamental_load_audit = load_cached_fundamentals(config, shortlist)
    official_limit = max(1, int(settings.get("official_fundamental_limit") or 60))
    official_tickers = shortlist[:official_limit]
    official_fundamental_frame, official_fundamental_load_audit = load_cached_idx_official_fundamentals(config, official_tickers)
    # Preserve ticker inside each payload. pandas set_index(...).to_dict(orient="index")
    # removes the index column from record values; reconcile_fundamental_snapshot needs
    # the ticker to key the reconciled result back into the final radar. Losing it here
    # silently orphaned every fundamental snapshot in v1.9.x.
    def _fundamental_record_map(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
        if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for _, row in frame.iterrows():
            ticker = normalize_ticker(row.get("ticker"))
            if not ticker:
                continue
            payload = row.to_dict()
            payload["ticker"] = ticker
            out[ticker] = payload
        return out

    proxy_map = _fundamental_record_map(fundamental_proxy_frame)
    official_map = _fundamental_record_map(official_fundamental_frame)
    reconciled=[]
    for ticker in shortlist:
        proxy_payload = dict(proxy_map.get(ticker) or {})
        official_payload = dict(official_map.get(ticker) or {})
        # Defensive fallback: even an incomplete cache payload must retain its join key.
        proxy_payload.setdefault("ticker", ticker)
        if official_payload:
            official_payload.setdefault("ticker", ticker)
        reconciled.append(reconcile_fundamental_snapshot(proxy_payload, official_payload, now=now))
    fundamental_frame = pd.DataFrame(reconciled)
    if not fundamental_frame.empty and "ticker" in fundamental_frame.columns:
        fundamental_frame = fundamental_frame.drop_duplicates("ticker", keep="last").reset_index(drop=True)

    if not ksei_profiles.empty:
        profile_index = ksei_profiles.drop_duplicates("ticker").set_index("ticker")
        for column in ("company_name", "sector"):
            if column in profile_index.columns:
                mapped = universe["ticker"].map(profile_index[column]).fillna("").astype(str).str.strip()
                universe[column] = universe[column].where(universe[column].astype(str).str.strip().ne(""), mapped)

    context = compute_fast_context(universe, frames, benchmark, as_of=now)
    fast = context["fast"]
    market_context = context["market_context"]
    sector_map = context["sector_map"]
    benchmark_freshness = context["benchmark_freshness"]

    manual_events = normalize_manual_events(_frame(settings.get("manual_events")))
    ksei_events = ksei_actions_to_events(ksei_actions, as_of=now)
    official_events=[]
    if isinstance(official_fundamental_frame, pd.DataFrame) and not official_fundamental_frame.empty:
        for _, row in official_fundamental_frame.iterrows():
            if not bool(row.get("idx_official_source_verified")): continue
            ticker=str(row.get("ticker") or ""); period_end=str(row.get("idx_official_period_end") or "")
            official_events.append({
                "ticker": ticker, "published_at": pd.to_datetime(period_end, errors="coerce", utc=True),
                "title": f"IDX Official Financial Statement {ticker.replace('.JK','')} {period_end}",
                "summary": f"Official IDX XBRL filing; revenue_yoy={row.get('idx_official_revenue_growth_yoy_pct')}; earnings_yoy={row.get('idx_official_earnings_growth_yoy_pct')}; cashflow={row.get('idx_official_cashflow_state')}",
                "publisher":"Indonesia Stock Exchange", "url":row.get("idx_official_source_url"),
                "source_tier":"OFFICIAL", "materiality_score":88, "financial_bridge_score":92,
                "top_down_catalyst_score":50, "industry_translation_score":70, "issuer_alignment_score":95,
                "category":"EARNINGS_CONVERSION", "collection_provider":"IDX_OFFICIAL_XBRL", "source_verified":True,
            })
    official_events_frame=pd.DataFrame(official_events)
    event_frames = [frame for frame in (manual_events, online_events, ksei_events, official_events_frame) if isinstance(frame, pd.DataFrame) and not frame.empty]
    all_events = pd.concat(event_frames, ignore_index=True, sort=False) if event_frames else pd.DataFrame()
    if not all_events.empty:
        all_events["ticker"] = all_events["ticker"].map(normalize_ticker)
        dedupe = [column for column in ("ticker", "title", "url") if column in all_events.columns]
        if dedupe:
            all_events = all_events.drop_duplicates(dedupe, keep="first")

    fundamental_map = fundamental_frame.set_index("ticker").to_dict(orient="index") if not fundamental_frame.empty else {}
    broker_proxy_map = {str(row["ticker"]): build_broker_inventory_proxy(row.to_dict()) for _, row in fast.iterrows()}
    orderbook_proxy_map = {str(row["ticker"]): build_orderbook_proxy(row.to_dict()) for _, row in fast.iterrows()}
    ownership_auto_map, integrity_auto_map = ksei_profiles_to_maps(ksei_profiles, ksei_actions, as_of=now)
    integrity_auto_map = apply_regulatory_event_overlay(integrity_auto_map, all_events, as_of=now)

    raw_broker = _frame(settings.get("manual_broker"))
    raw_ownership = _frame(settings.get("manual_ownership"))
    raw_orderbook = _frame(settings.get("manual_orderbook"))
    raw_idx_integrity = _frame(settings.get("manual_idx_integrity"))
    raw_outcomes = _frame(settings.get("manual_outcomes"))
    broker_map = {**broker_proxy_map, **aggregate_broker_summary(raw_broker)}
    ownership_map = {**ownership_auto_map, **parse_ownership(raw_ownership)}
    orderbook_map = {**orderbook_proxy_map, **parse_orderbook_evidence(raw_orderbook)}
    idx_integrity_map = {**integrity_auto_map, **parse_idx_integrity(raw_idx_integrity, as_of=now)}
    outcome_calibration_map = build_outcome_calibration(raw_outcomes)
    direct_evidence = combine_direct_evidence(raw_broker, raw_ownership, raw_orderbook, raw_idx_integrity)
    autonomous_evidence = autonomous_evidence_frame(ksei_profiles, ksei_actions, fundamental_proxy_frame, broker_proxy_map, orderbook_proxy_map, now, official_fundamentals=official_fundamental_frame)
    metadata_map = universe.set_index("ticker").to_dict(orient="index")

    capital = float(settings.get("capital") or 5_000_000)
    risk_pct = float(settings.get("risk_pct") or 1.0)
    output_rows: list[dict[str, Any]] = []
    for _, fast_row in fast.iterrows():
        ticker = str(fast_row["ticker"])
        deep_reviewed = ticker in shortlist
        ticker_events = all_events[all_events["ticker"].eq(ticker)] if deep_reviewed and not all_events.empty else pd.DataFrame()
        narrative = score_narrative_events(ticker_events, as_of=now, issuer_context=metadata_map.get(ticker))
        profile = build_emir_profile(
            ticker=ticker,
            features=fast_row.to_dict(),
            narrative=narrative,
            broker=broker_map.get(ticker),
            ownership=ownership_map.get(ticker),
            orderbook=orderbook_map.get(ticker),
            market=market_context,
            sector=sector_map.get(ticker),
            integrity=idx_integrity_map.get(ticker),
            fundamental=fundamental_map.get(ticker),
            outcome_calibration_map=outcome_calibration_map,
            deep_reviewed=deep_reviewed,
            max_position_cap_pct=float(settings.get("max_position_cap_pct") or 20.0),
            capital_idr=capital,
            risk_budget_pct=risk_pct,
            calibration_mode=str(settings.get("calibration_mode") or "GUARDED"),
            capital_mode=str(settings.get("capital_mode") or "GUARDED_REAL_MONEY"),
        )
        output_rows.append({**metadata_map.get(ticker, {}), **profile, **position_builder(pd.Series(profile), capital, float(profile.get("risk_budget_pct") or risk_pct))})
    radar = radar_sort(pd.DataFrame(output_rows))

    # Fundamental join integrity gate. Provider/cache success is not enough: every
    # qualifying reconciled fundamental row must survive into the final radar under
    # the same ticker key. v1.9.x once orphaned all rows by losing ticker during
    # set_index(...).to_dict(orient="index"), producing an empty Next Leader table
    # despite successful Yahoo refreshes. Fail loudly instead of publishing that state.
    if isinstance(fundamental_frame, pd.DataFrame) and not fundamental_frame.empty and "ticker" in fundamental_frame.columns:
        def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
            source = frame[column] if column in frame.columns else pd.Series(index=frame.index, dtype=float)
            return pd.to_numeric(source, errors="coerce")
        src_score = _numeric_column(fundamental_frame, "fundamental_conversion_score")
        src_cov = _numeric_column(fundamental_frame, "fundamental_coverage_pct")
        src_dq = _numeric_column(fundamental_frame, "fundamental_data_quality_score")
        src_mask = src_score.notna() & src_cov.ge(35) & src_dq.ge(35)
        expected_fund_tickers = set(fundamental_frame.loc[src_mask, "ticker"].astype(str))
        if expected_fund_tickers:
            radar_score = _numeric_column(radar, "fundamental_conversion_score")
            radar_cov = _numeric_column(radar, "fundamental_coverage_pct")
            radar_dq = _numeric_column(radar, "fundamental_data_quality_score")
            radar_mask = radar_score.notna() & radar_cov.ge(35) & radar_dq.ge(35)
            actual_fund_tickers = set(radar.loc[radar_mask, "ticker"].astype(str)) if "ticker" in radar.columns else set()
            missing_fund_tickers = sorted(expected_fund_tickers - actual_fund_tickers)
            if missing_fund_tickers:
                raise RuntimeError(
                    "FUNDAMENTAL_JOIN_INTEGRITY_FAILURE: "
                    f"provider/reconciled qualified={len(expected_fund_tickers)}, final qualified={len(actual_fund_tickers)}, "
                    f"missing={len(missing_fund_tickers)} sample={missing_fund_tickers[:12]}"
                )

    # Persist dashboard factor scores and the derived Top 3 summary with the scan.
    # This makes a reloaded database result visually equivalent to the original session.
    radar = enrich_dashboard_scores(radar, frames)
    top3 = select_top3(radar, limit=3)
    next_leaders = select_next_leaders(radar, limit=20)
    top3_summary = [
        {
            "rank": int(index + 1),
            "ticker": str(row.get("ticker") or ""),
            "final_score": float(row.get("emir_final_score") or row.get("emir_conviction_score") or 0.0),
            "recommendation": str(row.get("dashboard_recommendation") or row.get("action") or ""),
            "decision_state": str(row.get("emir_decision_state") or ""),
        }
        for index, row in top3.reset_index(drop=True).iterrows()
    ]

    chunk_audit = _chunk_audits(config, scan_id)
    audit_frames = [frame for frame in (
        chunk_audit, ohlcv_load_audit, benchmark_load_audit, ksei_load_audit, news_load_audit, fundamental_load_audit, official_fundamental_load_audit,
    ) if isinstance(frame, pd.DataFrame) and not frame.empty]
    provider_audit = pd.concat(audit_frames, ignore_index=True, sort=False) if audit_frames else pd.DataFrame()
    provider_audit = pd.concat([provider_audit, pd.DataFrame([{
        "ticker": "^JKSE",
        "provider": "BENCHMARK_FRESHNESS_GATE",
        "status": benchmark_freshness.get("benchmark_freshness_state"),
        "bars": len(benchmark),
        "last_date": benchmark_freshness.get("benchmark_last_date"),
        "detail": (
            f"universe_reference={benchmark_freshness.get('universe_reference_date')}; "
            f"business_lag={benchmark_freshness.get('benchmark_business_lag_days')}; "
            f"usable={benchmark_freshness.get('benchmark_usable')}"
        ),
        "audit_family": "BENCHMARK_FRESHNESS",
    }])], ignore_index=True, sort=False)

    research_memory_rows = build_research_memory_rows(scan_id, all_events, autonomous_evidence)
    research_memory_write_report, research_memory_verification = persist_verify_research_memory(
        config, scan_id=scan_id, rows=research_memory_rows
    )

    write_report, verification, commit_report = persist_verify_scan_best_effort(
        config,
        scan_id=scan_id,
        as_of=now,
        radar=radar,
        events=all_events,
        provider_audit=provider_audit,
        direct_evidence=direct_evidence,
        autonomous_evidence=autonomous_evidence,
        outcomes=raw_outcomes,
        mode=str(settings.get("scan_mode") or "EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP"),
    )
    persistence_state = str(commit_report.iloc[0].get("state", "SCAN_COMPLETED_MEMORY_ONLY")) if not commit_report.empty else "SCAN_COMPLETED_MEMORY_ONLY"
    radar_row = write_report.loc[write_report.get("table", pd.Series(dtype=str)).eq("cak_radar_snapshots")] if not write_report.empty else pd.DataFrame()
    radar_persisted = bool(
        not radar_row.empty
        and int(radar_row.iloc[0].get("rows_written", 0) or 0) == len(radar)
        and len(radar) > 0
    )
    terminal_status = "COMPLETED" if persistence_state == "SCAN_COMPLETED_FULL_PERSISTENCE" else "COMPLETED_PARTIAL_PERSISTENCE"
    if not radar_persisted and config.ready:
        terminal_status = "FINALIZE_RETRY_REQUIRED"
    updated = update_scan_job(config, scan_id, {
        "status": terminal_status,
        "current_stage": "COMPLETED" if terminal_status != "FINALIZE_RETRY_REQUIRED" else "FINALIZE",
        "current_offset": 0,
        "progress_pct": 100.0 if terminal_status != "FINALIZE_RETRY_REQUIRED" else 96.0,
        "result_status": persistence_state if radar_persisted or not config.ready else "RADAR_PERSISTENCE_RETRY_REQUIRED",
        "result_summary": {
            "ticker_count": len(radar),
            "deep_reviewed": int(radar.get("deep_review_state", pd.Series(dtype=str)).eq("DEEP_REVIEWED").sum()),
            "deep_review_target": len(shortlist),
            "deep_review_scope": str(settings.get("deep_review_scope") or "ALL_ELIGIBLE"),
            "production_ready": int(radar.get("production_ready", pd.Series(dtype=bool)).fillna(False).sum()),
            "top3": top3_summary,
            "next_leaders": [{"rank": int(i+1), "ticker": str(r.get("ticker") or ""), "score": float(r.get("next_leader_score") or 0.0)} for i, r in next_leaders.reset_index(drop=True).iterrows()],
            "research_memory_rows": len(research_memory_rows),
            "research_memory_verified_exact": bool(not research_memory_verification.empty and str(research_memory_verification.iloc[0].get("state")) in {"RESEARCH_MEMORY_VERIFIED_EXACT","RESEARCH_MEMORY_VERIFIED_EMPTY"}),
            "official_fundamental_verified": int(official_fundamental_frame.get("idx_official_source_verified", pd.Series(dtype=bool)).fillna(False).sum()) if not official_fundamental_frame.empty else 0,
            "radar_persisted": radar_persisted,
        },
        "last_error": "" if terminal_status != "FINALIZE_RETRY_REQUIRED" else "Radar result was computed but not fully persisted; finalisation can be retried from cache.",
    })
    result = {
        "radar": radar,
        "frames": frames,
        "events": all_events,
        "provider_audit": provider_audit,
        "direct_evidence": direct_evidence,
        "autonomous_evidence": autonomous_evidence,
        "fundamentals": fundamental_frame,
        "fundamental_proxy": fundamental_proxy_frame,
        "idx_official_fundamentals": official_fundamental_frame,
        "ksei_profiles": ksei_profiles,
        "ksei_actions": ksei_actions,
        "write_report": write_report,
        "verification": verification,
        "commit_report": commit_report,
        "cache_write_report": pd.DataFrame(),
        "cache_verification": pd.DataFrame(),
        "cache_summary": pd.DataFrame(),
        "scan_id": scan_id,
        "as_of": now,
        "database_ready": config.ready,
        "database_commit_state": persistence_state,
        "cache_persistence_state": "CHUNK_CHECKPOINTED",
        "expected_events": len(all_events),
        "expected_provider_audit": len(provider_audit),
        "expected_direct_evidence": len(direct_evidence),
        "expected_autonomous_evidence": len(autonomous_evidence),
        "expected_outcomes": len(raw_outcomes),
        "outcomes": raw_outcomes,
        "market_context": market_context,
        "benchmark_freshness": benchmark_freshness,
        "shortlist": shortlist,
        "top3_summary": top3_summary,
        "next_leaders": next_leaders,
        "research_memory_write_report": research_memory_write_report,
        "research_memory_verification": research_memory_verification,
        "expected_research_memory": len(research_memory_rows),
        "deep_review_scope": str(settings.get("deep_review_scope") or "ALL_ELIGIBLE"),
        "job_status": terminal_status,
    }
    return result, updated


def _get_paginated(config: DatabaseConfig, table: str, scan_id: str, *, page_size: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not config.ready:
        return rows
    order_column = {
        "cak_radar_snapshots": "ticker",
        "cak_narrative_events": "event_id",
        "cak_provider_audit": "audit_id",
        "cak_direct_evidence": "evidence_id",
        "cak_autonomous_evidence": "evidence_id",
        "cak_outcome_memory": "outcome_id",
    }.get(table, "scan_id")
    for start in range(0, 50_000, page_size):
        response = _request(
            config,
            "GET",
            table,
            params={"select": "*", "scan_id": f"eq.{scan_id}", "order": f"{order_column}.asc"},
            extra_headers={"Range": f"{start}-{start + page_size - 1}"},
            timeout=20,
        )
        payload = response.json()
        if not isinstance(payload, list):
            break
        rows.extend(payload)
        if len(payload) < page_size:
            break
    return rows


def load_persisted_scan_result(config: DatabaseConfig, scan_id: str) -> dict[str, Any] | None:
    job = get_scan_job(config, scan_id)
    radar_rows = _get_paginated(config, "cak_radar_snapshots", scan_id)
    if not radar_rows:
        return None
    radar_records: list[dict[str, Any]] = []
    for row in radar_rows:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        radar_records.append({"ticker": row.get("ticker"), **payload})
    radar = radar_sort(pd.DataFrame(radar_records))
    event_rows = _get_paginated(config, "cak_narrative_events", scan_id)
    provider_rows = _get_paginated(config, "cak_provider_audit", scan_id)
    direct_rows = _get_paginated(config, "cak_direct_evidence", scan_id)
    autonomous_rows = _get_paginated(config, "cak_autonomous_evidence", scan_id)
    outcome_rows = _get_paginated(config, "cak_outcome_memory", scan_id)
    events = pd.DataFrame([row.get("payload") or {} for row in event_rows])
    provider_audit = pd.DataFrame([row.get("payload") or {} for row in provider_rows])
    direct = pd.DataFrame([row.get("payload") or {} for row in direct_rows])
    autonomous = pd.DataFrame([row.get("payload") or {} for row in autonomous_rows])
    outcomes = pd.DataFrame([row.get("payload") or {} for row in outcome_rows])
    tickers = radar.get("ticker", pd.Series(dtype=str)).astype(str).tolist()
    settings = dict((job or {}).get("settings") or {})
    frames, _ = load_cached_ohlcv_frames(
        config, tickers, period=str(settings.get("period") or "5y"), now=pd.Timestamp.now(tz="Asia/Jakarta"),
        completed_only=bool(settings.get("completed_only", True)),
    )
    first = radar.iloc[0].to_dict() if not radar.empty else {}
    loaded_counts = {
        "cak_scan_runs": 1,
        "cak_radar_snapshots": len(radar),
        "cak_narrative_events": len(events),
        "cak_provider_audit": len(provider_audit),
        "cak_direct_evidence": len(direct),
        "cak_autonomous_evidence": len(autonomous),
        "cak_outcome_memory": len(outcomes),
    }
    write_report = pd.DataFrame([
        {
            "table": table,
            "rows_attempted": count,
            "rows_written": count,
            "state": "LOADED_FROM_DATABASE",
            "detail": "Rows loaded from persisted scan_id.",
        }
        for table, count in loaded_counts.items()
    ])
    verification = pd.DataFrame([
        {
            "state": "VERIFIED_EXACT",
            "table": table,
            "scan_id": scan_id,
            "rows_attempted": count,
            "rows_written": count,
            "rows_verified": count,
            "verification_pct": 100.0,
            "detail": "Persisted rows loaded exactly by scan_id pagination.",
        }
        for table, count in loaded_counts.items()
    ])
    commit_report = pd.DataFrame([{
        "state": str((job or {}).get("result_status") or "PERSISTED_RESULT_LOADED"),
        "scan_id": scan_id,
        "publishable": True,
        "detail": "Result reconstructed from Supabase scan tables and persistent OHLCV cache.",
    }])
    return {
        "radar": radar,
        "frames": frames,
        "events": events,
        "provider_audit": provider_audit,
        "direct_evidence": direct,
        "autonomous_evidence": autonomous,
        "fundamentals": pd.DataFrame(),
        "ksei_profiles": pd.DataFrame(),
        "ksei_actions": pd.DataFrame(),
        "write_report": write_report,
        "verification": verification,
        "commit_report": commit_report,
        "cache_write_report": pd.DataFrame(),
        "cache_verification": pd.DataFrame(),
        "cache_summary": pd.DataFrame(),
        "scan_id": scan_id,
        "as_of": (job or {}).get("updated_at") or pd.Timestamp.now(tz="Asia/Jakarta"),
        "database_ready": config.ready,
        "database_commit_state": (job or {}).get("result_status") or "PERSISTED_RESULT_LOADED",
        "cache_persistence_state": "PERSISTED_CACHE_LOADED",
        "expected_events": len(events),
        "expected_provider_audit": len(provider_audit),
        "expected_direct_evidence": len(direct),
        "expected_autonomous_evidence": len(autonomous),
        "expected_outcomes": len(outcomes),
        "outcomes": outcomes,
        "market_context": {
            "market_regime": first.get("market_regime", "UNKNOWN"),
            "market_context_score": first.get("market_context_score"),
        },
        "benchmark_freshness": {},
        "shortlist": list((job or {}).get("shortlist") or []),
        "top3_summary": list(((job or {}).get("result_summary") or {}).get("top3") or []),
        "deep_review_scope": str(((job or {}).get("result_summary") or {}).get("deep_review_scope") or settings.get("deep_review_scope") or "ALL_ELIGIBLE"),
        "job_status": (job or {}).get("status") or "COMPLETED",
    }


__all__ = [
    "PIPELINE_VERSION", "DEEP_REVIEW_SCOPES", "compute_fast_context", "choose_shortlist", "process_next_job_step",
    "finalize_job", "load_persisted_scan_result", "normalize_manual_events",
]
