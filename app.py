from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_providers import assess_benchmark_freshness, fetch_many_news, fetch_many_ohlcv, normalize_ticker, parse_universe_frame
from autonomous_enrichment import (
    apply_regulatory_event_overlay,
    autonomous_evidence_frame,
    build_broker_inventory_proxy,
    build_orderbook_proxy,
    fetch_many_fundamentals,
    fetch_many_ksei_profiles,
    ksei_actions_to_events,
    ksei_profiles_to_maps,
)
from narrative_flow_engine import (
    ENGINE_VERSION,
    aggregate_broker_summary,
    build_emir_profile,
    build_outcome_calibration,
    calculate_market_context,
    calculate_market_context_from_universe,
    calculate_market_features,
    calculate_sector_context,
    formula_registry_frame,
    make_scan_id,
    parse_idx_integrity,
    parse_orderbook_evidence,
    parse_ownership,
    score_narrative_events,
)
from persistence import (
    config_from_mapping, database_commit_succeeded, database_status,
    persist_verify_commit_scan, test_connection, verify_scan,
)
from persistent_cache import (
    cache_commit_succeeded,
    cache_summary,
    fetch_fundamental_cache_first,
    fetch_ksei_cache_first,
    fetch_news_cache_first,
    fetch_ohlcv_cache_first,
    persist_verify_cache_bundle,
)


st.set_page_config(page_title="IDX Emir Autonomous Scanner", page_icon="🧭", layout="wide")


@st.cache_data(ttl=1800, show_spinner=False)
def cached_fetch_ohlcv(tickers: tuple[str, ...], period: str, max_workers: int, completed_only: bool):
    return fetch_many_ohlcv(tickers, period=period, max_workers=max_workers, completed_only=completed_only)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_news(
    universe_records: tuple[tuple[str, str], ...],
    limit: int,
    max_workers: int,
    use_yahoo: bool,
    use_google: bool,
):
    universe = pd.DataFrame(universe_records, columns=["ticker", "company_name"])
    return fetch_many_news(universe, limit=limit, max_workers=max_workers, use_yahoo=use_yahoo, use_google=use_google)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_fetch_ksei_profiles(tickers: tuple[str, ...], max_workers: int):
    return fetch_many_ksei_profiles(tickers, max_workers=max_workers)


@st.cache_data(ttl=21600, show_spinner=False)
def cached_fetch_fundamentals(tickers: tuple[str, ...], max_workers: int):
    return fetch_many_fundamentals(tickers, max_workers=max_workers)


def read_optional_csv(uploaded: Any) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(uploaded)
    except Exception as exc:
        st.warning(f"CSV opsional tidak dapat dibaca: {exc}")
        return pd.DataFrame()


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
    local = local.sort_values(
        ["_state_order", "emir_conviction_score", "broker_inventory_score", "smart_money_score", "liquidity_score"],
        ascending=[True, False, False, False, False],
        na_position="last",
    ).drop(columns="_state_order")
    return local.reset_index(drop=True)


def make_chart(frame: pd.DataFrame, row: pd.Series) -> go.Figure:
    local = frame.tail(180)
    figure = go.Figure()
    figure.add_trace(
        go.Candlestick(
            x=local.index,
            open=local["Open"], high=local["High"], low=local["Low"], close=local["Close"],
            name="OHLC",
        )
    )
    for span in (20, 50, 200):
        figure.add_trace(
            go.Scatter(
                x=local.index,
                y=local["Close"].ewm(span=span, adjust=False).mean(),
                mode="lines",
                name=f"EMA{span}",
            )
        )
    for label, key in (
        ("Entry Low", "entry_low"), ("Entry High", "entry_high"), ("Trigger", "trigger"),
        ("Stop", "stop_loss"), ("TP1", "tp1"), ("TP2", "tp2"), ("Defended", "defended_level"),
    ):
        value = pd.to_numeric(pd.Series([row.get(key)]), errors="coerce").iloc[0]
        if np.isfinite(value):
            figure.add_hline(y=float(value), annotation_text=label)
    figure.update_layout(height=620, xaxis_rangeslider_visible=False, title=str(row.get("ticker")))
    return figure


def secrets_mapping() -> Any:
    try:
        return st.secrets
    except Exception:
        return {}


def render_methodology() -> None:
    with st.expander("Kontrak Emir Public Framework — public-only clean-room", expanded=False):
        st.markdown(
            """
Scanner ini memodelkan kerangka publik Emir melalui pipeline otomatis: **ticker → persistent cache → incremental OHLCV/KSEI/news/fundamental → market/sector context → thesis → flow/inventory proxy → market structure → EOD microstructure proxy → scenario/invalidation/risk**.

Broker inventory dan bid–offer otomatis adalah `EMPIRICAL_PROXY`, bukan data broker atau live market depth. `EMIR_AUTO_EOD_READY` memakai position cap rendah; `EMIR_READY_WITH_PRECISE_TRIGGER` tetap memerlukan direct evidence. Bobot numerik bukan formula resmi CAK.
            """
        )


st.title("IDX Emir Autonomous Scanner")
st.caption(f"Versi {ENGINE_VERSION} · Database-first + persistent cache · incremental public-data refresh + transparent EOD proxies")
render_methodology()

# Database is mandatory. Results are not published until exact Supabase readback succeeds.
db_config = config_from_mapping(secrets_mapping())
db_state = database_status(db_config)
with st.expander("Database connection & readiness", expanded=False):
    st.dataframe(pd.DataFrame([db_state]), width="stretch", hide_index=True)
    if st.button("Test koneksi database", key="db_preflight"):
        st.session_state["emir_db_preflight"] = test_connection(db_config)
    preflight = st.session_state.get("emir_db_preflight")
    if isinstance(preflight, pd.DataFrame):
        summary_state = str(preflight.iloc[0]["state"])
        if summary_state == "HEALTHY_EMIR_DATABASE_V6":
            st.success(summary_state)
        else:
            st.warning(summary_state)
        st.dataframe(preflight, width="stretch", hide_index=True)

with st.sidebar:
    st.header("Autonomous Emir Scan")
    universe_file = st.file_uploader("Upload CSV ticker", type=["csv"], key="universe")
    scan_mode = st.selectbox(
        "Mode",
        ["EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP", "EMIR_AUTONOMOUS_DEEP_REVIEW", "EMIR_FLOW_RADAR_ONLY"],
        help="HYBRID: 400-ticker OHLCV radar, then automatic KSEI/news/fundamental deep review. DEEP: all ticker, recommended ≤100.",
    )
    period = st.selectbox("OHLCV history", ["3y", "5y"], index=1)
    completed_only = st.checkbox("Gunakan completed daily session saja", value=True)
    workers = st.slider("Concurrent provider workers", 1, 4, 3)
    deep_limit = st.slider("Maksimum deep autonomous review", 5, 100, 30, disabled=scan_mode == "EMIR_FLOW_RADAR_ONLY")
    news_per_ticker = st.slider("News per deep ticker per provider", 2, 12, 6, disabled=scan_mode == "EMIR_FLOW_RADAR_ONLY")
    use_google_news = st.checkbox("Google News RSS", value=True, disabled=scan_mode == "EMIR_FLOW_RADAR_ONLY")
    use_yahoo_news = st.checkbox("Yahoo public news", value=True, disabled=scan_mode == "EMIR_FLOW_RADAR_ONLY")
    auto_ksei = st.checkbox("Ambil profil & corporate action KSEI otomatis", value=True)
    auto_fundamental = st.checkbox("Ambil fundamental publik otomatis untuk shortlist", value=True, disabled=scan_mode == "EMIR_FLOW_RADAR_ONLY")
    force_cache_refresh = st.checkbox("Paksa refresh seluruh cache", value=False, help="Gunakan hanya jika data cache dicurigai rusak atau provider sudah memperbarui data.")
    capital = st.number_input("Modal simulasi (IDR)", min_value=100_000.0, value=5_000_000.0, step=100_000.0)
    risk_pct = st.slider("Risk budget per idea (%)", 0.25, 2.0, 1.0, 0.25)
    max_position_cap_pct = st.slider("Maximum position cap (%)", 2.0, 30.0, 20.0, 1.0)
    calibration_mode = st.selectbox("Outcome calibration mode", ["SHADOW_ONLY", "GUARDED"], index=0)

    broker_file = narrative_file = ownership_file = orderbook_file = idx_integrity_file = outcome_file = None
    with st.expander("Advanced direct-evidence overrides (optional)", expanded=False):
        st.caption("Tidak diperlukan untuk autonomous EOD. Gunakan hanya untuk upgrade evidence tier.")
        broker_file = st.file_uploader("Direct broker inventory CSV", type=["csv"], key="broker")
        narrative_file = st.file_uploader("Direct narrative/issuer event CSV", type=["csv"], key="narrative")
        ownership_file = st.file_uploader("Direct ownership/free-float CSV", type=["csv"], key="ownership")
        orderbook_file = st.file_uploader("Direct bid-offer CSV", type=["csv"], key="orderbook")
        idx_integrity_file = st.file_uploader("Direct IDX integrity CSV", type=["csv"], key="idx_integrity")
        outcome_file = st.file_uploader("Verified outcome memory CSV", type=["csv"], key="outcomes")
    st.caption("Database v6 wajib sehat. Cache sumber dan hasil scan harus lolos write + exact readback sebelum hasil diterbitkan.")
    run_scan = st.button(
        "Jalankan autonomous scanner", type="primary", width="stretch",
        disabled=not db_config.ready,
    )

if universe_file is None:
    st.info(
        "Upload CSV dengan satu kolom `ticker`. Metadata lain akan dicoba diisi otomatis dari sumber publik."
    )
    st.stop()

try:
    universe = parse_universe_frame(universe_file)
except Exception as exc:
    st.error(f"Universe CSV gagal dibaca: {exc}")
    st.stop()
if universe.empty:
    st.error("Tidak ada ticker valid pada CSV.")
    st.stop()

tickers = universe["ticker"].tolist()
st.write(f"Universe terdeteksi: **{len(tickers)} ticker unik**")
if scan_mode == "EMIR_AUTONOMOUS_DEEP_REVIEW" and len(tickers) > 100:
    st.warning("EMIR_DEEP_REVIEW untuk >100 ticker berpotensi lambat. Gunakan HYBRID.")
st.caption("CSV ticker-only didukung; company name dan sector akan diisi dari KSEI bila provider tersedia.")

if run_scan:
    # Clear any previously published result before a new official scan starts.
    st.session_state.pop("emir_scan", None)
    st.session_state.pop("emir_commit_failure", None)
    st.session_state.pop("emir_cache_failure", None)
    st.session_state.pop("emir_verification", None)

    db_config = config_from_mapping(secrets_mapping())
    preflight = test_connection(db_config)
    st.session_state["emir_db_preflight"] = preflight
    preflight_state = str(preflight.iloc[0].get("state", "")) if not preflight.empty else "DATABASE_NOT_READY_V6"
    if preflight_state != "HEALTHY_EMIR_DATABASE_V6":
        st.error("Scan diblokir: database v6 belum sehat. Jalankan migration_v6.sql; tidak ada hasil resmi yang diterbitkan.")
        st.dataframe(preflight, width="stretch", hide_index=True)
        st.stop()

    as_of = pd.Timestamp.now(tz="Asia/Jakarta")
    scan_id = make_scan_id(as_of, tickers)
    progress = st.progress(0, text="Database sehat. Membaca cache IHSG lalu refresh incremental bila diperlukan...")
    benchmark_frames, benchmark_audit, benchmark_cache_rows = fetch_ohlcv_cache_first(
        db_config, ("^JKSE",), period=period, max_workers=min(workers, 3), completed_only=completed_only,
        now=as_of, force_refresh=force_cache_refresh, last_scan_id=scan_id,
    )
    benchmark = benchmark_frames.get("^JKSE", pd.DataFrame())

    progress.progress(7, text=f"Membaca cache OHLCV {len(tickers)} ticker dan mengambil hanya data yang berubah...")
    frames, ohlcv_audit, ohlcv_cache_rows = fetch_ohlcv_cache_first(
        db_config, tuple(tickers), period=period, max_workers=workers, completed_only=completed_only,
        now=as_of, force_refresh=force_cache_refresh, last_scan_id=scan_id,
    )
    benchmark_freshness = assess_benchmark_freshness(benchmark, frames, min_universe_count=min(20, max(1, len(tickers))))
    benchmark_for_features = benchmark if benchmark_freshness.get("benchmark_usable") else pd.DataFrame()
    market_context = calculate_market_context(benchmark_for_features)
    if not benchmark_audit.empty and not benchmark_freshness.get("benchmark_usable"):
        benchmark_audit = benchmark_audit.copy()
        benchmark_audit["quality_state"] = benchmark_freshness.get("benchmark_freshness_state")
        benchmark_audit["completed_session_state"] = "STALE_RELATIVE_TO_UNIVERSE"
        benchmark_audit["detail"] = benchmark_audit.get("detail", "").astype(str) + (
            "; universe_reference_date=" + str(benchmark_freshness.get("universe_reference_date") or "")
            + "; business_lag=" + str(benchmark_freshness.get("benchmark_business_lag_days"))
        )

    progress.progress(35, text="Mengisi company profile, sector, dan corporate action KSEI...")
    ksei_profiles = pd.DataFrame()
    ksei_actions = pd.DataFrame()
    ksei_audit = pd.DataFrame()
    ksei_cache_rows: list[dict[str, Any]] = []
    if auto_ksei:
        ksei_profiles, ksei_actions, ksei_audit, ksei_cache_rows = fetch_ksei_cache_first(
            db_config, tuple(tickers), max_workers=workers, now=as_of,
            force_refresh=force_cache_refresh, last_scan_id=scan_id,
        )
        if not ksei_profiles.empty:
            profile_index = ksei_profiles.drop_duplicates("ticker").set_index("ticker")
            for column in ("company_name", "sector"):
                if column in profile_index.columns:
                    mapped = universe["ticker"].map(profile_index[column]).fillna("").astype(str).str.strip()
                    universe[column] = universe[column].where(universe[column].astype(str).str.strip().ne(""), mapped)

    progress.progress(47, text="Membaca structure, seller exhaustion, flow, dan distribution...")
    fast_rows: list[dict[str, Any]] = []
    for ticker in tickers:
        fast_rows.append({"ticker": ticker, **calculate_market_features(frames.get(ticker, pd.DataFrame()), benchmark_for_features, as_of=as_of)})
    fast = pd.DataFrame(fast_rows)
    if str(market_context.get("market_regime")) == "MARKET_CONTEXT_UNAVAILABLE":
        market_context = calculate_market_context_from_universe(fast)
    sector_map = calculate_sector_context(fast, universe)

    eligible = fast[fast.get("feature_state", "").eq("OK")].copy() if not fast.empty else pd.DataFrame()
    if not eligible.empty:
        eligible["emir_discovery_score"] = (
            0.27 * pd.to_numeric(eligible["smart_money_score"], errors="coerce")
            + 0.21 * pd.to_numeric(eligible["market_structure_score"], errors="coerce")
            + 0.13 * pd.to_numeric(eligible["seller_exhaustion_score"], errors="coerce")
            + 0.12 * pd.to_numeric(eligible["absorption_score"], errors="coerce")
            + 0.09 * pd.to_numeric(eligible["relative_strength60_pct"], errors="coerce").clip(-20, 20).add(20).mul(2.5)
            + 0.08 * pd.to_numeric(eligible["trend_score"], errors="coerce")
            + 0.05 * pd.to_numeric(eligible["liquidity_score"], errors="coerce")
            - 0.03 * pd.to_numeric(eligible["distribution_score"], errors="coerce")
            - 0.02 * pd.to_numeric(eligible["crowding_score"], errors="coerce")
        )
        eligible["sector_overlay"] = eligible["ticker"].map(
            lambda ticker: float(sector_map.get(ticker, {}).get("sector_leadership_score", 50))
        )
        eligible["emir_discovery_score"] += 0.05 * (eligible["sector_overlay"] - 50)

    if scan_mode == "EMIR_AUTONOMOUS_DEEP_REVIEW":
        shortlist = eligible["ticker"].tolist()
    elif scan_mode == "EMIR_FLOW_RADAR_ONLY":
        shortlist = []
    else:
        shortlist = eligible.sort_values("emir_discovery_score", ascending=False).head(min(deep_limit, len(eligible)))["ticker"].tolist()

    progress.progress(58, text=f"Mengambil news dan fundamental publik untuk {len(shortlist)} deep ticker...")
    manual_events = normalize_manual_events(read_optional_csv(narrative_file))
    online_events, news_audit = pd.DataFrame(), pd.DataFrame()
    if shortlist:
        shortlist_universe = universe[universe["ticker"].isin(shortlist)][["ticker", "company_name"]]
        online_events, news_audit, news_cache_rows = fetch_news_cache_first(
            db_config, shortlist_universe, limit=news_per_ticker, max_workers=workers,
            use_yahoo=use_yahoo_news, use_google=use_google_news, now=as_of,
            force_refresh=force_cache_refresh, last_scan_id=scan_id,
        )
    else:
        news_cache_rows = []
    ksei_events = ksei_actions_to_events(ksei_actions, as_of=as_of)
    event_frames = [frame for frame in (manual_events, online_events, ksei_events) if isinstance(frame, pd.DataFrame) and not frame.empty]
    all_events = pd.concat(event_frames, ignore_index=True, sort=False) if event_frames else pd.DataFrame()
    if not all_events.empty:
        all_events["ticker"] = all_events["ticker"].map(normalize_ticker)
        all_events = all_events.drop_duplicates(["ticker", "title", "url"], keep="first")

    fundamental_frame, fundamental_audit = pd.DataFrame(), pd.DataFrame()
    fundamental_cache_rows: list[dict[str, Any]] = []
    if shortlist and auto_fundamental:
        fundamental_frame, fundamental_audit, fundamental_cache_rows = fetch_fundamental_cache_first(
            db_config, tuple(shortlist), max_workers=min(workers, 3), now=as_of,
            force_refresh=force_cache_refresh, last_scan_id=scan_id,
        )
    fundamental_map = fundamental_frame.set_index("ticker").to_dict(orient="index") if not fundamental_frame.empty else {}

    progress.progress(66, text="Menyimpan cache sumber dan memverifikasi hash readback...")
    cache_write_report, cache_verification = persist_verify_cache_bundle(
        db_config,
        scan_id=scan_id,
        ohlcv_rows=[*benchmark_cache_rows, *ohlcv_cache_rows],
        source_rows=[*ksei_cache_rows, *news_cache_rows, *fundamental_cache_rows],
    )
    if not cache_commit_succeeded(cache_verification):
        st.session_state["emir_cache_failure"] = {
            "scan_id": scan_id,
            "cache_write_report": cache_write_report,
            "cache_verification": cache_verification,
        }
        progress.progress(100, text="CACHE_NOT_COMMITTED — hasil diblokir")
        st.error("CACHE_NOT_COMMITTED: cache sumber gagal ditulis atau hash readback tidak exact. Hasil scan tidak diterbitkan.")
        st.dataframe(cache_write_report, width="stretch", hide_index=True)
        st.dataframe(cache_verification, width="stretch", hide_index=True)
        st.stop()

    progress.progress(69, text="Membangun broker inventory dan bid-offer EOD proxy...")
    broker_proxy_map = {str(row["ticker"]): build_broker_inventory_proxy(row.to_dict()) for _, row in fast.iterrows()}
    orderbook_proxy_map = {str(row["ticker"]): build_orderbook_proxy(row.to_dict()) for _, row in fast.iterrows()}
    ownership_auto_map, integrity_auto_map = ksei_profiles_to_maps(ksei_profiles, ksei_actions, as_of=as_of)
    integrity_auto_map = apply_regulatory_event_overlay(integrity_auto_map, all_events, as_of=as_of)

    raw_broker = read_optional_csv(broker_file)
    raw_ownership = read_optional_csv(ownership_file)
    raw_orderbook = read_optional_csv(orderbook_file)
    raw_idx_integrity = read_optional_csv(idx_integrity_file)
    raw_outcomes = read_optional_csv(outcome_file)
    broker_direct_map = aggregate_broker_summary(raw_broker)
    ownership_direct_map = parse_ownership(raw_ownership)
    orderbook_direct_map = parse_orderbook_evidence(raw_orderbook)
    integrity_direct_map = parse_idx_integrity(raw_idx_integrity, as_of=as_of)
    outcome_calibration_map = build_outcome_calibration(raw_outcomes)

    broker_map = {**broker_proxy_map, **broker_direct_map}
    ownership_map = {**ownership_auto_map, **ownership_direct_map}
    orderbook_map = {**orderbook_proxy_map, **orderbook_direct_map}
    idx_integrity_map = {**integrity_auto_map, **integrity_direct_map}
    direct_evidence = combine_direct_evidence(raw_broker, raw_ownership, raw_orderbook, raw_idx_integrity)
    autonomous_evidence = autonomous_evidence_frame(
        ksei_profiles, ksei_actions, fundamental_frame, broker_proxy_map, orderbook_proxy_map, as_of
    )
    metadata_map = universe.set_index("ticker").to_dict(orient="index")

    progress.progress(77, text="Menyusun autonomous thesis, lifecycle, scenario, dan invalidation...")
    output_rows: list[dict[str, Any]] = []
    for _, fast_row in fast.iterrows():
        ticker = str(fast_row["ticker"])
        deep_reviewed = ticker in shortlist
        ticker_events = all_events[all_events["ticker"].eq(ticker)] if deep_reviewed and not all_events.empty else pd.DataFrame()
        narrative = score_narrative_events(ticker_events, as_of=as_of, issuer_context=metadata_map.get(ticker))
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
            max_position_cap_pct=max_position_cap_pct,
            capital_idr=capital,
            risk_budget_pct=risk_pct,
            calibration_mode=calibration_mode,
        )
        output_rows.append({**metadata_map.get(ticker, {}), **profile, **position_builder(pd.Series(profile), capital, risk_pct)})

    radar = radar_sort(pd.DataFrame(output_rows))
    provider_frames = [
        ohlcv_audit.assign(audit_family="OHLCV"),
        benchmark_audit.assign(audit_family="BENCHMARK"),
        pd.DataFrame([{
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
        }]),
    ]
    if str(market_context.get("market_context_provenance_state", "")).startswith("UNIVERSE_BREADTH_PROXY"):
        provider_frames.append(pd.DataFrame([{
            "ticker": "MARKET_CONTEXT",
            "provider": "UNIVERSE_BREADTH_PROXY",
            "status": "PROXY_FALLBACK",
            "bars": int(market_context.get("market_proxy_valid_tickers", 0) or 0),
            "last_date": as_of.date().isoformat(),
            "detail": "Direct ^JKSE unavailable; market regime derived from valid universe breadth. Not direct IHSG data.",
            "audit_family": "BENCHMARK_PROXY",
        }]))
    for frame, family in ((ksei_audit, "KSEI"), (news_audit, "NARRATIVE"), (fundamental_audit, "FUNDAMENTAL")):
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            provider_frames.append(frame.assign(audit_family=family))
    provider_audit = pd.concat(provider_frames, ignore_index=True, sort=False)

    progress.progress(88, text="Database-first commit: menulis hasil scan ke Supabase setelah cache terverifikasi...")
    write_report, verification, commit_report = persist_verify_commit_scan(
        db_config,
        scan_id=scan_id,
        as_of=as_of,
        radar=radar,
        events=all_events,
        provider_audit=provider_audit,
        direct_evidence=direct_evidence,
        autonomous_evidence=autonomous_evidence,
        outcomes=raw_outcomes,
        mode=scan_mode,
    )
    commit_ok = database_commit_succeeded(commit_report)
    if not commit_ok:
        failure = {
            "scan_id": scan_id,
            "as_of": as_of,
            "write_report": write_report,
            "verification": verification,
            "commit_report": commit_report,
        }
        st.session_state["emir_commit_failure"] = failure
        progress.progress(100, text="SCAN_NOT_COMMITTED — hasil diblokir")
        st.error("SCAN_NOT_COMMITTED: write/readback database tidak 100%. Radar, ranking, dan file hasil tidak diterbitkan.")
        st.subheader("Database write report")
        st.dataframe(write_report, width="stretch", hide_index=True)
        st.subheader("Exact readback audit")
        st.dataframe(verification, width="stretch", hide_index=True)
        st.subheader("Final commit state")
        st.dataframe(commit_report, width="stretch", hide_index=True)
        st.download_button(
            "Download database failure audit",
            pd.concat([
                write_report.assign(report_family="WRITE"),
                verification.assign(report_family="READBACK"),
                commit_report.assign(report_family="COMMIT"),
            ], ignore_index=True, sort=False).to_csv(index=False).encode("utf-8"),
            "emir_database_commit_failure_v1_5_1.csv",
            "text/csv",
        )
        st.stop()

    st.session_state["emir_scan"] = {
        "radar": radar,
        "frames": frames,
        "events": all_events,
        "provider_audit": provider_audit,
        "direct_evidence": direct_evidence,
        "autonomous_evidence": autonomous_evidence,
        "fundamentals": fundamental_frame,
        "ksei_profiles": ksei_profiles,
        "ksei_actions": ksei_actions,
        "write_report": write_report,
        "verification": verification,
        "commit_report": commit_report,
        "cache_write_report": cache_write_report,
        "cache_verification": cache_verification,
        "cache_summary": cache_summary([benchmark_audit, ohlcv_audit, ksei_audit, news_audit, fundamental_audit]),
        "scan_id": scan_id,
        "as_of": as_of,
        "database_ready": db_config.ready,
        "database_commit_state": "DATABASE_FIRST_COMMITTED_WITH_PERSISTENT_CACHE",
        "expected_events": len(all_events),
        "expected_provider_audit": len(provider_audit),
        "expected_direct_evidence": len(direct_evidence),
        "expected_autonomous_evidence": len(autonomous_evidence),
        "expected_outcomes": len(raw_outcomes),
        "outcomes": raw_outcomes,
        "market_context": market_context,
        "benchmark_freshness": benchmark_freshness,
        "shortlist": shortlist,
    }
    st.session_state["emir_verification"] = verification
    st.session_state.pop("emir_commit_failure", None)
    st.session_state.pop("emir_cache_failure", None)
    progress.progress(100, text="DATABASE_FIRST_COMMITTED_WITH_PERSISTENT_CACHE — hasil resmi siap ditampilkan")

result = st.session_state.get("emir_scan")
if not result:
    cache_failure = st.session_state.get("emir_cache_failure")
    if isinstance(cache_failure, dict):
        st.error("CACHE_NOT_COMMITTED: cache sumber gagal ditulis atau dibaca kembali secara exact.")
        for title, key in (("Persistent cache write", "cache_write_report"), ("Persistent cache readback", "cache_verification")):
            frame = cache_failure.get(key)
            if isinstance(frame, pd.DataFrame):
                st.subheader(title)
                st.dataframe(frame, width="stretch", hide_index=True)
    failure = st.session_state.get("emir_commit_failure")
    if isinstance(failure, dict):
        st.error("SCAN_NOT_COMMITTED: hasil scan terakhir tidak diterbitkan karena database belum terverifikasi 100%.")
        for title, key in (("Database write report", "write_report"), ("Exact readback audit", "verification"), ("Final commit state", "commit_report")):
            frame = failure.get(key)
            if isinstance(frame, pd.DataFrame):
                st.subheader(title)
                st.dataframe(frame, width="stretch", hide_index=True)
    st.stop()

st.success("DATABASE_FIRST_COMMITTED_WITH_PERSISTENT_CACHE · cache sumber dan hasil scan telah ditulis serta dibaca kembali secara exact dari Supabase.")
radar = result["radar"]
metrics = st.columns(8)
metrics[0].metric("Ticker", len(radar))
metrics[1].metric("Deep reviewed", int(radar["deep_review_state"].eq("DEEP_REVIEWED").sum()))
metrics[2].metric("Auto EOD ready", int(radar["emir_decision_state"].eq("EMIR_AUTO_EOD_READY").sum()))
metrics[3].metric("Precise ready", int(radar["emir_decision_state"].eq("EMIR_READY_WITH_PRECISE_TRIGGER").sum()))
metrics[4].metric("Core thesis ready", int(radar["emir_decision_state"].isin(["EMIR_CORE_THESIS_READY_WAIT_IDX_INTEGRITY", "EMIR_THESIS_READY_WAIT_BID_OFFER"]).sum()))
metrics[5].metric("Inventory watch", int(radar["emir_decision_state"].eq("EMIR_WATCH_INVENTORY_COLLECTION").sum()))
metrics[6].metric("Wait", int(radar["emir_decision_state"].isin(["EMIR_WAIT_NARRATIVE", "EMIR_WAIT_MONEY_FLOW"]).sum()))
metrics[7].metric("Reject/Euphoria", int(radar["emir_decision_state"].isin(["EMIR_REJECT_SMART_MONEY_DISTRIBUTION", "EMIR_AVOID_RETAIL_EUPHORIA"]).sum()))
st.caption(
    f"Scan ID `{result['scan_id']}` · commit `{result.get('database_commit_state')}` · "
    f"as-of {result['as_of']} · market regime `{result['market_context'].get('market_regime')}`"
)

tabs = st.tabs([
    "Emir Radar", "Thesis & Lifecycle", "Inventory & Smart Money", "Structure & Sector",
    "IDX Integrity & Capacity", "Outcome Calibration", "Scenario & Risk", "Chart",
    "Evidence Audit", "Formula Registry", "Database",
])
(
    tab_radar, tab_thesis, tab_flow, tab_structure, tab_integrity, tab_outcomes, tab_scenario,
    tab_chart, tab_evidence, tab_formula, tab_database,
) = tabs

with tab_radar:
    columns = [
        "ticker", "company_name", "sector", "emir_decision_state", "action", "emir_lifecycle",
        "market_structure_mode", "emir_conviction_score", "emir_evidence_coverage_pct",
        "broker_inventory_score", "smart_money_score", "narrative_score", "story_runway_score",
        "sector_rrg_state", "retail_adoption_stage", "idx_integrity_state", "execution_capacity_state",
        "distribution_score", "crowding_score", "risk_flags",
    ]
    st.dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)
    st.download_button(
        "Download Emir radar CSV",
        radar.to_csv(index=False).encode("utf-8"),
        "idx_emir_autonomous_radar_v1_5_1.csv",
        "text/csv",
    )

with tab_thesis:
    columns = [
        "ticker", "emir_lifecycle", "narrative_state", "narrative_category", "narrative_event_count",
        "narrative_materiality_score", "top_down_catalyst_score", "industry_translation_score",
        "story_runway_score", "financial_conversion_score", "fundamental_conversion_score", "fundamental_state",
        "fundamental_coverage_pct", "fundamental_statement_availability_pct",
        "fundamental_critical_metric_completeness_pct", "fundamental_official_source_coverage_pct",
        "revenue_growth_pct", "earnings_growth_pct", "ocf_conversion_ratio",
        "conversion_path", "issuer_alignment_score",
        "retail_adoption_stage", "thesis_statement", "narrative_latest_title", "what_must_happen_next",
    ]
    st.dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)

with tab_flow:
    columns = [
        "ticker", "broker_inventory_score", "broker_inventory_coverage_pct", "broker_inventory_shift_state",
        "holder_persistence_score", "inventory_dryness_score", "retail_exit_score",
        "retail_cannibalisation_risk", "fund_like_flow_score", "jumbo_crossing_score", "defended_level",
        "smart_money_score", "absorption_score", "up_value_ratio20_pct", "close_acceptance20_pct",
        "accumulation_days20", "absorption_days20", "distribution_days20", "failed_absorption_days20",
        "beneficial_owner_inference_state",
    ]
    st.dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)

with tab_structure:
    columns = [
        "ticker", "market_structure_mode", "market_structure_score", "reversal_score",
        "continuation_price_flow_score", "sideways_quality_score", "seller_exhaustion_score",
        "structure_change_score", "fakeout_reclaim_score", "range_compression_score",
        "relative_strength20_pct", "relative_strength60_pct", "relative_strength_momentum_pct",
        "sector", "sector_rrg_state", "sector_leadership_score", "market_regime", "market_context_score",
        "reported_free_float_pct", "effective_free_float_pct", "fake_float_gap_pct", "passive_flow_risk_score",
    ]
    st.dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)

with tab_integrity:
    columns = [
        "ticker", "idx_integrity_state", "idx_integrity_score", "idx_integrity_coverage_pct",
        "listing_board", "listing_board_verification_state",
        "hsc_flag", "hsc_verification_state",
        "special_monitoring_flag", "special_monitoring_verification_state",
        "full_call_auction_flag", "full_call_auction_verification_state",
        "suspension_flag", "suspension_verification_state",
        "uma_flag", "uma_verification_state", "sanctions_flag", "sanctions_verification_state",
        "regulatory_free_float_pct", "regulatory_free_float_verification_state",
        "idx_integrity_unknown_critical_count",
        "corporate_action_flag", "corporate_action_type", "idx_integrity_age_days",
        "ohlcv_integrity_state", "ohlcv_integrity_score", "corporate_action_anomaly_flag",
        "execution_friction_score", "gap_risk_score", "extreme_move_risk_score",
        "requested_position_value_idr", "max_safe_position_value_idr",
        "estimated_participation_rate_pct", "max_participation_rate_pct",
        "slippage_bps_proxy", "execution_capacity_state",
    ]
    st.dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)
    st.caption("IDX integrity is a direct-evidence production gate. HSC, special monitoring/FCA, suspension, serious sanctions, stale evidence, or unresolved corporate-action anomalies cannot be treated as bullish scarcity.")

with tab_outcomes:
    columns = [
        "ticker", "emir_lifecycle", "market_structure_mode", "calibration_mode",
        "outcome_sample_n", "outcome_win_rate_pct", "outcome_median_return_pct",
        "outcome_median_drawdown_pct", "outcome_thesis_invalidation_rate_pct",
        "outcome_calibration_state", "emir_decision_state",
    ]
    st.dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)
    if isinstance(result.get("outcomes"), pd.DataFrame) and not result["outcomes"].empty:
        st.download_button(
            "Download uploaded outcome memory",
            result["outcomes"].to_csv(index=False).encode("utf-8"),
            "idx_emir_outcome_memory.csv",
            "text/csv",
        )
    else:
        st.info("Belum ada outcome memory terverifikasi. Calibration tetap NO_OUTCOME_MEMORY dan tidak memengaruhi score.")

with tab_scenario:
    columns = [
        "ticker", "emir_decision_state", "action", "why_now", "what_must_happen_next", "thesis_invalidation",
        "execution_state", "entry_low", "entry_high", "trigger", "trigger_provenance", "precise_trigger_price",
        "orderbook_trigger_score", "stop_loss", "hard_stop_distance_pct", "tp1", "tp2", "rr_tp1", "rr_tp2",
        "position_cap_pct", "lot", "position_value", "risk_idr", "position_state",
        "max_safe_position_value_idr", "estimated_participation_rate_pct", "slippage_bps_proxy",
        "execution_capacity_state", "trim_state",
    ]
    st.dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)
    st.caption(
        "Level harga adalah scenario plan. Precise entry hanya aktif bila direct bid-offer evidence terverifikasi. "
        "1.8R/3R dan 5% cap adalah empirical/public-discussion proxies, bukan formula resmi CAK."
    )

with tab_chart:
    selected = st.selectbox("Ticker", radar["ticker"].tolist())
    row = radar.loc[radar["ticker"].eq(selected)].iloc[0]
    frame = result["frames"].get(selected, pd.DataFrame())
    if frame.empty:
        st.warning("OHLCV tidak tersedia.")
    else:
        st.plotly_chart(make_chart(frame, row), width="stretch")
        st.json({
            key: row.get(key)
            for key in (
                "emir_decision_state", "emir_lifecycle", "market_structure_mode", "thesis_statement",
                "why_now", "what_must_happen_next", "thesis_invalidation", "defended_level",
                "trigger_provenance", "trim_state", "risk_flags",
            )
        })

with tab_evidence:
    st.subheader("Provider audit")
    st.dataframe(result["provider_audit"], width="stretch", hide_index=True)
    st.subheader("Autonomous evidence")
    if result.get("autonomous_evidence", pd.DataFrame()).empty:
        st.warning("Autonomous provider tidak menghasilkan evidence.")
    else:
        st.dataframe(result["autonomous_evidence"], width="stretch", hide_index=True)
        st.download_button(
            "Download autonomous evidence",
            result["autonomous_evidence"].to_csv(index=False).encode("utf-8"),
            "idx_emir_autonomous_evidence_v1_5_1.csv",
            "text/csv",
        )
    st.subheader("Direct evidence overrides")
    if result["direct_evidence"].empty:
        st.info("Tidak ada direct evidence override. Scanner tetap berjalan dengan autonomous public data dan proxy yang diberi label eksplisit.")
    else:
        st.dataframe(result["direct_evidence"], width="stretch", hide_index=True)
        st.download_button(
            "Download direct evidence",
            result["direct_evidence"].to_csv(index=False).encode("utf-8"),
            "idx_emir_direct_evidence.csv",
            "text/csv",
        )
    st.subheader("Narrative evidence")
    if result["events"].empty:
        st.warning("Tidak ada narrative event yang berhasil dikoleksi. Flow radar bukan thesis narrative terkonfirmasi.")
    else:
        st.dataframe(result["events"], width="stretch", hide_index=True)
        st.download_button(
            "Download narrative evidence",
            result["events"].to_csv(index=False).encode("utf-8"),
            "idx_emir_narrative_evidence.csv",
            "text/csv",
        )

with tab_formula:
    st.subheader("Public Research Formula Registry")
    registry = formula_registry_frame()
    st.dataframe(registry, width="stretch", hide_index=True)
    st.download_button(
        "Download formula registry CSV",
        registry.to_csv(index=False).encode("utf-8"),
        "idx_emir_autonomous_formula_registry_v1_5_1.csv",
        "text/csv",
    )
    st.info(
        "EXPLICIT_PUBLIC = dinyatakan secara publik; PUBLIC_SYNTHESIS = sintesis beberapa pernyataan publik; "
        "EMPIRICAL_PROXY = formula numerik independen yang harus diuji; MANUAL_EVIDENCE_REQUIRED = tidak boleh ditebak."
    )

with tab_database:
    st.subheader("Persistent source-cache commit")
    st.dataframe(result.get("cache_write_report", pd.DataFrame()), width="stretch", hide_index=True)
    st.subheader("Persistent source-cache hash readback")
    st.dataframe(result.get("cache_verification", pd.DataFrame()), width="stretch", hide_index=True)
    st.subheader("Cache utilization")
    st.dataframe(result.get("cache_summary", pd.DataFrame()), width="stretch", hide_index=True)
    st.subheader("Final database commit")
    st.dataframe(result["commit_report"], width="stretch", hide_index=True)
    st.subheader("Database write report")
    st.dataframe(result["write_report"], width="stretch", hide_index=True)
    st.subheader("Automatic exact readback")
    verification = result.get("verification")
    if isinstance(verification, pd.DataFrame):
        st.dataframe(verification, width="stretch", hide_index=True)
        st.download_button(
            "Download database readback audit",
            verification.to_csv(index=False).encode("utf-8"),
            "emir_database_readback_v6_v1_5_1.csv",
            "text/csv",
        )
    if st.button("Verifikasi ulang scan committed", key="reverify_committed_scan"):
        recheck = verify_scan(
            config_from_mapping(secrets_mapping()),
            scan_id=result["scan_id"],
            expected_radar=len(radar),
            expected_events=result["expected_events"],
            expected_provider_audit=result["expected_provider_audit"],
            expected_direct_evidence=result["expected_direct_evidence"],
            expected_autonomous_evidence=result["expected_autonomous_evidence"],
            expected_outcomes=result["expected_outcomes"],
        )
        st.session_state["emir_reverification"] = recheck
    recheck = st.session_state.get("emir_reverification")
    if isinstance(recheck, pd.DataFrame):
        state = str(recheck.iloc[0].get("state", ""))
        if state == "VERIFIED_ALL_TABLES":
            st.success("Reverification tetap 100% exact.")
        else:
            st.error(f"Database berubah atau readback gagal: {state}")
        st.dataframe(recheck, width="stretch", hide_index=True)
