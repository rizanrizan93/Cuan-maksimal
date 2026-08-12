from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ui_dataframe import streamlit_dataframe as safe_dataframe

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
    config_from_mapping, database_status, full_persistence_succeeded,
    persist_verify_scan_best_effort, scan_publication_allowed, test_connection, verify_scan,
)
from scan_jobs import (
    ACTIVE_JOB_STATUSES, cancel_scan_job, create_scan_job, find_latest_job, get_scan_job,
    job_status_frame, universe_hash,
)
from resumable_scan import load_persisted_scan_result, process_next_job_step
from top3_dashboard import enrich_dashboard_scores, render_top3_dashboard_html, select_top3, select_next_leaders, select_real_money_top3
from dashboard_persistence import build_database_transfer_summary, database_transfer_totals
from persistent_cache import (
    cache_commit_succeeded,
    cache_persistence_state,
    cache_summary,
    fetch_fundamental_cache_first,
    fetch_ksei_cache_first,
    fetch_news_cache_first,
    fetch_ohlcv_cache_first,
    load_cached_ohlcv_frames,
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
        if hasattr(uploaded, "seek"):
            uploaded.seek(0)
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
        # Accessing ``st.secrets`` itself is lazy and does not raise when no
        # secrets file exists; converting it forces parsing inside this guard.
        return dict(st.secrets)
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
st.caption(
    f"Versi {ENGINE_VERSION} · resumable chunked scan · progressive deep review · "
    "hasil dan evidence dipindahkan ke Supabase secara terukur"
)
render_methodology()

# The database owns resumable checkpoints. Final result persistence remains best-effort.
db_config = config_from_mapping(secrets_mapping())
db_state = database_status(db_config)
with st.expander("Database connection & resumable-job readiness", expanded=False):
    safe_dataframe(pd.DataFrame([db_state]), width="stretch", hide_index=True)
    if st.button("Test koneksi database", key="db_preflight"):
        st.session_state["emir_db_preflight"] = test_connection(db_config)
    preflight = st.session_state.get("emir_db_preflight")
    if isinstance(preflight, pd.DataFrame):
        summary_state = str(preflight.iloc[0]["state"])
        if summary_state == "HEALTHY_EMIR_DATABASE_V8":
            st.success(summary_state)
        else:
            st.warning(summary_state)
        safe_dataframe(preflight, width="stretch", hide_index=True)

with st.sidebar:
    st.header("Resumable Emir Scan")
    universe_file = st.file_uploader("Upload CSV ticker", type=["csv"], key="universe")
    scan_mode = st.selectbox(
        "Mode",
        ["EMIR_AUTONOMOUS_HYBRID_400_TO_DEEP", "EMIR_AUTONOMOUS_DEEP_REVIEW", "EMIR_FLOW_RADAR_ONLY"],
        help="HYBRID memproses OHLCV seluruh universe, lalu deep review progresif sesuai cakupan yang dipilih.",
    )
    period = st.selectbox("OHLCV history", ["3y", "5y"], index=1)
    completed_only = st.checkbox("Gunakan completed daily session saja", value=True)
    workers = st.slider("Concurrent provider workers", 1, 4, 3)
    chunk_size = st.slider("Ticker per checkpoint", 10, 30, 20, 5, help="Setiap batch disimpan sebelum batch berikutnya.")
    deep_scope_label = st.selectbox(
        "Cakupan deep review",
        [
            "Semua ticker eligible (progresif)",
            "Seimbang — Top 60",
            "Cepat — Top 30",
            "Batas custom",
        ],
        index=0,
        disabled=scan_mode == "EMIR_FLOW_RADAR_ONLY",
        help=(
            "Semua ticker eligible diproses bertahap per checkpoint. Koneksi dapat diputus dan scan dilanjutkan. "
            "Mode cepat/seimbang tetap tersedia untuk kebutuhan harian."
        ),
    )
    deep_review_scope = {
        "Semua ticker eligible (progresif)": "ALL_ELIGIBLE",
        "Seimbang — Top 60": "BALANCED_TOP_60",
        "Cepat — Top 30": "FAST_TOP_30",
        "Batas custom": "CUSTOM_LIMIT",
    }[deep_scope_label]
    deep_limit = st.slider(
        "Batas custom deep review",
        5,
        500,
        100,
        5,
        disabled=scan_mode == "EMIR_FLOW_RADAR_ONLY" or deep_review_scope != "CUSTOM_LIMIT",
    )
    news_per_ticker = st.slider("News per deep ticker per provider", 2, 10, 6, disabled=scan_mode == "EMIR_FLOW_RADAR_ONLY")
    use_google_news = st.checkbox("Google News RSS", value=True, disabled=scan_mode == "EMIR_FLOW_RADAR_ONLY")
    use_yahoo_news = st.checkbox("Yahoo public news", value=True, disabled=scan_mode == "EMIR_FLOW_RADAR_ONLY")
    auto_ksei = st.checkbox("KSEI untuk target deep review", value=True)
    auto_fundamental = st.checkbox("Fundamental public proxy untuk target deep review", value=True, disabled=scan_mode == "EMIR_FLOW_RADAR_ONLY")
    auto_idx_official_fundamental = st.checkbox("IDX official XBRL untuk deep universe (recommended)", value=True, disabled=scan_mode == "EMIR_FLOW_RADAR_ONLY")
    official_fundamental_limit = st.slider("Batas IDX official deep review", 10, 500, 400, 10, disabled=not auto_idx_official_fundamental or scan_mode == "EMIR_FLOW_RADAR_ONLY", help="Untuk universe 400 ticker, 400 direkomendasikan pada full refresh pertama; hasil official disimpan di persistent cache.")
    force_cache_refresh = st.checkbox("Paksa refresh cache", value=False)
    capital_mode = st.selectbox("Capital mode", ["GUARDED_REAL_MONEY", "RESEARCH"], index=0)
    capital = st.number_input("Modal (IDR)", min_value=100_000.0, value=5_000_000.0, step=100_000.0)
    risk_pct = st.slider("Risk budget per idea (%)", 0.25, 2.0, 0.5 if capital_mode == "GUARDED_REAL_MONEY" else 1.0, 0.25)
    max_position_cap_pct = st.slider("Maximum position cap (%)", 2.0, 30.0, 10.0 if capital_mode == "GUARDED_REAL_MONEY" else 20.0, 1.0)
    calibration_mode = st.selectbox("Outcome calibration mode", ["GUARDED", "SHADOW_ONLY"], index=0 if capital_mode == "GUARDED_REAL_MONEY" else 1)
    if capital_mode == "GUARDED_REAL_MONEY":
        st.warning("Mode modal riil: Yahoo/public statement tetap boleh memberi fundamental score dan kandidat MANUAL_CONFIRMATION_REQUIRED bila data current/berkualitas. Official IDX/issuer adalah confidence upgrade; DIRECT_VERIFIED_READY tetap hanya jika official/cash-flow + direct IDX integrity + live bid-offer lolos. Risk proxy-only dibatasi maksimum 0,50%.")

    broker_file = narrative_file = ownership_file = orderbook_file = idx_integrity_file = outcome_file = None
    with st.expander("Advanced direct-evidence overrides (optional)", expanded=False):
        broker_file = st.file_uploader("Direct broker inventory CSV", type=["csv"], key="broker")
        narrative_file = st.file_uploader("Direct narrative/issuer event CSV", type=["csv"], key="narrative")
        ownership_file = st.file_uploader("Direct ownership/free-float CSV", type=["csv"], key="ownership")
        orderbook_file = st.file_uploader("Direct bid-offer CSV", type=["csv"], key="orderbook")
        idx_integrity_file = st.file_uploader("Direct IDX integrity CSV", type=["csv"], key="idx_integrity")
        outcome_file = st.file_uploader("Verified outcome memory CSV", type=["csv"], key="outcomes")
    st.caption(
        "Deep review berjalan progresif per checkpoint dan disimpan ke database job. Bila koneksi terputus, "
        "buka kembali lalu lanjutkan; ticker yang sudah selesai tidak diulang."
    )

if universe_file is None:
    st.info("Upload CSV dengan satu kolom `ticker`.")
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
universe_fingerprint = universe_hash(universe)
st.write(f"Universe terdeteksi: **{len(tickers)} ticker unik**")
st.caption("OHLCV diproses bertahap. Deep review kemudian berjalan progresif sesuai cakupan yang dipilih dan disimpan per checkpoint.")

preflight = st.session_state.get("emir_db_preflight")
if not isinstance(preflight, pd.DataFrame):
    preflight = test_connection(db_config)
    st.session_state["emir_db_preflight"] = preflight
preflight_state = str(preflight.iloc[0].get("state", "")) if not preflight.empty else "DATABASE_NOT_READY_V7"
job_tables_ready = preflight_state == "HEALTHY_EMIR_DATABASE_V8"
if not job_tables_ready:
    if preflight_state == "DATABASE_UNREACHABLE_OR_RESOURCE_EXHAUSTED":
        st.error(
            "Database Supabase sedang tidak menerima koneksi / resource exhausted. "
            "Ini BUKAN bukti schema v8 hilang. Pulihkan health database terlebih dahulu; "
            "jangan menjalankan migration ulang selama database belum menerima koneksi."
        )
    else:
        st.error(
            "Resumable scan belum melihat schema v8 lengkap/berizin (`cak_scan_jobs`, `cak_scan_job_chunks`, dan `cak_research_memory`). "
            "Periksa detail preflight; lakukan migration/permission repair hanya bila table memang missing atau permission gagal."
        )
    safe_dataframe(preflight, width="stretch", hide_index=True)
    st.stop()

active_scan_id = str(st.session_state.get("emir_active_scan_id") or "")
active_job = get_scan_job(db_config, active_scan_id) if active_scan_id else None

def _job_engine_matches(job: dict | None) -> bool:
    if not job:
        return False
    settings = job.get("settings") if isinstance(job.get("settings"), dict) else {}
    return str(settings.get("engine_version") or "") == ENGINE_VERSION

if active_job and not _job_engine_matches(active_job):
    # Never resume a pre-upgrade job under a new scoring/fundamental schema.
    st.session_state.pop("emir_active_scan_id", None)
    active_scan_id = ""
    active_job = None
if active_job and str(active_job.get("status")) not in ACTIVE_JOB_STATUSES:
    if str(active_job.get("status")) in {"COMPLETED", "COMPLETED_PARTIAL_PERSISTENCE"} and "emir_scan" not in st.session_state:
        loaded = load_persisted_scan_result(db_config, str(active_job.get("scan_id")))
        if loaded:
            st.session_state["emir_scan"] = loaded
    st.session_state.pop("emir_active_scan_id", None)
    active_job = None
if not active_job or str(active_job.get("universe_hash")) != universe_fingerprint:
    active_job = find_latest_job(db_config, universe_hash_value=universe_fingerprint, include_completed=False)
    if active_job and _job_engine_matches(active_job):
        active_scan_id = str(active_job.get("scan_id"))
        st.session_state["emir_active_scan_id"] = active_scan_id
    elif active_job:
        active_job = None

latest_any = find_latest_job(db_config, universe_hash_value=universe_fingerprint, include_completed=True)
if latest_any and not _job_engine_matches(latest_any):
    latest_any = None

def build_current_job_settings() -> dict[str, Any]:
    return {
        "engine_version": ENGINE_VERSION,
        "scan_mode": scan_mode,
        "period": period,
        "completed_only": completed_only,
        "workers": workers,
        "deep_review_scope": deep_review_scope,
        "deep_limit": deep_limit,
        "news_per_ticker": news_per_ticker,
        "use_google_news": use_google_news,
        "use_yahoo_news": use_yahoo_news,
        "auto_ksei": auto_ksei,
        "auto_fundamental": auto_fundamental,
        "auto_idx_official_fundamental": auto_idx_official_fundamental,
        "official_fundamental_limit": official_fundamental_limit,
        "force_cache_refresh": force_cache_refresh,
        "capital_mode": capital_mode,
        "capital": capital,
        "risk_pct": risk_pct,
        "max_position_cap_pct": max_position_cap_pct,
        "calibration_mode": calibration_mode,
        "manual_broker": read_optional_csv(broker_file).to_dict(orient="records"),
        "manual_events": read_optional_csv(narrative_file).to_dict(orient="records"),
        "manual_ownership": read_optional_csv(ownership_file).to_dict(orient="records"),
        "manual_orderbook": read_optional_csv(orderbook_file).to_dict(orient="records"),
        "manual_idx_integrity": read_optional_csv(idx_integrity_file).to_dict(orient="records"),
        "manual_outcomes": read_optional_csv(outcome_file).to_dict(orient="records"),
    }


def start_new_scan_job(*, auto_continue: bool = True) -> None:
    as_of = pd.Timestamp.now(tz="Asia/Jakarta")
    scan_id = make_scan_id(as_of, tickers)
    try:
        create_scan_job(
            db_config,
            scan_id=scan_id,
            universe=universe,
            settings=build_current_job_settings(),
            chunk_size=chunk_size,
        )
    except Exception as exc:
        st.session_state["emir_auto_continue"] = False
        st.error("Job scan belum berhasil dibuat di Supabase. Tidak ada checkpoint yang dihapus; silakan uji koneksi lalu coba lagi.")
        st.caption(f"Detail database: {exc}")
        return
    st.session_state["emir_active_scan_id"] = scan_id
    st.session_state["emir_auto_continue"] = bool(auto_continue)
    st.session_state.pop("emir_scan", None)
    st.session_state.pop("emir_reverification", None)
    st.rerun()


existing_result = st.session_state.get("emir_scan")
start_label = "🚀 Mulai Scan" if not existing_result else "🔄 Scan Ulang Universe Ini"
start_clicked = st.button(
    start_label,
    type="primary",
    disabled=bool(active_job),
    key="dashboard_start_scan",
    width="stretch",
    help="Membuat job database, memproses checkpoint otomatis, dan menyimpan hasil scan ke Supabase secara best-effort.",
)
if start_clicked:
    start_new_scan_job(auto_continue=True)

control_cols = st.columns(3)
resume_clicked = control_cols[0].button("Proses 1 checkpoint", disabled=not bool(active_job), width="stretch")
auto_clicked = control_cols[1].button("Lanjut otomatis", disabled=not bool(active_job), width="stretch")
pause_clicked = control_cols[2].button("Jeda", disabled=not bool(active_job), width="stretch")

if pause_clicked:
    st.session_state["emir_auto_continue"] = False
if auto_clicked:
    st.session_state["emir_auto_continue"] = True

if active_job:
    st.subheader("Resumable scan job")
    safe_dataframe(job_status_frame(active_job), width="stretch", hide_index=True)
    progress_value = float(active_job.get("progress_pct") or 0.0)
    st.progress(min(100, max(0, int(progress_value))), text=f"{active_job.get('current_stage')} · {progress_value:.1f}%")
    job_shortlist = list(active_job.get("shortlist") or [])
    if job_shortlist:
        current_stage = str(active_job.get("current_stage") or "")
        deep_stage_offset = int(active_job.get("current_offset") or 0) if current_stage in {"KSEI_SHORTLIST", "NEWS_SHORTLIST", "FUNDAMENTAL_SHORTLIST", "IDX_FUNDAMENTAL_SHORTLIST"} else 0
        progress_cols = st.columns(3)
        active_stage_target = min(len(job_shortlist), int((active_job.get("settings") or {}).get("official_fundamental_limit") or 400)) if current_stage == "IDX_FUNDAMENTAL_SHORTLIST" else len(job_shortlist)
        progress_cols[0].metric("Target deep review", len(job_shortlist))
        progress_cols[1].metric("Progress stage aktif", min(deep_stage_offset, active_stage_target))
        progress_cols[2].metric("Scope", str((active_job.get("settings") or {}).get("deep_review_scope") or "ALL_ELIGIBLE"))
    failures = active_job.get("failures") or {}
    if failures:
        failed_total = len(set(sum((list(values) for values in failures.values()), [])))
        if failed_total:
            st.warning(f"{failed_total} ticker/provider item gagal dan dilewati atau menunggu retry terbatas.")

    cancel_clicked = st.button("Batalkan job aktif", disabled=str(active_job.get("status")) not in ACTIVE_JOB_STATUSES)
    if cancel_clicked:
        cancel_scan_job(db_config, str(active_job.get("scan_id")))
        st.session_state["emir_auto_continue"] = False
        st.session_state.pop("emir_active_scan_id", None)
        st.rerun()

    should_process = resume_clicked or bool(st.session_state.get("emir_auto_continue", False))
    if should_process and str(active_job.get("status")) in ACTIVE_JOB_STATUSES:
        try:
            with st.spinner(f"Memproses checkpoint {active_job.get('current_stage')}..."):
                updated_job, step_report, step_result = process_next_job_step(db_config, active_job)
        except Exception as exc:
            st.session_state["emir_auto_continue"] = False
            st.error("Checkpoint berhenti aman. Progress yang sudah committed tetap dapat dilanjutkan.")
            st.caption(f"Detail checkpoint: {exc}")
            st.stop()
        active_job = updated_job
        st.session_state["emir_active_scan_id"] = str(updated_job.get("scan_id"))
        if step_result:
            st.session_state["emir_scan"] = step_result
        state = str(step_report.get("state") or "")
        if "FAILED" in state:
            st.warning(state)
            st.session_state["emir_auto_continue"] = False
        if str(updated_job.get("status")) in {"COMPLETED", "COMPLETED_PARTIAL_PERSISTENCE"}:
            st.session_state["emir_auto_continue"] = False
            if "emir_scan" not in st.session_state:
                loaded = load_persisted_scan_result(db_config, str(updated_job.get("scan_id")))
                if loaded:
                    st.session_state["emir_scan"] = loaded
        if bool(st.session_state.get("emir_auto_continue", False)) and str(updated_job.get("status")) in ACTIVE_JOB_STATUSES:
            st.rerun()
        st.rerun()

if not active_job and latest_any and str(latest_any.get("status")) in {"COMPLETED", "COMPLETED_PARTIAL_PERSISTENCE"}:
    st.info(f"Hasil terakhir tersedia: `{latest_any.get('scan_id')}`")
    if st.button("Muat hasil scan terakhir"):
        loaded = load_persisted_scan_result(db_config, str(latest_any.get("scan_id")))
        if loaded:
            st.session_state["emir_scan"] = loaded
        else:
            st.warning("Radar hasil terakhir tidak ditemukan lengkap di database.")

result = st.session_state.get("emir_scan")
if not result:
    st.stop()

persistence_state = str(result.get("database_commit_state") or "SCAN_COMPLETED_MEMORY_ONLY")
if persistence_state == "SCAN_COMPLETED_FULL_PERSISTENCE":
    st.success("SCAN_COMPLETED_FULL_PERSISTENCE · hasil scan dan cache terverifikasi penuh di Supabase.")
elif persistence_state == "SCAN_COMPLETED_PARTIAL_PERSISTENCE":
    st.warning("SCAN_COMPLETED_PARTIAL_PERSISTENCE · hasil scan tetap diterbitkan; sebagian data database akan dicari ulang pada scan berikutnya.")
else:
    st.warning("SCAN_COMPLETED_MEMORY_ONLY · hasil tersedia untuk sesi ini; data yang tidak ada di database akan diambil ulang dari provider.")
radar = enrich_dashboard_scores(result["radar"], result.get("frames", {}))
top3 = select_top3(radar, limit=3)
next_leaders = select_next_leaders(radar, limit=20)
real_money_top3 = select_real_money_top3(radar, limit=3)
deep_reviewed_count = int(radar["deep_review_state"].eq("DEEP_REVIEWED").sum())
deep_target_count = len(result.get("shortlist") or [])
deep_scope = str(result.get("deep_review_scope") or "ALL_ELIGIBLE")
metrics = st.columns(9)
metrics[0].metric("Ticker", len(radar))
metrics[1].metric("Deep target", deep_target_count)
metrics[2].metric("Deep reviewed", deep_reviewed_count)
metrics[3].metric("Auto EOD ready", int(radar["emir_decision_state"].eq("EMIR_AUTO_EOD_READY").sum()))
metrics[4].metric("Precise ready", int(radar["emir_decision_state"].eq("EMIR_READY_WITH_PRECISE_TRIGGER").sum()))
metrics[5].metric("Core thesis ready", int(radar["emir_decision_state"].isin(["EMIR_CORE_THESIS_READY_WAIT_IDX_INTEGRITY", "EMIR_THESIS_READY_WAIT_BID_OFFER"]).sum()))
metrics[6].metric("Inventory watch", int(radar["emir_decision_state"].eq("EMIR_WATCH_INVENTORY_COLLECTION").sum()))
metrics[7].metric("Wait", int(radar["emir_decision_state"].isin(["EMIR_WAIT_NARRATIVE", "EMIR_WAIT_MONEY_FLOW"]).sum()))
metrics[8].metric("Reject/Euphoria", int(radar["emir_decision_state"].isin(["EMIR_REJECT_SMART_MONEY_DISTRIBUTION", "EMIR_AVOID_RETAIL_EUPHORIA"]).sum()))
rm_quality_candidates = int(radar.get("real_money_candidate", pd.Series(dtype=bool)).fillna(False).sum())
rm_entry_candidates = int(radar.get("real_money_entry_candidate", pd.Series(dtype=bool)).fillna(False).sum())
rm_ready = int(radar.get("real_money_ready", pd.Series(dtype=bool)).fillna(False).sum())
st.caption(
    f"Real Money Gate: **{rm_quality_candidates} quality-qualified** · **{rm_entry_candidates} timing/actionable manual candidates** · "
    f"**{rm_ready} direct-verified ready**. Proxy/public evidence may qualify for manual confirmation, but never self-authorizes capital."
)

transfer_summary = build_database_transfer_summary(result)
transfer_totals = database_transfer_totals(transfer_summary)
expected_db_rows = int(transfer_totals["expected"])
written_db_rows = int(transfer_totals["written"])
verified_db_rows = int(transfer_totals["verified"])
persistence_cols = st.columns(3)
persistence_cols[0].metric("DB expected rows", expected_db_rows)
persistence_cols[1].metric("DB written rows", written_db_rows)
persistence_cols[2].metric("DB exact verified", verified_db_rows)
research_mem_verify = result.get("research_memory_verification", pd.DataFrame())
if isinstance(research_mem_verify, pd.DataFrame) and not research_mem_verify.empty:
    rv = research_mem_verify.iloc[0]
    rm_state = str(rv.get("state") or "")
    verified_rm = int(rv.get("rows_verified", 0) or 0)
    expected_rm = int(rv.get("rows_expected", 0) or 0)
    if rm_state == "RESEARCH_MEMORY_VERIFIED_EXACT":
        st.caption(f"Research memory: {rm_state} · {verified_rm}/{expected_rm} rows exact-readback. Evidence fundamental/KSEI/narrative terverifikasi tersimpan lintas scan; source cache tetap jalur reuse tercepat.")
    elif verified_rm > 0:
        st.warning(f"Research memory: {rm_state} · {verified_rm}/{expected_rm} rows terverifikasi. Persistence bersifat PARTIAL; source cache masih dapat dipakai, tetapi durable-memory belum exact.")
    else:
        st.warning(f"Research memory: {rm_state} · {verified_rm}/{expected_rm} rows terverifikasi. Durable research-memory BELUM terverifikasi; lihat Database Health untuk detail write/readback sebelum menganggap history aman.")
if transfer_totals["state"] == "DATABASE_RESULT_TRANSFER_VERIFIED":
    st.success("DATABASE_RESULT_TRANSFER_VERIFIED · seluruh hasil final terverifikasi di Supabase.")
elif transfer_totals["state"] == "DATABASE_RESULT_TRANSFER_PARTIAL":
    st.warning("DATABASE_RESULT_TRANSFER_PARTIAL · sebagian hasil sudah masuk database; sisanya tetap tersedia dan akan dicoba ulang.")
else:
    st.warning("DATABASE_RESULT_TRANSFER_NOT_CONFIRMED · hasil sesi tersedia, tetapi transfer database belum terkonfirmasi.")
st.caption(
    f"Scan ID `{result['scan_id']}` · deep scope `{deep_scope}` · commit `{result.get('database_commit_state')}` · "
    f"as-of {result['as_of']} · market regime `{result['market_context'].get('market_regime')}`"
)

tabs = st.tabs([
    "The Next Leader", "Execution Research Top 3", "Emir Radar", "Thesis & Lifecycle", "Inventory & Smart Money", "Structure & Sector",
    "IDX Integrity & Capacity", "Outcome Calibration", "Real Money Gate", "Scenario & Risk", "Chart",
    "Evidence Audit", "Formula Registry", "Database",
])
(
    tab_leader, tab_top3, tab_radar, tab_thesis, tab_flow, tab_structure, tab_integrity, tab_outcomes, tab_real_money, tab_scenario,
    tab_chart, tab_evidence, tab_formula, tab_database,
) = tabs

with tab_leader:
    st.caption("The Next Leader memisahkan kualitas bisnis/future fundamental dari timing entry. Saham markup tetap boleh menjadi leader, tetapi execution dapat WAIT_REACCUMULATION.")
    leader_cols = ["next_leader_rank", "ticker", "company_name", "sector", "next_leader_score", "next_leader_state", "fundamental_conversion_score", "fundamental_data_quality_score", "future_fundamental_score", "future_fundamental_coverage_pct", "future_fundamental_state", "future_fundamental_horizon_state", "future_fundamental_drivers", "fundamental_cashflow_state", "fundamental_cashflow_quality_state", "fundamental_leverage_risk_state", "story_runway_score", "financial_conversion_score", "next_leader_business_momentum_score", "next_leader_business_quality_adjustment", "next_leader_sector_model_state", "next_leader_quality_flags", "sector_rrg_state", "sector_leadership_score", "smart_money_score", "emir_decision_state", "action"]
    if next_leaders.empty:
        fund_scores = pd.to_numeric(radar.get("fundamental_conversion_score", pd.Series(index=radar.index, dtype=float)), errors="coerce")
        fund_cov = pd.to_numeric(radar.get("fundamental_coverage_pct", pd.Series(index=radar.index, dtype=float)), errors="coerce")
        data_q = pd.to_numeric(radar.get("fundamental_data_quality_score", pd.Series(index=radar.index, dtype=float)), errors="coerce")
        st.warning(
            "Belum ada kandidat Next Leader dengan fundamental evidence minimum. "
            f"Diagnostic: fundamental score finite={int(fund_scores.notna().sum())}, coverage>=35={int((fund_cov>=35).sum())}, "
            f"data-quality>=35={int((data_q>=35).sum())}. Next Leader tidak lagi diblokir oleh timing/Real Money state."
        )
    else:
        safe_dataframe(next_leaders[[c for c in leader_cols if c in next_leaders.columns]], width="stretch", hide_index=True)
        st.download_button("Download The Next Leader CSV", next_leaders.to_csv(index=False).encode("utf-8"), "idx_emir_next_leader_v1_9_14.csv", "text/csv")

with tab_top3:
    if st.button("🔄 Scan Ulang dari Dashboard", type="primary", key="top3_dashboard_rescan", width="stretch"):
        start_new_scan_job(auto_continue=True)
    st.caption(
        "Execution Research Top 3 adalah radar timing/flow untuk riset dan tidak identik dengan kandidat modal riil. Final Score = Emir Conviction Score; "
        "flow dan silent accumulation tetap diberi label proxy bila direct broker data tidak tersedia."
    )
    if top3.empty:
        st.warning("Belum ada kandidat yang layak masuk Top 3. Scanner tidak memaksakan saham reject atau data-integrity block.")
    else:
        top3_html = render_top3_dashboard_html(
            top3,
            scan_id=str(result.get("scan_id", "")),
            as_of=result.get("as_of", ""),
            market_regime=str(result.get("market_context", {}).get("market_regime", "")),
        )
        st.markdown(top3_html, unsafe_allow_html=True)
        download_html = "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"></head><body style=\"margin:0;background:#040b12\">" + top3_html + "</body></html>"
        dl1, dl2 = st.columns(2)
        dl1.download_button(
            "Download Top 3 report HTML",
            download_html.encode("utf-8"),
            "idx_emir_execution_research_top3_v1_9_14.html",
            "text/html",
        )
        dl2.download_button(
            "Download Top 3 data CSV",
            top3.to_csv(index=False).encode("utf-8"),
            "idx_emir_execution_research_top3_v1_9_14.csv",
            "text/csv",
        )

with tab_radar:
    columns = [
        "ticker", "company_name", "sector", "emir_decision_state", "action", "emir_lifecycle",
        "market_structure_mode", "next_leader_universe_rank", "next_leader_score", "next_leader_state", "dashboard_universe_rank", "emir_final_score", "emir_conviction_score", "emir_evidence_coverage_pct",
        "broker_inventory_score", "dashboard_flow_score", "dashboard_silent_accum_score", "smart_money_score", "narrative_score", "story_runway_score",
        "sector_rrg_state", "retail_adoption_stage", "idx_integrity_state", "execution_capacity_state",
        "distribution_score", "crowding_score", "real_money_gate_state", "real_money_candidate", "real_money_entry_candidate", "real_money_candidate_score", "real_money_ready", "real_money_block_reasons", "risk_flags",
    ]
    safe_dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)
    st.download_button(
        "Download Emir radar CSV",
        radar.to_csv(index=False).encode("utf-8"),
        "idx_emir_autonomous_radar_v1_9_14.csv",
        "text/csv",
    )

with tab_thesis:
    columns = [
        "ticker", "emir_lifecycle", "narrative_state", "narrative_category", "narrative_event_count",
        "narrative_materiality_score", "top_down_catalyst_score", "industry_translation_score",
        "story_runway_score", "financial_conversion_score", "fundamental_conversion_score", "fundamental_state",
        "fundamental_coverage_pct", "fundamental_statement_availability_pct",
        "fundamental_critical_metric_completeness_pct", "fundamental_official_source_coverage_pct", "fundamental_data_quality_score", "fundamental_score_cap", "fundamental_authority_state", "fundamental_cross_source_state", "fundamental_official_source_url", "fundamental_period_alignment_state", "fundamental_period_freshness_state", "fundamental_absolute_freshness_state", "fundamental_cross_sectional_reference_period", "fundamental_period_lag_days", "fundamental_cashflow_state", "fundamental_cashflow_quality_state", "fundamental_leverage_risk_state",
        "fundamental_growth_consistency_state", "fundamental_growth_consistency_score", "fundamental_ytd_quarters_count",
        "revenue_growth_yoy_pct", "revenue_growth_qoq_pct", "earnings_growth_yoy_pct", "earnings_growth_qoq_pct",
        "revenue_growth_ytd_yoy_pct", "earnings_growth_ytd_yoy_pct",
        "revenue_growth_pct", "earnings_growth_pct", "roe_ttm_pct", "roa_ttm_pct",
        "net_margin_ttm_pct", "operating_margin_ttm_pct", "ocf_conversion_ratio",
        "operating_cash_flow_latest", "free_cash_flow_proxy_latest",
        "operating_cash_flow_ttm", "free_cash_flow_proxy_ttm",
        "interest_bearing_debt_to_equity", "total_liabilities_to_equity", "net_debt_to_equity", "current_ratio", "cash_to_debt_ratio",
        "conversion_path", "issuer_alignment_score",
        "retail_adoption_stage", "thesis_statement", "narrative_latest_title", "what_must_happen_next",
    ]
    safe_dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)

with tab_flow:
    columns = [
        "ticker", "broker_inventory_score", "broker_inventory_coverage_pct", "broker_inventory_shift_state",
        "holder_persistence_score", "inventory_dryness_score", "retail_exit_score",
        "retail_cannibalisation_risk", "fund_like_flow_score", "jumbo_crossing_score", "defended_level",
        "smart_money_score", "absorption_score", "up_value_ratio20_pct", "close_acceptance20_pct",
        "accumulation_days20", "absorption_days20", "distribution_days20", "failed_absorption_days20",
        "beneficial_owner_inference_state",
    ]
    safe_dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)

with tab_structure:
    columns = [
        "ticker", "market_structure_mode", "market_structure_score", "reversal_score",
        "continuation_price_flow_score", "sideways_quality_score", "seller_exhaustion_score",
        "structure_change_score", "fakeout_reclaim_score", "range_compression_score",
        "relative_strength20_pct", "relative_strength60_pct", "relative_strength_momentum_pct",
        "sector", "sector_rrg_state", "sector_leadership_score", "market_regime", "market_context_score",
        "reported_free_float_pct", "effective_free_float_pct", "fake_float_gap_pct", "passive_flow_risk_score",
    ]
    safe_dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)

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
    safe_dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)
    st.caption("IDX integrity is a direct-evidence production gate. HSC, special monitoring/FCA, suspension, serious sanctions, stale evidence, or unresolved corporate-action anomalies cannot be treated as bullish scarcity.")

with tab_outcomes:
    columns = [
        "ticker", "emir_lifecycle", "market_structure_mode", "calibration_mode",
        "outcome_sample_n", "outcome_win_rate_pct", "outcome_median_return_pct",
        "outcome_median_drawdown_pct", "outcome_thesis_invalidation_rate_pct",
        "outcome_calibration_state", "emir_decision_state",
    ]
    safe_dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)
    if isinstance(result.get("outcomes"), pd.DataFrame) and not result["outcomes"].empty:
        st.download_button(
            "Download verified outcome memory",
            result["outcomes"].to_csv(index=False).encode("utf-8"),
            "idx_emir_outcome_memory.csv",
            "text/csv",
        )
    else:
        st.info("Belum ada outcome memory terverifikasi (upload maupun database historis). Calibration tetap NO_OUTCOME_MEMORY dan tidak memengaruhi score.")

with tab_real_money:
    st.caption(
        "GUARDED_REAL_MONEY menerima Yahoo/public statement dan narrative proxy yang cukup terdiversifikasi sebagai evidence untuk MANUAL confirmation. "
        "Official/direct evidence tetap wajib untuk DIRECT_VERIFIED_READY. WAIT_REACCUMULATION/NO_EDGE boleh tetap menjadi quality candidate, tetapi tidak masuk Real Money Candidate Top 3."
    )
    st.subheader("Real Money Candidate Top 3")
    rm_top_cols = [
        "real_money_rank", "ticker", "company_name", "sector", "real_money_candidate_score", "real_money_gate_state",
        "next_leader_universe_rank", "next_leader_score", "emir_decision_state", "real_money_fundamental_evidence_tier", "real_money_narrative_evidence_tier",
        "fundamental_conversion_score", "fundamental_data_quality_score", "liquidity_score", "distribution_score", "market_structure_score",
        "dashboard_silent_accum_score", "dashboard_flow_score", "risk_budget_pct", "guarded_position_cap_after_manual_confirmation_pct",
        "preferred_execution_path", "execution_entry_low", "execution_entry_high", "execution_trigger",
        "execution_stop_loss", "execution_tp1", "execution_tp2", "execution_rr_tp1", "execution_rr_tp2",
        "execution_geometry_state", "real_money_manual_conditions",
    ]
    if real_money_top3.empty:
        st.warning("Belum ada kandidat modal riil yang lolos hard-risk + timing gate. Scanner tidak mempromosikan WAIT/NO_EDGE hanya untuk mengisi Top 3.")
    else:
        safe_dataframe(real_money_top3[[c for c in rm_top_cols if c in real_money_top3.columns]], width="stretch", hide_index=True)
        st.download_button(
            "Download Real Money Candidate Top 3 CSV",
            real_money_top3.to_csv(index=False).encode("utf-8"),
            "idx_emir_real_money_top3_v1_9_14.csv",
            "text/csv",
        )
    st.divider()
    rm_cols = [
        "ticker", "company_name", "sector", "next_leader_universe_rank", "emir_decision_state",
        "real_money_gate_state", "real_money_candidate", "real_money_entry_candidate", "real_money_candidate_score", "real_money_ready", "entry_authorization_state",
        "real_money_block_reasons", "real_money_manual_conditions", "real_money_fundamental_evidence_tier", "real_money_narrative_evidence_tier", "fundamental_authority_state", "fundamental_official_source_coverage_pct", "fundamental_cashflow_quality_state",
        "fundamental_cashflow_state", "fundamental_data_quality_score", "fundamental_leverage_risk_state",
        "market_regime", "sector_rrg_state", "liquidity_score", "distribution_score", "execution_friction_score",
        "risk_budget_pct", "guarded_position_cap_after_manual_confirmation_pct",
        "preferred_execution_path", "execution_entry_low", "execution_entry_high", "execution_trigger",
        "execution_stop_loss", "execution_tp1", "execution_tp2", "execution_rr_tp1", "execution_rr_tp2", "execution_geometry_state",
        "accumulation_entry_low", "accumulation_entry_high", "accumulation_stop_loss", "accumulation_tp1", "accumulation_tp2",
        "breakout_entry", "breakout_stop_loss", "breakout_tp1", "breakout_tp2",
        "real_money_manual_checklist",
    ]
    rm = radar[[c for c in rm_cols if c in radar.columns]].copy()
    if "real_money_entry_candidate" in rm.columns:
        sort_cols = [c for c in ["real_money_ready", "real_money_entry_candidate", "real_money_candidate", "real_money_candidate_score"] if c in rm.columns]
        ascending = [False, False, False, False][:len(sort_cols)]
        rm = rm.sort_values(sort_cols, ascending=ascending, na_position="last")
    safe_dataframe(rm, width="stretch", hide_index=True)
    st.download_button("Download Real Money Gate CSV", rm.to_csv(index=False).encode("utf-8"), "idx_emir_real_money_gate_v1_9_14.csv", "text/csv")

with tab_scenario:
    columns = [
        "ticker", "emir_decision_state", "action", "why_now", "what_must_happen_next", "thesis_invalidation",
        "execution_state", "preferred_execution_path", "execution_entry_low", "execution_entry_high", "execution_trigger",
        "execution_stop_loss", "execution_tp1", "execution_tp2", "execution_rr_tp1", "execution_rr_tp2", "execution_geometry_state",
        "accumulation_entry_low", "accumulation_entry_high", "accumulation_stop_loss", "accumulation_tp1", "accumulation_tp2",
        "breakout_entry", "breakout_stop_loss", "breakout_tp1", "breakout_tp2",
        "trigger_provenance", "precise_trigger_price", "orderbook_trigger_score", "hard_stop_distance_pct",
        "position_cap_pct", "lot", "position_value", "risk_idr", "position_state",
        "max_safe_position_value_idr", "estimated_participation_rate_pct", "slippage_bps_proxy",
        "execution_capacity_state", "real_money_gate_state", "real_money_candidate", "real_money_entry_candidate", "real_money_candidate_score", "real_money_ready", "real_money_block_reasons", "real_money_manual_checklist", "guarded_position_cap_after_manual_confirmation_pct", "entry_authorization_state", "risk_budget_pct", "trim_state",
    ]
    safe_dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)
    st.caption(
        "Level harga adalah scenario plan. Precise entry hanya aktif bila direct bid-offer evidence terverifikasi. "
        "RR adalah scenario geometry independen, bukan formula resmi CAK. Dalam GUARDED_REAL_MONEY, proxy EOD tidak memberi otorisasi entry."
    )

with tab_chart:
    selected = st.selectbox("Ticker", radar["ticker"].tolist())
    row = radar.loc[radar["ticker"].eq(selected)].iloc[0]
    frame = result.get("frames", {}).get(selected, pd.DataFrame())
    if frame.empty and db_config.ready:
        lazy_frames, _ = load_cached_ohlcv_frames(
            db_config, (selected,), period=str(period or "5y"), completed_only=True,
        )
        frame = lazy_frames.get(selected, pd.DataFrame())
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
    safe_dataframe(result["provider_audit"], width="stretch", hide_index=True)
    st.subheader("Autonomous evidence")
    if result.get("autonomous_evidence", pd.DataFrame()).empty:
        st.warning("Autonomous provider tidak menghasilkan evidence.")
    else:
        safe_dataframe(result["autonomous_evidence"], width="stretch", hide_index=True)
        st.download_button(
            "Download autonomous evidence",
            result["autonomous_evidence"].to_csv(index=False).encode("utf-8"),
            "idx_emir_autonomous_evidence_v1_6_3.csv",
            "text/csv",
        )
    st.subheader("Direct evidence overrides")
    if result["direct_evidence"].empty:
        st.info("Tidak ada direct evidence override. Scanner tetap berjalan dengan autonomous public data dan proxy yang diberi label eksplisit.")
    else:
        safe_dataframe(result["direct_evidence"], width="stretch", hide_index=True)
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
        safe_dataframe(result["events"], width="stretch", hide_index=True)
        st.download_button(
            "Download narrative evidence",
            result["events"].to_csv(index=False).encode("utf-8"),
            "idx_emir_narrative_evidence.csv",
            "text/csv",
        )

with tab_formula:
    st.subheader("Public Research Formula Registry")
    registry = formula_registry_frame()
    safe_dataframe(registry, width="stretch", hide_index=True)
    st.download_button(
        "Download formula registry CSV",
        registry.to_csv(index=False).encode("utf-8"),
        "idx_emir_autonomous_formula_registry_v1_9_7.csv",
        "text/csv",
    )
    st.info(
        "EXPLICIT_PUBLIC = dinyatakan secara publik; PUBLIC_SYNTHESIS = sintesis beberapa pernyataan publik; "
        "EMPIRICAL_PROXY = formula numerik independen yang harus diuji; MANUAL_EVIDENCE_REQUIRED = tidak boleh ditebak."
    )

with tab_database:
    st.subheader("Final result transfer reconciliation")
    safe_dataframe(transfer_summary, width="stretch", hide_index=True)
    st.caption(
        "Tabel ini membandingkan jumlah hasil yang seharusnya disimpan, jumlah yang ditulis, dan exact readback per scan_id. "
        "Radar, narrative, provider audit, autonomous evidence, direct evidence, dan outcome memory ditulis terpisah."
    )
    st.subheader("Durable research memory")
    st.caption("Source cache adalah jalur reuse cepat. Research memory menyimpan versi evidence fundamental/KSEI/narrative lintas scan agar history tidak hilang saat cache terbaru diperbarui.")
    safe_dataframe(result.get("research_memory_write_report", pd.DataFrame()), width="stretch", hide_index=True)
    safe_dataframe(result.get("research_memory_verification", pd.DataFrame()), width="stretch", hide_index=True)
    db_health_parts=[]
    for label, frame in (("research_memory", result.get("research_memory_verification")), ("final_transfer", transfer_summary), ("scan_commit", result.get("commit_report"))):
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            local=frame.copy(); local.insert(0,"audit_group",label); db_health_parts.append(local)
    if db_health_parts:
        db_health=pd.concat(db_health_parts,ignore_index=True,sort=False)
        st.download_button("Download database health v1.9.14", db_health.to_csv(index=False).encode("utf-8"), "idx_emir_database_health_v1_9_14.csv", "text/csv")
    st.subheader("Persistent source-cache commit")
    safe_dataframe(result.get("cache_write_report", pd.DataFrame()), width="stretch", hide_index=True)
    st.subheader("Persistent source-cache hash readback")
    safe_dataframe(result.get("cache_verification", pd.DataFrame()), width="stretch", hide_index=True)
    st.subheader("Cache utilization")
    safe_dataframe(result.get("cache_summary", pd.DataFrame()), width="stretch", hide_index=True)
    st.subheader("Database persistence state")
    safe_dataframe(result["commit_report"], width="stretch", hide_index=True)
    st.subheader("Database write report")
    safe_dataframe(result["write_report"], width="stretch", hide_index=True)
    st.subheader("Automatic exact readback")
    verification = result.get("verification")
    if isinstance(verification, pd.DataFrame):
        safe_dataframe(verification, width="stretch", hide_index=True)
        st.download_button(
            "Download database readback audit",
            verification.to_csv(index=False).encode("utf-8"),
            "emir_database_readback_v8_v1_9_7.csv",
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
            st.success("Reverification 100% exact.")
        else:
            st.warning(f"Readback belum penuh: {state}. Hasil scan tetap dapat digunakan; baris yang hilang akan dihitung/diambil ulang pada scan berikutnya.")
        safe_dataframe(recheck, width="stretch", hide_index=True)
