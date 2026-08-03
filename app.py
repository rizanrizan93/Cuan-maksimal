from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Any
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_providers import (
    bare_ticker,
    fetch_many_news,
    fetch_many_ohlcv,
    normalize_ticker,
    parse_universe_csv,
)
from narrative_flow_engine import (
    ENGINE_VERSION,
    aggregate_broker_summary,
    build_public_method_profile,
    calculate_market_features,
    make_scan_id,
    parse_ownership,
    score_narrative_events,
)
from persistence import config_from_mapping, persist_scan, verify_scan


st.set_page_config(page_title="IDX Narrative Flow Scanner", page_icon="🧭", layout="wide")


@st.cache_data(ttl=1800, show_spinner=False)
def cached_fetch_ohlcv(tickers: tuple[str, ...], period: str, max_workers: int):
    return fetch_many_ohlcv(tickers, period=period, max_workers=max_workers)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch_news(tickers: tuple[str, ...], limit: int, max_workers: int):
    return fetch_many_news(tickers, limit=limit, max_workers=max_workers)


def read_optional_csv(uploaded: Any) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(uploaded)
    except Exception as exc:
        st.warning(f"CSV opsional tidak dapat dibaca: {exc}")
        return pd.DataFrame()


def normalize_manual_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "published_at", "title", "summary", "publisher", "url", "source_tier", "materiality_score", "financial_bridge_score", "category"])
    local = frame.copy()
    local.columns = [str(column).strip().lower() for column in local.columns]
    if "ticker" not in local.columns:
        return pd.DataFrame()
    local["ticker"] = local["ticker"].map(normalize_ticker)
    if "published_at" not in local.columns:
        local["published_at"] = local.get("event_date")
    local["published_at"] = pd.to_datetime(local["published_at"], errors="coerce", utc=True)
    rename = {"source_url": "url"}
    local = local.rename(columns={key: value for key, value in rename.items() if key in local.columns and value not in local.columns})
    for column in ("title", "summary", "publisher", "url", "source_tier", "category"):
        if column not in local.columns:
            local[column] = ""
    for column in ("materiality_score", "financial_bridge_score"):
        if column not in local.columns:
            local[column] = np.nan
    return local[["ticker", "published_at", "title", "summary", "publisher", "url", "source_tier", "materiality_score", "financial_bridge_score", "category"]]


def radar_sort(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    local = frame.copy()
    local["production_ready"] = local["production_ready"].fillna(False).astype(bool)
    return local.sort_values(
        ["production_ready", "narrative_flow_conviction_score", "smart_money_score", "liquidity_score"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def make_chart(frame: pd.DataFrame, row: pd.Series) -> go.Figure:
    local = frame.tail(180)
    figure = go.Figure()
    figure.add_trace(go.Candlestick(
        x=local.index,
        open=local["Open"], high=local["High"], low=local["Low"], close=local["Close"],
        name="OHLC",
    ))
    for column, label in (("ema20", "EMA20"), ("ema50", "EMA50"), ("ema200", "EMA200")):
        span = int(column.removeprefix("ema"))
        series = local["Close"].ewm(span=span, adjust=False).mean()
        figure.add_trace(go.Scatter(x=local.index, y=series, mode="lines", name=label))
    levels = {
        "Entry Low": row.get("entry_low"),
        "Entry High": row.get("entry_high"),
        "Trigger": row.get("trigger"),
        "Stop": row.get("stop_loss"),
        "TP1": row.get("tp1"),
        "TP2": row.get("tp2"),
    }
    for label, value in levels.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            figure.add_hline(y=number, annotation_text=label)
    figure.update_layout(height=620, xaxis_rangeslider_visible=False, title=str(row.get("ticker")))
    return figure


def position_builder(row: pd.Series, capital: float, risk_pct: float) -> dict[str, Any]:
    entry_low = pd.to_numeric(pd.Series([row.get("entry_low")]), errors="coerce").iloc[0]
    entry_high = pd.to_numeric(pd.Series([row.get("entry_high")]), errors="coerce").iloc[0]
    stop = pd.to_numeric(pd.Series([row.get("stop_loss")]), errors="coerce").iloc[0]
    cap = pd.to_numeric(pd.Series([row.get("position_cap_pct")]), errors="coerce").iloc[0]
    if not all(np.isfinite(value) for value in (entry_low, entry_high, stop, cap)) or cap <= 0:
        return {"lot": 0, "position_value": 0.0, "risk_idr": 0.0, "state": "NO_EXECUTABLE_POSITION"}
    entry = (entry_low + entry_high) / 2
    per_share_risk = entry - stop
    if per_share_risk <= 0:
        return {"lot": 0, "position_value": 0.0, "risk_idr": 0.0, "state": "INVALID_RISK"}
    risk_budget = capital * risk_pct / 100
    max_value = capital * cap / 100
    shares_by_risk = np.floor(risk_budget / per_share_risk / 100) * 100
    shares_by_cap = np.floor(max_value / entry / 100) * 100
    shares = max(0, min(shares_by_risk, shares_by_cap))
    return {
        "lot": int(shares / 100),
        "position_value": round(shares * entry, 2),
        "risk_idr": round(shares * per_share_risk, 2),
        "state": "POSITION_READY" if shares >= 100 else "CAPITAL_OR_RISK_TOO_SMALL",
    }


def render_methodology() -> None:
    with st.expander("Metodologi public-framework dan batasan", expanded=False):
        st.markdown(
            """
Scanner ini adalah rekonstruksi clean-room dari prinsip yang disampaikan secara publik: memahami universe terbatas, membaca lifecycle narasi, menguji money flow/smart-money behavior, menghubungkan fundamental–teknikal–flow, dan mengeksekusi dengan skenario serta risiko terukur.

**Bukan** produk, afiliasi, formula proprietary, atau track record CAK Investment Club/Emir Parengkuan. OHLCV tidak dapat mengidentifikasi beneficial owner. Broker code tidak identik dengan satu pemilik. Evidence kosong tetap kosong dan tidak diisi angka netral seolah-olah tersedia.
            """
        )


st.title("IDX Narrative Flow Scanner")
st.caption(f"Versi {ENGINE_VERSION} · standalone public-framework reconstruction · narrative play + money flow + smart-money behavior + risk scenario")
render_methodology()

with st.sidebar:
    st.header("Scan Configuration")
    universe_file = st.file_uploader("Upload CSV universe", type=["csv"], key="universe")
    scan_mode = st.selectbox(
        "Mode",
        ["HYBRID_400_TO_DEEP", "FAST_DISCOVERY", "DEEP_REVIEW"],
        help="HYBRID: OHLCV seluruh universe lalu narrative review hanya shortlist. FAST: tanpa online news. DEEP: review semua ticker yang diunggah, disarankan ≤100.",
    )
    period = st.selectbox("OHLCV history", ["3y", "5y"], index=1)
    workers = st.slider("Concurrent OHLCV workers", 2, 16, 8)
    deep_limit = st.slider("Maksimum deep narrative review", 5, 100, 30)
    news_per_ticker = st.slider("News per deep ticker", 2, 12, 6)
    use_online_news = st.checkbox("Ambil public Yahoo news", value=True, disabled=scan_mode == "FAST_DISCOVERY")
    capital = st.number_input("Modal simulasi (IDR)", min_value=100_000.0, value=5_000_000.0, step=100_000.0)
    risk_pct = st.slider("Risk budget per idea (%)", 0.25, 2.0, 1.0, 0.25)
    st.divider()
    broker_file = st.file_uploader("Broker summary CSV (opsional)", type=["csv"], key="broker")
    narrative_file = st.file_uploader("Narrative events CSV (opsional)", type=["csv"], key="narrative")
    ownership_file = st.file_uploader("Ownership CSV (opsional)", type=["csv"], key="ownership")
    run_scan = st.button("Jalankan scanner", type="primary", use_container_width=True)

if universe_file is None:
    st.info("Upload CSV ticker. File contoh tersedia di paket: `sample_universe_100.csv`.")
    st.stop()

try:
    tickers = parse_universe_csv(universe_file)
except Exception as exc:
    st.error(f"Universe CSV gagal dibaca: {exc}")
    st.stop()

if not tickers:
    st.error("Tidak ada ticker valid pada CSV.")
    st.stop()

st.write(f"Universe terdeteksi: **{len(tickers)} ticker unik**")
if scan_mode == "DEEP_REVIEW" and len(tickers) > 100:
    st.warning("DEEP_REVIEW untuk >100 ticker berpotensi lambat. Gunakan HYBRID untuk universe besar.")

if run_scan:
    progress = st.progress(0, text="Mengambil benchmark IHSG...")
    benchmark_frames, benchmark_audit = cached_fetch_ohlcv(("^JKSE",), period, min(workers, 4))
    benchmark = benchmark_frames.get("^JKSE", pd.DataFrame())
    progress.progress(5, text=f"Mengambil OHLCV {len(tickers)} ticker...")
    frames, provider_audit = cached_fetch_ohlcv(tuple(tickers), period, workers)
    progress.progress(45, text="Menghitung radar money flow dan market structure...")
    fast_rows: list[dict[str, Any]] = []
    for ticker in tickers:
        frame = frames.get(ticker, pd.DataFrame())
        features = calculate_market_features(frame, benchmark)
        fast_rows.append({"ticker": ticker, **features})
    fast = pd.DataFrame(fast_rows)
    eligible = fast[fast.get("feature_state", "") == "OK"].copy() if not fast.empty else pd.DataFrame()
    eligible["fast_discovery_score"] = (
        0.48 * pd.to_numeric(eligible.get("smart_money_score"), errors="coerce")
        + 0.27 * pd.to_numeric(eligible.get("trend_score"), errors="coerce")
        + 0.15 * pd.to_numeric(eligible.get("liquidity_score"), errors="coerce")
        + 0.10 * (100 - pd.to_numeric(eligible.get("distribution_score"), errors="coerce"))
    )
    shortlist_n = len(eligible) if scan_mode == "DEEP_REVIEW" else min(deep_limit, len(eligible))
    shortlist = eligible.sort_values("fast_discovery_score", ascending=False).head(shortlist_n)["ticker"].tolist()
    progress.progress(58, text=f"Deep review {len(shortlist)} ticker...")
    manual_events = normalize_manual_events(read_optional_csv(narrative_file))
    online_events = pd.DataFrame()
    if use_online_news and scan_mode != "FAST_DISCOVERY" and shortlist:
        online_events = cached_fetch_news(tuple(shortlist), news_per_ticker, min(workers, 8))
    all_events = pd.concat([manual_events, online_events], ignore_index=True, sort=False) if not manual_events.empty or not online_events.empty else pd.DataFrame()
    broker_map = aggregate_broker_summary(read_optional_csv(broker_file))
    ownership_map = parse_ownership(read_optional_csv(ownership_file))
    progress.progress(72, text="Membangun narrative-flow lifecycle dan execution scenarios...")
    output_rows: list[dict[str, Any]] = []
    for _, fast_row in fast.iterrows():
        ticker = fast_row["ticker"]
        events = all_events[all_events["ticker"] == ticker] if not all_events.empty else pd.DataFrame()
        narrative = score_narrative_events(events)
        profile = build_public_method_profile(
            ticker=ticker,
            features=fast_row.to_dict(),
            narrative=narrative,
            broker=broker_map.get(ticker),
            ownership=ownership_map.get(ticker),
        )
        allocation = position_builder(pd.Series(profile), capital, risk_pct)
        output_rows.append({**profile, **allocation})
    radar = radar_sort(pd.DataFrame(output_rows))
    as_of = pd.Timestamp.now(tz="Asia/Jakarta")
    scan_id = make_scan_id(as_of, tickers)
    progress.progress(85, text="Menyimpan hasil opsional ke Supabase...")
    try:
        secret_mapping = st.secrets
    except Exception:
        secret_mapping = {}
    database_config = config_from_mapping(secret_mapping)
    write_report = persist_scan(
        database_config,
        scan_id=scan_id,
        as_of=as_of,
        radar=radar,
        events=all_events,
        mode=scan_mode,
    )
    st.session_state["nf_scan"] = {
        "radar": radar,
        "frames": frames,
        "events": all_events,
        "provider_audit": provider_audit,
        "benchmark_audit": benchmark_audit,
        "write_report": write_report,
        "scan_id": scan_id,
        "as_of": as_of,
        "database_ready": database_config.ready,
        "expected_events": len(all_events),
    }
    progress.progress(100, text="Selesai")

result = st.session_state.get("nf_scan")
if not result:
    st.stop()

radar = result["radar"]
summary_cols = st.columns(5)
summary_cols[0].metric("Ticker", len(radar))
summary_cols[1].metric("Ready", int(radar.get("production_ready", pd.Series(False)).fillna(False).sum()))
summary_cols[2].metric("Watch", int((radar.get("public_method_state") == "PUBLIC_FRAMEWORK_WATCH").sum()))
summary_cols[3].metric("Evidence Pending", int((radar.get("public_method_state") == "PUBLIC_FRAMEWORK_EVIDENCE_PENDING").sum()))
summary_cols[4].metric("Reject", int((radar.get("public_method_state") == "PUBLIC_FRAMEWORK_REJECT").sum()))
st.caption(f"Scan ID: `{result['scan_id']}` · as-of {result['as_of']}")

tab_radar, tab_lifecycle, tab_plan, tab_chart, tab_evidence, tab_database = st.tabs([
    "Narrative Flow Radar", "Lifecycle", "Execution Scenarios", "Chart Review", "Evidence & Provider", "Database",
])

with tab_radar:
    columns = [
        "ticker", "public_method_state", "action", "narrative_flow_lifecycle",
        "narrative_flow_conviction_score", "narrative_flow_coverage_pct",
        "narrative_score", "smart_money_score", "trend_score", "liquidity_score",
        "distribution_score", "crowding_score", "price_stage", "position_cap_pct", "risk_flags",
    ]
    st.dataframe(radar[[column for column in columns if column in radar.columns]], width="stretch", hide_index=True)
    st.download_button("Download full radar CSV", radar.to_csv(index=False).encode("utf-8"), "idx_narrative_flow_radar.csv", "text/csv")

with tab_lifecycle:
    lifecycle_cols = [
        "ticker", "narrative_flow_lifecycle", "narrative_state", "narrative_category", "narrative_event_count",
        "narrative_latest_title", "smart_money_score", "accumulation_days20", "absorption_days20",
        "distribution_days20", "failed_absorption_days20", "broker_summary_score",
        "broker_summary_provenance_state", "ownership_score", "free_float_pct",
    ]
    st.dataframe(radar[[column for column in lifecycle_cols if column in radar.columns]], width="stretch", hide_index=True)

with tab_plan:
    plan_cols = [
        "ticker", "action", "execution_state", "entry_low", "entry_high", "trigger", "stop_loss", "tp1", "tp2",
        "rr_tp1", "rr_tp2", "position_cap_pct", "lot", "position_value", "risk_idr", "state",
    ]
    st.dataframe(radar[[column for column in plan_cols if column in radar.columns]], width="stretch", hide_index=True)
    st.caption("Execution plan hanya aktif bila narrative–flow convergence, coverage, likuiditas, dan distribution gate lulus.")

with tab_chart:
    choices = radar["ticker"].tolist()
    selected = st.selectbox("Ticker", choices)
    selected_row = radar.loc[radar["ticker"] == selected].iloc[0]
    selected_frame = result["frames"].get(selected, pd.DataFrame())
    if selected_frame.empty:
        st.warning("OHLCV tidak tersedia.")
    else:
        st.plotly_chart(make_chart(selected_frame, selected_row), use_container_width=True)
        st.json({key: selected_row.get(key) for key in ["public_method_state", "action", "narrative_flow_lifecycle", "risk_flags", "narrative_latest_title"]})

with tab_evidence:
    st.subheader("Provider audit")
    st.dataframe(result["provider_audit"], width="stretch", hide_index=True)
    st.subheader("Narrative events")
    if result["events"].empty:
        st.info("Tidak ada event publik/manual yang dikoleksi pada run ini.")
    else:
        st.dataframe(result["events"], width="stretch", hide_index=True)
        st.download_button("Download narrative events", result["events"].to_csv(index=False).encode("utf-8"), "idx_narrative_events.csv", "text/csv")

with tab_database:
    st.dataframe(result["write_report"], width="stretch", hide_index=True)
    if result["database_ready"]:
        if st.button("Verifikasi upload scan ini"):
            try:
                secret_mapping = st.secrets
            except Exception:
                secret_mapping = {}
            database_config = config_from_mapping(secret_mapping)
            verification = verify_scan(
                database_config,
                scan_id=result["scan_id"],
                expected_radar=len(radar),
                expected_events=result["expected_events"],
            )
            st.session_state["nf_verification"] = verification
        verification = st.session_state.get("nf_verification")
        if isinstance(verification, pd.DataFrame):
            state = str(verification.iloc[0]["state"])
            if state == "VERIFIED_ALL_TABLES":
                st.success("Upload database terverifikasi 100%.")
            else:
                st.warning(f"Readback belum lengkap: {state}")
            st.dataframe(verification, width="stretch", hide_index=True)
            st.download_button("Download database readback audit", verification.to_csv(index=False).encode("utf-8"), "narrative_flow_database_readback.csv", "text/csv")
    else:
        st.info("Database opsional belum aktif. Jalankan migration_v1.sql lalu isi Streamlit Secrets.")
