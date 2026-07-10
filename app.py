import concurrent.futures as cf
import os
import json
import re
import tempfile
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema, hilbert, periodogram

# =========================================================
# IDX / IHSG DUAL TAB SCANNER - FINAL VERSION
# Tab 1: Market Structure Top 20 + reversal signals
# Tab 2: Institutional Forward Score with sub-tabs + entry plan / benchmark / time analysis
# =========================================================

st.set_page_config(page_title="IDX Dual Tab Scanner", layout="wide", initial_sidebar_state="collapsed")
st.title("📊 IDX Dual Tab Scanner")
st.caption(
    "Global watchlist untuk ranking cepat, lalu deep dive untuk bedah detail per ticker dengan institutional forward score, entry plan, dan time analysis."
)
st.markdown("---")

st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.8rem;
            padding-left: 0.7rem;
            padding-right: 0.7rem;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.9rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Prefer remote ledger storage when secrets are present.
def _bootstrap_portfolio_db_url() -> None:
    try:
        if os.environ.get("PORTFOLIO_DB_URL", "").strip():
            return
        for key in ("PORTFOLIO_DB_URL", "SUPABASE_DB_URL", "DATABASE_URL"):
            try:
                secret_value = st.secrets.get(key, "")
            except Exception:
                secret_value = ""
            if secret_value:
                os.environ["PORTFOLIO_DB_URL"] = str(secret_value).strip()
                break
    except Exception:
        pass


_bootstrap_portfolio_db_url()

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("🎯 Universe Source")
universe_mode = st.sidebar.radio(
    "Pilih sumber universe",
    ["Paste tickers", "Upload CSV", "Local file midcap_universe.csv"],
    index=0,
)

paste_text = ""
uploaded_file = None
if universe_mode == "Paste tickers":
    paste_text = st.sidebar.text_area(
        "Paste tickers (satu per baris / dipisah koma)",
        value="BMRI\nBBCA\nTLKM\nASII",
        height=140,
    )
elif universe_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV universe", type=["csv"])
else:
    st.sidebar.info("Mode ini akan membaca file `midcap_universe.csv` dari folder aplikasi.")

st.sidebar.markdown("---")
st.sidebar.header("🧭 Scan Settings")
months = st.sidebar.slider("Periode data historis (bulan)", 12, 60, 24)
min_price = st.sidebar.number_input("Min harga (Rp)", value=200.0, step=10.0)
max_price = st.sidebar.number_input("Max harga (Rp)", value=25000.0, step=500.0)
min_avg_volume = st.sidebar.number_input("Min rata-rata volume 20D", value=150000, step=50000)
min_history_bars = st.sidebar.slider("Min candle valid", 60, 240, 100)
scan_limit = st.sidebar.slider("Maks tickers per scan", 20, 500, 200, step=10)
mobile_mode = st.sidebar.checkbox("HP Compact Mode", value=True)


st.sidebar.markdown("---")
st.sidebar.header("🚀 Execution")
max_workers = st.sidebar.slider("Max parallel workers", 2, 12, 6)
ranking_sort_mode = st.sidebar.selectbox("Ranking order", ["Descending", "Ascending"], index=0)
run_global_scan = st.sidebar.button("Run global scan", type="primary")

GLOBAL_MODE = "Conservative"  # Profit-only mode: prioritize precision over signal count

# =========================================================
# Utilities
# =========================================================

from data_engine import *
from fundamental_analyst import *
from technical_analyst import *
from technical_analyst import _safe_float as _safe_float
from catalyst_nlp import *
from idx_edge_lab import StrategyParams, summarize_walk_forward, walk_forward_test
from research_io import read_uploaded_ohlcv_bundle, read_uploaded_research_bundle, save_research_bundle
from ohlcv_downloader import download_batch_idx_ohlcv, extract_universe_tickers, load_universe_from_csv as load_universe_df_from_csv, save_batch_bundle
from data_engine import load_universe_from_csv as load_universe_tickers_from_csv
import portfolio_engine as pe




def _plan_value(*values):
    """Return first finite numeric value among candidates."""
    for value in values:
        try:
            if pd.notna(value):
                return value
        except Exception:
            continue
    return np.nan

def _safe_text(value) -> str:
    try:
        text = str(value or "").strip()
        return text
    except Exception:
        return ""




def _effective_entry_plan(stock_res: dict) -> dict:
    """Return the canonical entry plan, falling back to candidate fields when needed."""
    stock_res = stock_res or {}
    plan = stock_res.get("entry_plan", {}) if isinstance(stock_res.get("entry_plan", {}), dict) else {}
    if plan:
        eta = _estimate_entry_eta(stock_res, plan)
        if eta:
            merged = dict(plan)
            merged.update(eta)
            return merged
        return plan
    fallback = {
        "entry_zone_low": stock_res.get("candidate_entry_zone_low", stock_res.get("entry_zone_low", np.nan)),
        "entry_zone_high": stock_res.get("candidate_entry_zone_high", stock_res.get("entry_zone_high", np.nan)),
        "entry_price_plan": stock_res.get("candidate_entry_price", stock_res.get("entry_price_plan", np.nan)),
        "entry_trigger": stock_res.get("entry_trigger", "Candidate_Fallback"),
        "projected_first_leg": stock_res.get("projected_first_leg", stock_res.get("entry_projection_first_leg", "n/a")),
        "projected_rebound_leg": stock_res.get("projected_rebound_leg", stock_res.get("entry_projection_rebound_leg", "n/a")),
        "entry_zone_role": stock_res.get("entry_zone_role", "n/a"),
        "entry_zone_label": stock_res.get("entry_zone_label", "n/a"),
        "entry_projection_summary": stock_res.get("entry_projection_summary", "n/a"),
        "stop_loss_plan": stock_res.get("candidate_stop_price", stock_res.get("stop_loss_plan", stock_res.get("stop_price", np.nan))),
        "target_1": stock_res.get("candidate_target_1", stock_res.get("target_1", np.nan)),
        "target_2": stock_res.get("candidate_target_2", stock_res.get("target_2", np.nan)),
        "risk_per_share": stock_res.get("risk_per_share", np.nan),
        "risk_reward_1": stock_res.get("candidate_risk_reward_1", stock_res.get("risk_reward_1", np.nan)),
        "risk_reward_2": stock_res.get("candidate_risk_reward_2", stock_res.get("risk_reward_2", np.nan)),
        "upside_to_t1_pct": stock_res.get("upside_to_t1_pct", np.nan),
        "upside_to_t2_pct": stock_res.get("upside_to_t2_pct", np.nan),
        "plan_reason": stock_res.get("plan_reason", stock_res.get("execution_status_reason", "Candidate fallback")),
        "setup_kind": stock_res.get("setup_kind", "Candidate"),
        "execution_status": stock_res.get("execution_status", stock_res.get("entry_candidate_label", "WATCHLIST_ENTRY")),
        "execution_status_reason": stock_res.get("execution_status_reason", stock_res.get("tradeability_gate_reason", "n/a")),
    }
    return fallback




def _estimate_entry_eta(stock_res: dict, plan: dict | None = None) -> dict:
    """Approximate when the setup may be touched.

    Assumes daily bars: one bar is treated as roughly one trading day.
    """
    try:
        base = stock_res if isinstance(stock_res, dict) else {}
        plan = plan if isinstance(plan, dict) else {}

        kind = str(plan.get("setup_kind", base.get("setup_kind", "")) or "").strip().upper()
        fill = _safe_float(
            plan.get("setup_fill_probability", base.get("setup_fill_probability", np.nan)),
            np.nan,
        )
        dist_atr = _safe_float(
            plan.get("setup_distance_to_entry_atr", base.get("setup_distance_to_entry_atr", np.nan)),
            np.nan,
        )
        age_bars = _safe_float(
            plan.get("setup_age_bars", base.get("setup_age_bars", np.nan)),
            np.nan,
        )
        fresh = bool(plan.get("setup_fresh", base.get("setup_fresh", False)))

        if not np.isfinite(fill) and not np.isfinite(dist_atr):
            return {}

        if not np.isfinite(fill):
            fill = 60.0
        if not np.isfinite(dist_atr):
            dist_atr = 1.0

        kind_factor = {
            "SNIPER": 0.72,
            "UNICORN": 0.88,
            "PULLBACK": 1.08,
            "BREAKOUT": 1.00,
            "REVERSAL": 1.18,
        }.get(kind, 1.00)

        fill_factor = float(np.clip(2.05 - (fill / 100.0) * 1.15, 0.58, 1.80))
        dist_factor = float(np.clip(0.85 + dist_atr * 0.72, 0.65, 3.25))
        age_factor = float(np.clip(1.0 + max(0.0, age_bars if np.isfinite(age_bars) else 0.0) / 18.0, 1.0, 2.8))
        freshness_factor = 0.88 if fresh else 1.10

        eta_bars = float(np.clip(dist_factor * fill_factor * kind_factor * age_factor * freshness_factor, 0.5, 30.0))
        eta_days = eta_bars

        return {
            "entry_eta_bars": eta_bars,
            "entry_eta_days": eta_days,
            "entry_eta_label": f"~{eta_days:.1f} hari" if np.isfinite(eta_days) else "n/a",
            "entry_eta_range_days": (
                float(np.clip(eta_days * 0.70, 0.5, 30.0)),
                float(np.clip(eta_days * 1.40, 0.5, 30.0)),
            ),
        }
    except Exception:
        return {}

def _setup_entry_plan_bundle(stock_res: dict, setup_kind: str) -> dict:
    """Build a setup-specific entry plan without letting other setups interfere."""
    try:
        kind = str(setup_kind or "").strip().upper()
        if kind not in {"BREAKOUT", "PULLBACK", "UNICORN", "SNIPER", "REVERSAL"}:
            return {}

        base = dict(stock_res or {})
        neutral_state = {
            "valid": False,
            "status": "INVALID",
            "reason": "scope_neutralized",
            "support_anchor": np.nan,
            "resistance_anchor": np.nan,
            "stop_price": np.nan,
            "invalidation_level": np.nan,
        }
        base["breakout_confirmed"] = False
        base["breakout_setup_valid"] = False
        base["pullback_continuation_valid"] = False
        base["reversal_accumulation_valid"] = False
        base["unicorn_setup_valid"] = False
        base["unicorn_sniper_valid"] = False
        base["pullback_continuation_state"] = neutral_state.copy()
        base["reversal_accumulation_state"] = neutral_state.copy()

        if kind == "BREAKOUT":
            base["breakout_confirmed"] = bool(stock_res.get("breakout_confirmed", False) or stock_res.get("breakout_setup_valid", False))
            base["breakout_setup_valid"] = bool(stock_res.get("breakout_setup_valid", False) or base["breakout_confirmed"])
        elif kind == "PULLBACK":
            base["pullback_continuation_valid"] = bool(stock_res.get("pullback_continuation_valid", False))
            if isinstance(stock_res.get("pullback_continuation_state"), dict):
                base["pullback_continuation_state"] = dict(stock_res.get("pullback_continuation_state", {}))
        elif kind == "UNICORN":
            base["unicorn_setup_valid"] = bool(stock_res.get("unicorn_setup_valid", False))
            base["unicorn_sniper_valid"] = bool(stock_res.get("unicorn_sniper_valid", False))
        elif kind == "SNIPER":
            # Sniper is a distinct label, but it still uses the Unicorn structural family underneath.
            base["unicorn_setup_valid"] = bool(stock_res.get("unicorn_setup_valid", False) or stock_res.get("unicorn_sniper_valid", False))
            base["unicorn_sniper_valid"] = bool(stock_res.get("unicorn_sniper_valid", False))
        elif kind == "REVERSAL":
            base["reversal_accumulation_valid"] = bool(stock_res.get("reversal_accumulation_valid", False))
            if isinstance(stock_res.get("reversal_accumulation_state"), dict):
                base["reversal_accumulation_state"] = dict(stock_res.get("reversal_accumulation_state", {}))

        plan = build_entry_plan(base)
        return plan if isinstance(plan, dict) else {}
    except Exception:
        return {}

def _build_stockbit_ticket(row: dict | pd.Series) -> str:
    """Create a concise manual-execution ticket for Stockbit mobile use."""
    try:
        if isinstance(row, pd.Series):
            row = row.to_dict()
        row = row or {}
        ticker = _safe_text(row.get("Ticker") or row.get("symbol") or "n/a")
        decision = _safe_text(row.get("Decision") or "n/a")
        setup = _safe_text(row.get("Lifecycle") or row.get("BreakoutStatus") or row.get("Setup") or "n/a")
        validity = _safe_text(row.get("Validity") or "n/a")
        trade_gate = _safe_text(row.get("TradeGate") or "n/a")
        next_action = _safe_text(row.get("NextAction") or "WAIT")
        score = row.get("Score", np.nan)
        ifs = row.get("IFS", np.nan)
        tradeability = row.get("Tradeability", np.nan)
        entry = row.get("Entry", row.get("ProjectedEntry", np.nan))
        stop = row.get("Stop", row.get("ProjectedStop", np.nan))
        tp1 = row.get("TP1", np.nan)
        tp2 = row.get("TP2", np.nan)
        rr1 = row.get("RR1", np.nan)
        rr2 = row.get("RR2", np.nan)
        reason = _safe_text(row.get("ValidityReason") or row.get("TradeGateReason") or "")
        notes = _safe_text(row.get("Notes") or "")
        value20d = row.get("Value20D_Bn", np.nan)
        spread = row.get("SpreadPct", np.nan)
        gap = row.get("GapPct", np.nan)
        parts = [
            f"STOCKBIT TRADE TICKET - {ticker}",
            f"Decision      : {decision}",
            f"Setup         : {setup}",
            f"Validity      : {validity}",
            f"Trade Gate    : {trade_gate}",
            f"Next Action   : {next_action}",
            f"Score / IFS   : {score:.2f} / {ifs:.2f}" if pd.notna(score) and pd.notna(ifs) else f"Score / IFS   : n/a",
            f"Tradeability  : {tradeability:.2f}" if pd.notna(tradeability) else "Tradeability  : n/a",
            f"Value 20D     : Rp {value20d:.2f}B" if pd.notna(value20d) else "Value 20D     : n/a",
            f"Spread / Gap  : {spread:.2f}% / {gap:.2f}%" if pd.notna(spread) and pd.notna(gap) else "Spread / Gap  : n/a",
            f"Entry         : Rp {entry:,.0f}" if pd.notna(entry) else "Entry         : n/a",
            f"Stop          : Rp {stop:,.0f}" if pd.notna(stop) else "Stop          : n/a",
            f"TP1 / TP2     : Rp {tp1:,.0f} / Rp {tp2:,.0f}" if pd.notna(tp1) and pd.notna(tp2) else "TP1 / TP2     : n/a",
            f"RR1 / RR2     : {rr1:.2f} / {rr2:.2f}" if pd.notna(rr1) and pd.notna(rr2) else "RR1 / RR2     : n/a",
        ]
        if reason:
            parts.append(f"Reason        : {reason}")
        if notes:
            parts.append(f"Notes         : {notes}")
        parts.append("")
        parts.append("Checklist:")
        parts.append("- Cek ulang harga terakhir di Stockbit sebelum entry.")
        parts.append("- Entry hanya jika setup masih valid dan trade gate OK.")
        parts.append("- Batalkan jika harga sudah tembus stop / setup expired.")
        return "\n".join(parts)
    except Exception as exc:
        return f"STOCKBIT TRADE TICKET - ERROR\n{type(exc).__name__}: {exc}"


def _build_setup_summary(row: dict | pd.Series) -> str:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    elif hasattr(row, "_asdict"):
        row = row._asdict()
    elif not isinstance(row, dict):
        row = dict(getattr(row, "__dict__", {}))
    row = row or {}
    parts = [
        f"**Market Struct:** `{row.get('MarketStruct', 'n/a')}`  \
",
        f"**Trend/Momentum:** `{row.get('Trend', 'n/a')}` / `{row.get('Momentum', 'n/a')}`  \
",
        f"**Cycle:** `{row.get('Cycle', 'n/a')}` bars | Rel `{row.get('CycleRel', 'n/a')}`  \
",
        f"**Risk:** `{row.get('Risk', 'n/a')}`  \
",
        f"**TP1/TP2:** `{row.get('TP1', 'n/a')}` / `{row.get('TP2', 'n/a')}`  \
",
        f"**RR1/RR2:** `{row.get('RR1', 'n/a')}` / `{row.get('RR2', 'n/a')}`  \
",
        f"**Smart Money:** `{row.get('SmartMoney', 'n/a')}`  \
",
        f"**Phase:** `{row.get('Phase', 'n/a')}`",
    ]
    return "".join(parts)

APP_TMP_DIR = Path(tempfile.gettempdir())
DECISION_CACHE_PATH = APP_TMP_DIR / "scanner_decision_cache.json"
RESEARCH_OUTPUT_DIR = APP_TMP_DIR / "research_outputs"


def _load_decision_cache() -> dict:
    try:
        if DECISION_CACHE_PATH.exists():
            data = json.loads(DECISION_CACHE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _save_decision_cache(cache: dict) -> None:
    try:
        DECISION_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _load_macro_context(benchmark_symbol: str, months_: int) -> dict:
    """Load benchmark data lazily so the dashboard can render immediately."""
    try:
        bench_df = load_ticker_data(benchmark_symbol, months_)
        if bench_df is None or bench_df.empty:
            return {
                "benchmark_symbol": benchmark_symbol,
                "benchmark_df": pd.DataFrame(),
                "market_regime": "SIDEWAYS",
                "market_regime_confidence": 0.0,
                "market_regime_reason": "benchmark_empty",
                "macro_gate_ok": False,
                "macro_gate_reason": "benchmark_empty",
                "macro_score": np.nan,
                "macro_multiplier": 1.0,
            }
        ctx = build_macro_liquidity_gate(bench_df.copy(), benchmark_symbol)
        if not isinstance(ctx, dict):
            ctx = {}
        ctx.setdefault("benchmark_symbol", benchmark_symbol)
        ctx.setdefault("benchmark_df", bench_df)
        return ctx
    except Exception as exc:
        return {
            "benchmark_symbol": benchmark_symbol,
            "benchmark_df": pd.DataFrame(),
            "market_regime": "SIDEWAYS",
            "market_regime_confidence": 0.0,
            "market_regime_reason": f"benchmark_load_failed:{type(exc).__name__}",
            "macro_gate_ok": False,
            "macro_gate_reason": f"benchmark_load_failed:{type(exc).__name__}",
            "macro_score": np.nan,
            "macro_multiplier": 1.0,
        }


def _apply_decision_hysteresis(symbol: str, result: dict, cache: dict) -> dict:
    """Keep buy-side decisions stable unless deterioration is broad and persistent."""
    sym = str(symbol or "").upper().strip()
    if not sym or not isinstance(result, dict):
        return result

    prev = cache.get(sym, {}) if isinstance(cache, dict) else {}
    prev_decision = str(prev.get("decision", "")).upper().strip()
    curr_decision = str(result.get("decision", "")).upper().strip()

    prev_rank = _decision_rank(prev_decision)
    curr_rank = _decision_rank(curr_decision)

    curr_score = float(result.get("score", np.nan)) if pd.notna(result.get("score", np.nan)) else np.nan
    prev_score = float(prev.get("score", np.nan)) if pd.notna(prev.get("score", np.nan)) else np.nan
    curr_ifs = float(result.get("ifs_score", np.nan)) if pd.notna(result.get("ifs_score", np.nan)) else np.nan
    prev_ifs = float(prev.get("ifs_score", np.nan)) if pd.notna(prev.get("ifs_score", np.nan)) else np.nan

    curr_market_struct = float(result.get("market_structure_score", np.nan)) if pd.notna(result.get("market_structure_score", np.nan)) else np.nan
    prev_market_struct = float(prev.get("market_structure_score", np.nan)) if pd.notna(prev.get("market_structure_score", np.nan)) else np.nan
    curr_smart_money = float(result.get("smart_money_score", np.nan)) if pd.notna(result.get("smart_money_score", np.nan)) else np.nan
    prev_smart_money = float(prev.get("smart_money_score", np.nan)) if pd.notna(prev.get("smart_money_score", np.nan)) else np.nan

    buyish_prev = prev_rank >= _decision_rank("BUY")
    downgrade = curr_rank < prev_rank

    score_drop = (prev_score - curr_score) if (np.isfinite(prev_score) and np.isfinite(curr_score)) else np.nan
    ifs_drop = (prev_ifs - curr_ifs) if (np.isfinite(prev_ifs) and np.isfinite(curr_ifs)) else np.nan
    struct_drop = (prev_market_struct - curr_market_struct) if (np.isfinite(prev_market_struct) and np.isfinite(curr_market_struct)) else np.nan
    smart_drop = (prev_smart_money - curr_smart_money) if (np.isfinite(prev_smart_money) and np.isfinite(curr_smart_money)) else np.nan

    strong_break = bool(
        curr_rank == 0
        and (
            (np.isfinite(curr_market_struct) and curr_market_struct < 46.0)
            or (np.isfinite(curr_ifs) and curr_ifs < 58.0)
            or (np.isfinite(curr_score) and curr_score < 55.0)
            or (np.isfinite(curr_smart_money) and curr_smart_money < 40.0)
        )
    )

    clear_deterioration = bool(
        (np.isfinite(score_drop) and score_drop >= 8.0)
        and (
            (np.isfinite(ifs_drop) and ifs_drop >= 6.0)
            or (np.isfinite(struct_drop) and struct_drop >= 6.0)
            or (np.isfinite(smart_drop) and smart_drop >= 6.0)
        )
    )

    weak_streak = int(prev.get("downgrade_streak", 0) or 0)

    if buyish_prev and downgrade:
        if not strong_break and not (clear_deterioration and weak_streak >= 1):
            weak_streak += 1
            preserved = prev_decision if prev_decision in {"BUY", "STRONG BUY"} else ("BUY" if curr_decision != "AVOID" else "WATCHLIST")
            result["decision"] = preserved
            notes = result.get("notes", [])
            if not isinstance(notes, list):
                notes = [str(notes)]
            notes.append(f"Decision_Hysteresis_Preserved(streak={weak_streak})")
            result["notes"] = notes
        else:
            weak_streak = 0
    else:
        weak_streak = 0

    if result.get("decision") in {"BUY", "STRONG BUY"}:
        for key in ("entry_price", "stop_price", "target_1", "target_2", "risk_reward_1", "risk_reward_2"):
            if pd.isna(result.get(key, np.nan)) and pd.notna(prev.get(key, np.nan)):
                result[key] = prev.get(key)

    result["downgrade_streak"] = int(weak_streak)
    result["decision_raw"] = curr_decision
    result["decision_prev"] = prev_decision
    return result


def _extract_news_value(item: dict, *keys, default=""):
    if not isinstance(item, dict):
        return default
    for key in keys:
        value = item.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, dict):
            for nested_key in ("text", "title", "summary", "description", "name", "url"):
                nested = value.get(nested_key)
                if nested not in (None, ""):
                    return nested
            if "content" in value and value["content"] not in (None, ""):
                return value["content"]
        else:
            return value
    return default


def _extract_news_datetime(item: dict) -> str:
    if not isinstance(item, dict):
        return ""
    for key in ("providerPublishTime", "published_at", "pubDate", "publishedAt", "date", "datetime", "time"):
        value = item.get(key)
        if value in (None, ""):
            continue
        try:
            if isinstance(value, (int, float)) and value > 10_000_000_000:
                return pd.to_datetime(value, unit="ms", utc=True).tz_convert(None).strftime("%Y-%m-%d %H:%M")
            if isinstance(value, (int, float)):
                return pd.to_datetime(value, unit="s", utc=True).tz_convert(None).strftime("%Y-%m-%d %H:%M")
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                if getattr(parsed, "tzinfo", None) is not None:
                    parsed = parsed.tz_convert(None)
                return parsed.strftime("%Y-%m-%d %H:%M")
            return str(value)
        except Exception:
            continue
    return ""


def _parse_manual_news_lines(raw_text: str) -> list[dict]:
    items: list[dict] = []
    for line in str(raw_text or "").splitlines():
        text = line.strip()
        if not text:
            continue
        parts = [p.strip() for p in re.split(r"\s*\|\|\s*|\s*\|\s*", text) if p.strip()]
        title = parts[0] if parts else text
        source = parts[1] if len(parts) > 1 else "Manual"
        summary = parts[2] if len(parts) > 2 else ""
        items.append(
            {
                "title": title,
                "summary": summary,
                "source": source,
                "published_at": "",
                "link": "",
            }
        )
    return items




def _news_item_fingerprint(item: dict) -> str:
    if not isinstance(item, dict):
        return ""
    title = _safe_text(item.get("title") or item.get("headline") or item.get("name") or "").lower()
    source = _safe_text(item.get("source") or item.get("publisher") or item.get("provider") or item.get("site") or "").lower()
    link = _safe_text(item.get("link") or item.get("url") or "").lower()
    summary = _safe_text(item.get("summary") or item.get("description") or item.get("snippet") or "").lower()[:120]
    return "|".join([title, source, link, summary])


def _dedupe_news_items(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        fp = _news_item_fingerprint(item)
        if not fp or fp in seen:
            continue
        seen.add(fp)
        out.append(item)
    return out


def _extract_company_name_candidates(symbol: str) -> list[str]:
    base = str(symbol or "").strip()
    if not base:
        return []
    names: list[str] = []
    try:
        info_fn = globals().get("load_yf_info")
        if callable(info_fn):
            info = info_fn(base) or {}
            for key in ("shortName", "longName", "displayName", "name", "symbol"):
                value = _safe_text(info.get(key, ""))
                if value:
                    names.append(value)
    except Exception:
        pass
    cleaned: list[str] = []
    for value in names:
        value = re.sub(r"\s+", " ", value).strip()
        if value and value.lower() not in {x.lower() for x in cleaned}:
            cleaned.append(value)
    return cleaned


def _build_indonesia_news_queries(symbol: str) -> list[str]:
    base = str(symbol or "").strip()
    if not base:
        return []

    candidate_fn = globals().get("_ticker_candidates")
    candidates = candidate_fn(base) if callable(candidate_fn) else [base]
    company_candidates = _extract_company_name_candidates(base)

    local_domains = (
        "site:kontan.co.id",
        "site:bisnis.com",
        "site:cnbcindonesia.com",
        "site:cnnindonesia.com",
        "site:tempo.co",
        "site:kompas.com",
        "site:detik.com",
        "site:antaranews.com",
        "site:idx.co.id",
        "site:ojk.go.id",
        "site:bi.go.id",
    )

    queries: list[str] = []

    def add(value: str) -> None:
        value = re.sub(r"\s+", " ", str(value or "")).strip()
        if value and value not in queries:
            queries.append(value)

    # Prioritize issuer / company names because Indonesian outlets often avoid
    # pure ticker strings and use the legal name or brand name instead.
    for company in company_candidates:
        comp = re.sub(r"\s+", " ", company).strip()
        if not comp:
            continue
        for q in (
            comp,
            f"{comp} saham",
            f"{comp} emiten",
            f"{comp} berita",
            f"{comp} Indonesia",
            f'"{comp}"',
        ):
            add(q)
        for domain in local_domains[:6]:
            add(f"{comp} {domain}")

    for candidate in candidates:
        cand = str(candidate or "").strip().upper()
        if not cand:
            continue
        base_term = cand.replace(".JK", "")
        if base_term.startswith("^"):
            add(base_term)
            continue

        for q in (
            base_term,
            f"{base_term} saham",
            f"{base_term} emiten",
            f"{base_term} berita",
            f"{base_term} Indonesia",
            f'"{base_term}"',
        ):
            add(q)
        for domain in local_domains[:4]:
            add(f"{base_term} {domain}")

    return queries[:18]


def _fetch_google_news_rss(query: str) -> list[dict]:
    import requests
    from urllib.parse import quote
    import xml.etree.ElementTree as ET

    out: list[dict] = []
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=id&gl=ID&ceid=ID:id"
    resp = requests.get(rss_url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
    if not resp.ok or not resp.text.strip():
        return out

    try:
        root = ET.fromstring(resp.text)
    except Exception:
        return out

    for item in root.findall(".//item"):
        title = _safe_text(item.findtext("title", default=""))
        if not title:
            continue
        out.append({
            "title": title,
            "summary": _safe_text(item.findtext("description", default="")),
            "source": _safe_text(item.findtext("source", default="Google News")) or "Google News",
            "published_at": _safe_text(item.findtext("pubDate", default="")),
            "link": _safe_text(item.findtext("link", default="")),
        })
    return out


def _fetch_indonesia_news_google_first(symbol: str, limit: int = 12) -> list[dict]:
    queries = _build_indonesia_news_queries(symbol)
    if not queries:
        return []

    all_items: list[dict] = []
    per_query_cap = max(5, min(10, int(limit)))
    for query in queries:
        try:
            items = _fetch_google_news_rss(query)
            if not items:
                continue
            all_items.extend(items[:per_query_cap])
            if len(all_items) >= max(2 * limit, 24):
                break
        except Exception:
            continue

    return _dedupe_news_items(all_items)[: max(1, int(limit))]

@st.cache_data(ttl=900, show_spinner=False)
def fetch_ticker_news_items(symbol: str, limit: int = 12) -> list[dict]:
    base = str(symbol or "").strip()
    if not base:
        return []

    candidate_fn = globals().get("_ticker_candidates")
    candidates = candidate_fn(base) if callable(candidate_fn) else [base]
    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        cand = str(candidate).strip()
        if cand and cand not in seen:
            seen.add(cand)
            ordered_candidates.append(cand)

    fetched: list[dict] = []
    for candidate in ordered_candidates:
        try:
            ticker = yf.Ticker(candidate)
            raw_news = None

            try:
                getter = getattr(ticker, "get_news", None)
                if callable(getter):
                    raw_news = getter()
            except Exception:
                raw_news = None

            if not raw_news:
                try:
                    raw_news = getattr(ticker, "news", None)
                    if callable(raw_news):
                        raw_news = raw_news()
                except Exception:
                    raw_news = None

            if not raw_news:
                continue

            if isinstance(raw_news, dict):
                raw_news = raw_news.get("news") or raw_news.get("items") or []

            for item in raw_news:
                if not isinstance(item, dict):
                    continue
                title = _safe_text(_extract_news_value(item, "title", "headline", default=""))
                if not title:
                    content = item.get("content")
                    if isinstance(content, dict):
                        title = _safe_text(content.get("title") or content.get("headline") or content.get("summary"))
                    elif isinstance(content, str):
                        title = _safe_text(content)
                summary = _safe_text(
                    _extract_news_value(item, "summary", "description", default="")
                )
                if not summary:
                    content = item.get("content")
                    if isinstance(content, dict):
                        summary = _safe_text(content.get("summary") or content.get("description") or content.get("text"))
                source = _safe_text(
                    _extract_news_value(item, "publisher", "provider", "source", default="Yahoo Finance")
                )
                link = _safe_text(_extract_news_value(item, "link", "url", default=""))
                if not link:
                    content = item.get("content")
                    if isinstance(content, dict):
                        link = _safe_text(content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else content.get("url"))
                fetched.append(
                    {
                        "title": title,
                        "summary": summary,
                        "source": source,
                        "published_at": _extract_news_datetime(item),
                        "link": link,
                    }
                )
            if fetched:
                break
        except Exception:
            continue


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ticker_news_search_items(symbol: str, limit: int = 12) -> list[dict]:
    base = str(symbol or "").strip()
    if not base:
        return []

    candidate_fn = globals().get("_ticker_candidates")
    candidates = candidate_fn(base) if callable(candidate_fn) else [base]
    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        cand = str(candidate).strip()
        if cand and cand not in seen:
            seen.add(cand)
            ordered_candidates.append(cand)

    import requests
    from urllib.parse import quote
    import xml.etree.ElementTree as ET

    fetched: list[dict] = []

    # Yahoo Finance search endpoint fallback
    for candidate in ordered_candidates:
        try:
            url = "https://query1.finance.yahoo.com/v1/finance/search"
            params = {"q": candidate, "newsCount": 20, "quotesCount": 5, "enableFuzzyQuery": "true"}
            resp = requests.get(url, params=params, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            if resp.ok:
                payload = resp.json()
                for item in payload.get("news", []) or []:
                    content = item if isinstance(item, dict) else {}
                    title = _safe_text(content.get("title") or content.get("headline") or "")
                    summary = _safe_text(content.get("summary") or content.get("description") or "")
                    source = _safe_text(content.get("publisher") or content.get("provider") or "Yahoo Finance")
                    link = _safe_text(content.get("link") or content.get("url") or "")
                    published = ""
                    for k in ("providerPublishTime", "pubDate", "published_at", "publishedAt", "date"):
                        v = content.get(k)
                        if v not in (None, ""):
                            published = _extract_news_datetime({k: v})
                            break
                    if title:
                        fetched.append({
                            "title": title,
                            "summary": summary,
                            "source": source,
                            "published_at": published,
                            "link": link,
                        })
                if fetched:
                    break
        except Exception:
            continue

    # Yahoo RSS fallback as a second layer
    if not fetched:
        for candidate in ordered_candidates:
            for q in (candidate, candidate.replace(".JK", "")):
                try:
                    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(q)}&region=US&lang=en-US"
                    resp = requests.get(rss_url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
                    if not resp.ok or not resp.text.strip():
                        continue
                    root = ET.fromstring(resp.text)
                    for item in root.findall(".//item"):
                        title = _safe_text(item.findtext("title", default=""))
                        summary = _safe_text(item.findtext("description", default=""))
                        source = _safe_text(item.findtext("source", default="Yahoo Finance"))
                        link = _safe_text(item.findtext("link", default=""))
                        pub = _safe_text(item.findtext("pubDate", default=""))
                        if title:
                            fetched.append({
                                "title": title,
                                "summary": summary,
                                "source": source or "Yahoo Finance",
                                "published_at": pub,
                                "link": link,
                            })
                    if fetched:
                        break
                except Exception:
                    continue
            if fetched:
                break

    return fetched[: max(1, int(limit))]


@st.cache_data(ttl=900, show_spinner=False)
def fetch_indonesia_news_items(symbol: str, limit: int = 12) -> list[dict]:
    """Fetch Indonesia-focused news using Google News RSS first.

    The collector now widens coverage beyond ticker-only queries by adding:
    - company / issuer name candidates from Yahoo metadata
    - Indonesian market aliases (saham, emiten, berita)
    - curated Indonesia-local source filters (Kontan, Bisnis, CNBC Indonesia,
      Antara, Kompas, Detik, Tempo, IDX, OJK, BI)

    Yahoo Finance remains a fallback only when Google News cannot surface any
    Indonesia-focused coverage.
    """
    base = str(symbol or "").strip()
    if not base:
        return []

    fetched = _fetch_indonesia_news_google_first(base, limit=limit)
    if not fetched:
        fetched = fetch_ticker_news_search_items(symbol, limit)

    return _dedupe_news_items(fetched)[: max(1, int(limit))]


def _news_decision_label(decision: str) -> str:
    d = str(decision or "").strip().upper()
    if d == "PASS":
        return "PASSED"
    if d == "REJECT":
        return "REJECTED"
    return "WATCH"


def _decision_rank(decision: str) -> int:
    mapping = {
        "STRONG BUY": 3,
        "BUY": 2,
        "WATCHLIST": 1,
        "AVOID": 0,
    }
    return mapping.get(str(decision or "").strip().upper(), 0)


def _setup_priority_rank(setup_type: str) -> int:
    """Higher values sort earlier in the default descending ranking; Unicorn is top priority."""
    mapping = {
        "UNICORN": 5,
        "SNIPER": 4,
        "BREAKOUT": 3,
        "PULLBACK": 2,
        "REVERSAL": 1,
        "WATCHLIST": 0,
        "NONE": 0,
        "N/A": 0,
    }
    return mapping.get(str(setup_type or "").strip().upper(), 0)


def _sniper_rank(r: dict) -> int:
    if not isinstance(r, dict):
        return 0
    if str(r.get("unicorn_sniper_status", "")).upper() == "ENTRY":
        return 2
    if r.get("unicorn_sniper_valid", False):
        return 1
    return 0


def _breakout_rank(r: dict) -> int:
    if not isinstance(r, dict):
        return 0
    if str(r.get("breakout_setup_status", "")).upper() == "ENTRY":
        return 2
    if r.get("breakout_setup_valid", False) or r.get("breakout_confirmed", False):
        return 1
    return 0


def _reversal_rank(r: dict) -> int:
    if not isinstance(r, dict):
        return 0
    if str(r.get("reversal_accumulation_status", "")).upper() == "ENTRY":
        return 2
    if r.get("reversal_accumulation_valid", False):
        return 1
    return 0


def _pullback_rank(r: dict) -> int:
    if not isinstance(r, dict):
        return 0
    if str(r.get("pullback_continuation_status", "")).upper() == "ENTRY":
        return 2
    if r.get("pullback_continuation_valid", False):
        return 1
    return 0


def _build_watch_row(r: dict) -> dict:
    if not isinstance(r, dict):
        r = {}
    notes = r.get("notes", [])
    if not isinstance(notes, list):
        notes = [str(notes)] if pd.notna(notes) and str(notes).strip() else []

    symbol = _safe_text(r.get("symbol") or r.get("Ticker") or r.get("ticker") or "")
    decision = _safe_text(r.get("decision") or r.get("Decision") or r.get("DecisionRaw") or "AVOID").upper() or "AVOID"
    trade_gate = "OK" if bool(r.get("tradeability_gate_ok", False)) else "BLOCK"
    validity_ok = bool(r.get("setup_validity_ok", False))
    lifecycle = _safe_text(r.get("setup_lifecycle_stage") or r.get("Lifecycle") or r.get("lifecycle_stage") or "NO_SETUP")
    validity_reason = _safe_text(r.get("setup_validity_reason") or r.get("tradeability_gate_reason") or r.get("reason") or "n/a")
    next_action = _safe_text(r.get("setup_next_action") or "WAIT")
    trade_tier = _safe_text(r.get("tradeability_tier") or "n/a")

    row = {
        "Ticker": symbol,
        "Decision": decision,
        "DecisionRank": _decision_rank(decision),
        "SniperRank": _sniper_rank(r),
        "BreakoutRank": _breakout_rank(r),
        "PullbackRank": _pullback_rank(r),
        "ReversalRank": _reversal_rank(r),
        "Score": round(_safe_float(r.get("score", np.nan), np.nan), 2) if pd.notna(r.get("score", np.nan)) else np.nan,
        "ScoreRaw": round(_safe_float(r.get("score_raw", np.nan), np.nan), 2) if pd.notna(r.get("score_raw", np.nan)) else np.nan,
        "MarketStruct": round(_safe_float(r.get("market_structure_score", np.nan), np.nan), 2) if pd.notna(r.get("market_structure_score", np.nan)) else np.nan,
        "Trend": round(_safe_float(r.get("trend_score", np.nan), np.nan), 1) if pd.notna(r.get("trend_score", np.nan)) else np.nan,
        "Momentum": round(_safe_float(r.get("momentum_score", np.nan), np.nan), 1) if pd.notna(r.get("momentum_score", np.nan)) else np.nan,
        "Cycle": r.get("dominant_period", np.nan),
        "CycleRel": round(_safe_float(r.get("cycle_reliability", np.nan), np.nan), 1) if pd.notna(r.get("cycle_reliability", np.nan)) else np.nan,
        "Risk": round(_safe_float(r.get("risk_score", np.nan), np.nan), 1) if pd.notna(r.get("risk_score", np.nan)) else np.nan,
        "SmartMoney": round(_safe_float(r.get("smart_money_score", np.nan), np.nan), 2) if pd.notna(r.get("smart_money_score", np.nan)) else np.nan,
        "ReversalHits": r.get("reversal_hits", 0),
        "FVG": "🔥 YES" if bool(r.get("fvg_present", False)) else "NO",
        "FVG_Age": r.get("fvg_age_bars", np.nan),
        "FVG_Status": r.get("fvg_status", "-"),
        "Unicorn": "🦄 YES" if bool(r.get("unicorn_setup", False)) else "NO",
        "UnicornValid": "YES" if bool(r.get("unicorn_setup_valid", False)) else "NO",
        "UnicornStatus": r.get("unicorn_setup_status", "-"),
        "UnicornState": r.get("unicorn_setup_state", "-"),
        "Sniper": "🎯 YES" if bool(r.get("unicorn_sniper", False)) else "NO",
        "SniperValid": "YES" if bool(r.get("unicorn_sniper_valid", False)) else "NO",
        "SniperStatus": r.get("unicorn_sniper_status", "-"),
        "SniperState": r.get("unicorn_sniper_state", "-"),
        "Breakout": "🚀 YES" if bool(r.get("breakout_confirmed", False)) else "NO",
        "BreakoutValid": "YES" if bool(r.get("breakout_setup_valid", False)) else "NO",
        "BreakoutStatus": r.get("breakout_setup_status", "-"),
        "BreakoutSetup": r.get("setup_kind", "-"),
        "Pullback": "🧲 YES" if bool(r.get("pullback_continuation_valid", False)) else "NO",
        "PullbackValid": "YES" if bool(r.get("pullback_continuation_valid", False)) else "NO",
        "PullbackStatus": r.get("pullback_continuation_status", "-"),
        "PullbackSetup": "Continuation" if bool(r.get("pullback_continuation_valid", False)) else "-",
        "ReversalSignal": "🔄 YES" if bool(r.get("reversal_accumulation_valid", False)) else "NO",
        "ReversalValid": "YES" if bool(r.get("reversal_accumulation_valid", False)) else "NO",
        "ReversalStatus": r.get("reversal_accumulation_status", "-"),
        "ReversalSetup": "Accumulation" if bool(r.get("reversal_accumulation_valid", False)) else "-",
        "OrderBlock": "🎯 YES" if bool(r.get("ob_present", False)) else "NO",
        "TrendState": "BULLISH" if bool(r.get("trend_ok", False)) else "BEARISH",
        "Phase": r.get("phase", "-"),
        "PhaseConf": round(_safe_float(r.get("phase_confidence", np.nan), np.nan), 0) if pd.notna(r.get("phase_confidence", np.nan)) else np.nan,
        "MacroPhase": r.get("macro_phase", "-"),
        "MarketRegime": r.get("market_regime", "-"),
        "RegimeConf": round(_safe_float(r.get("market_regime_confidence", np.nan), np.nan), 2) if pd.notna(r.get("market_regime_confidence", np.nan)) else np.nan,
        "RegimeReason": r.get("market_regime_reason", "-"),
        "MacroScore": round(_safe_float(r.get("macro_score", np.nan), np.nan), 1) if pd.notna(r.get("macro_score", np.nan)) else np.nan,
        "MacroGate": "ON" if bool(r.get("macro_gate_ok", True)) else "OFF",
        "IFS": round(_safe_float(r.get("ifs_score", np.nan), np.nan), 2) if pd.notna(r.get("ifs_score", np.nan)) else np.nan,
        "IFSGrade": r.get("ifs_grade", "n/a"),
        "Tradeability": round(_safe_float(r.get("tradeability_score", np.nan), np.nan), 2) if pd.notna(r.get("tradeability_score", np.nan)) else np.nan,
        "TradeTier": trade_tier,
        "TradeGate": trade_gate,
        "TradeGateReason": r.get("tradeability_gate_reason", "-"),
        "EntryTrigger": r.get("entry_trigger", r.get("entry_candidate_label", "-")),
        "ProjectedFirstLeg": r.get("projected_first_leg", "-"),
        "ProjectedRebound": r.get("projected_rebound_leg", "-"),
        "EntryZoneRole": r.get("entry_zone_role", "-"),
        "EntryZoneLabel": r.get("entry_zone_label", "-"),
        "EntryProjection": r.get("entry_projection_summary", "-"),
        "EntryZoneLow": round(_safe_float(r.get("entry_zone_low", r.get("candidate_entry_zone_low", np.nan)), np.nan), 2) if pd.notna(r.get("entry_zone_low", r.get("candidate_entry_zone_low", np.nan))) else np.nan,
        "EntryZoneHigh": round(_safe_float(r.get("entry_zone_high", r.get("candidate_entry_zone_high", np.nan)), np.nan), 2) if pd.notna(r.get("entry_zone_high", r.get("candidate_entry_zone_high", np.nan))) else np.nan,
        "ProjectedEntry": round(_safe_float(r.get("entry_price_plan", r.get("entry_price", np.nan)), np.nan), 2) if pd.notna(r.get("entry_price_plan", r.get("entry_price", np.nan))) else np.nan,
        "ProjectedStop": round(_safe_float(r.get("stop_loss_plan", r.get("stop_price", np.nan)), np.nan), 2) if pd.notna(r.get("stop_loss_plan", r.get("stop_price", np.nan))) else np.nan,
        "ProjectedTP1": round(_safe_float(r.get("target_1", np.nan), np.nan), 2) if pd.notna(r.get("target_1", np.nan)) else np.nan,
        "ProjectedTP2": round(_safe_float(r.get("target_2", np.nan), np.nan), 2) if pd.notna(r.get("target_2", np.nan)) else np.nan,
        "BreakoutReference": round(_safe_float(r.get("breakout_reference", np.nan), np.nan), 2) if pd.notna(r.get("breakout_reference", np.nan)) else np.nan,
        "BreakoutInvalidation": round(_safe_float(r.get("breakout_invalidation_level", np.nan), np.nan), 2) if pd.notna(r.get("breakout_invalidation_level", np.nan)) else np.nan,
        "PullbackReference": round(_safe_float(r.get("pullback_continuation_reference", np.nan), np.nan), 2) if pd.notna(r.get("pullback_continuation_reference", np.nan)) else np.nan,
        "PullbackInvalidation": round(_safe_float(r.get("pullback_continuation_invalidation", np.nan), np.nan), 2) if pd.notna(r.get("pullback_continuation_invalidation", np.nan)) else np.nan,
        "ReversalReference": round(_safe_float(r.get("reversal_accumulation_reference", np.nan), np.nan), 2) if pd.notna(r.get("reversal_accumulation_reference", np.nan)) else np.nan,
        "ReversalInvalidation": round(_safe_float(r.get("reversal_accumulation_invalidation", np.nan), np.nan), 2) if pd.notna(r.get("reversal_accumulation_invalidation", np.nan)) else np.nan,
        "DistToEntryATR": round(_safe_float(r.get("setup_distance_to_entry_atr", np.nan), np.nan), 2) if pd.notna(r.get("setup_distance_to_entry_atr", np.nan)) else np.nan,
        "FillProb": round(_safe_float(r.get("setup_fill_probability", np.nan), np.nan), 1) if pd.notna(r.get("setup_fill_probability", np.nan)) else np.nan,
        "Lifecycle": lifecycle,
        "SetupType": _safe_text(
            r.get("setup_kind")
            or r.get("BreakoutSetup")
            or r.get("PullbackSetup")
            or r.get("ReversalSetup")
            or r.get("unicorn_entry_style")
            or lifecycle
        ),
        "SetupPriority": _setup_priority_rank(
            r.get("setup_kind")
            or r.get("BreakoutSetup")
            or r.get("PullbackSetup")
            or r.get("ReversalSetup")
            or r.get("unicorn_entry_style")
            or lifecycle
        ),
        "ExecStatus": r.get("execution_status", "n/a"),
        "Validity": "YES" if validity_ok else "NO",
        "NextAction": next_action,
        "ExecStatus": _safe_text(r.get("execution_status", "n/a")),
        "SetupVariant": _safe_text(r.get("setup_variant") or r.get("unicorn_entry_style") or r.get("entry_mode") or r.get("setup_kind") or lifecycle),
        "ValidityReason": validity_reason,
        "Value20D_Bn": round(_safe_float(r.get("avg_value_traded_20d"), np.nan) / 1e9, 2) if pd.notna(r.get("avg_value_traded_20d", np.nan)) else np.nan,
        "ProjectedEntry": round(_safe_float(_plan_value(r.get("entry_price_plan"), r.get("candidate_entry_price")), np.nan), 2) if pd.notna(_plan_value(r.get("entry_price_plan"), r.get("candidate_entry_price"))) else np.nan,
        "Stop": round(_safe_float(_plan_value(r.get("stop_loss_plan"), r.get("candidate_stop_price"), r.get("stop_price")), np.nan), 2) if pd.notna(_plan_value(r.get("stop_loss_plan"), r.get("candidate_stop_price"), r.get("stop_price"))) else np.nan,
        "TP1": round(_safe_float(_plan_value(r.get("target_1"), r.get("candidate_target_1")), np.nan), 2) if pd.notna(_plan_value(r.get("target_1"), r.get("candidate_target_1"))) else np.nan,
        "TP2": round(_safe_float(_plan_value(r.get("target_2"), r.get("candidate_target_2")), np.nan), 2) if pd.notna(_plan_value(r.get("target_2"), r.get("candidate_target_2"))) else np.nan,
        "RR1": round(_safe_float(_plan_value(r.get("risk_reward_1"), r.get("candidate_risk_reward_1")), np.nan), 2) if pd.notna(_plan_value(r.get("risk_reward_1"), r.get("candidate_risk_reward_1"))) else np.nan,
        "RR2": round(_safe_float(_plan_value(r.get("risk_reward_2"), r.get("candidate_risk_reward_2")), np.nan), 2) if pd.notna(_plan_value(r.get("risk_reward_2"), r.get("candidate_risk_reward_2"))) else np.nan,
        "SpreadPct": round(_safe_float(r.get("spread_proxy_20d", np.nan), np.nan) * 100.0, 2) if pd.notna(r.get("spread_proxy_20d", np.nan)) else np.nan,
        "GapPct": round(_safe_float(r.get("gap_proxy_20d", np.nan), np.nan) * 100.0, 2) if pd.notna(r.get("gap_proxy_20d", np.nan)) else np.nan,
        "Notes": notes,

        "valid": bool(r.get("valid", False)),
        "unicorn_setup_valid": bool(r.get("unicorn_setup_valid", False)),
        "unicorn_sniper_valid": bool(r.get("unicorn_sniper_valid", False)),
        "breakout_setup_valid": bool(r.get("breakout_setup_valid", False)),
        "breakout_confirmed": bool(r.get("breakout_confirmed", False)),
        "pullback_continuation_valid": bool(r.get("pullback_continuation_valid", False)),
        "reversal_accumulation_valid": bool(r.get("reversal_accumulation_valid", False)),
        "setup_kind": _safe_text(r.get("setup_kind", "")),
    }

    setup_plan_specs = [
        ("Breakout", "BREAKOUT"),
        ("Pullback", "PULLBACK"),
        ("Unicorn", "UNICORN"),
        ("Sniper", "SNIPER"),
        ("Reversal", "REVERSAL"),
    ]
    for label, kind in setup_plan_specs:
        setup_plan = _setup_entry_plan_bundle(r, kind)
        row[f"{label}Entry"] = round(_safe_float(_plan_value(setup_plan.get("entry_price_plan"), np.nan), np.nan), 2) if pd.notna(_plan_value(setup_plan.get("entry_price_plan"), np.nan)) else np.nan
        row[f"{label}Stop"] = round(_safe_float(_plan_value(setup_plan.get("stop_loss_plan"), np.nan), np.nan), 2) if pd.notna(_plan_value(setup_plan.get("stop_loss_plan"), np.nan)) else np.nan
        row[f"{label}TP1"] = round(_safe_float(_plan_value(setup_plan.get("target_1"), np.nan), np.nan), 2) if pd.notna(_plan_value(setup_plan.get("target_1"), np.nan)) else np.nan
        row[f"{label}TP2"] = round(_safe_float(_plan_value(setup_plan.get("target_2"), np.nan), np.nan), 2) if pd.notna(_plan_value(setup_plan.get("target_2"), np.nan)) else np.nan
        row[f"{label}ZoneLow"] = round(_safe_float(_plan_value(setup_plan.get("entry_zone_low"), np.nan), np.nan), 2) if pd.notna(_plan_value(setup_plan.get("entry_zone_low"), np.nan)) else np.nan
        row[f"{label}ZoneHigh"] = round(_safe_float(_plan_value(setup_plan.get("entry_zone_high"), np.nan), np.nan), 2) if pd.notna(_plan_value(setup_plan.get("entry_zone_high"), np.nan)) else np.nan
        row[f"{label}Trigger"] = setup_plan.get("entry_trigger", "-")
        row[f"{label}Status"] = setup_plan.get("execution_status", "n/a")
        row[f"{label}Reason"] = setup_plan.get("execution_status_reason", "n/a")
        row[f"{label}Lifecycle"] = setup_plan.get("setup_lifecycle_stage", "NO_SETUP")
        row[f"{label}Validity"] = "YES" if bool(setup_plan.get("setup_validity_ok", False)) else "NO"

    return row


def _build_watch_df(watch_rows: list[dict], ascending: bool = False) -> pd.DataFrame:
    if not watch_rows:
        return pd.DataFrame()
    watch_df = pd.DataFrame(watch_rows)
    # Prioritize actual quality metrics first; decision labels come after.
    sort_cols = [
        "SetupPriority",
        "Score",
        "IFS",
        "MarketStruct",
        "SmartMoney",
        "CycleRel",
        "DecisionRank",
        "SniperRank",
        "BreakoutRank",
        "PullbackRank",
        "ReversalRank",
    ]
    present_cols = [c for c in sort_cols if c in watch_df.columns]
    if present_cols:
        watch_df = watch_df.sort_values(
            present_cols,
            ascending=[ascending] * len(present_cols),
            na_position="last",
        )
    watch_df = watch_df.reset_index(drop=True)
    watch_df.insert(0, "Rank", range(1, len(watch_df) + 1))
    return watch_df


def _display_watch_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.drop(columns=["DecisionRank", "SniperRank", "BreakoutRank", "PullbackRank", "ReversalRank"], errors="ignore")



def _setup_bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=bool)
    if column not in df.columns:
        return pd.Series(False, index=df.index)

    series = df[column]
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)

    normalized = series.fillna("").astype(str).str.strip().str.upper()
    truthy = {"1", "TRUE", "YES", "Y", "T", "ENTRY", "VALID", "CONFIRMED"}
    falsy = {"0", "FALSE", "NO", "N", "F", "", "NAN", "NONE", "NULL"}
    return normalized.isin(truthy) & ~normalized.isin(falsy)


def _format_concept_table(df: pd.DataFrame, scope: str = "ALL") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    scope = str(scope or "ALL").strip().upper()
    out = df.copy()

    setup_concept_map = {
        "ALL": "All Concepts",
        "UNICORN": "Unicorn",
        "SNIPER": "Sniper",
        "BREAKOUT": "Breakout Retest",
        "PULLBACK": "Pullback Continuation",
        "REVERSAL": "Reversal Accumulation",
    }
    out.insert(1, "Setup Concept", setup_concept_map.get(scope, scope.title()))

    rename_map = {
        "Decision": "Decision",
        "Score": "Score",
        "MarketStruct": "Market Struct",
        "Trend": "Trend",
        "Momentum": "Momentum",
        "Risk": "Risk",
        "SmartMoney": "Smart Money",
        "FVG": "FVG",
        "FVG_Age": "FVG Age",
        "FVG_Status": "FVG Status",
        "Unicorn": "Unicorn Signal",
        "UnicornValid": "Unicorn Valid",
        "UnicornStatus": "Unicorn Status",
        "UnicornState": "Unicorn State",
        "Sniper": "Sniper Signal",
        "SniperValid": "Sniper Valid",
        "SniperStatus": "Sniper Status",
        "SniperState": "Sniper State",
        "Breakout": "Breakout Signal",
        "BreakoutValid": "Breakout Valid",
        "BreakoutStatus": "Breakout Status",
        "BreakoutSetup": "Breakout Style",
        "BreakoutReference": "Breakout Reference",
        "BreakoutInvalidation": "Breakout Invalidation",
        "Pullback": "Pullback Signal",
        "PullbackValid": "Pullback Valid",
        "PullbackStatus": "Pullback Status",
        "PullbackSetup": "Pullback Style",
        "PullbackReference": "Pullback Reference",
        "PullbackInvalidation": "Pullback Invalidation",
        "ReversalSignal": "Reversal Signal",
        "ReversalValid": "Reversal Valid",
        "ReversalStatus": "Reversal Status",
        "ReversalSetup": "Reversal Style",
        "ReversalReference": "Reversal Reference",
        "ReversalInvalidation": "Reversal Invalidation",
        "OrderBlock": "Order Block",
        "TrendState": "Trend State",
        "Phase": "Phase",
        "PhaseConf": "Phase Conf",
        "MacroPhase": "Macro Phase",
        "MarketRegime": "Market Regime",
        "RegimeConf": "Regime Conf",
        "RegimeReason": "Regime Reason",
        "MacroScore": "Macro Score",
        "MacroGate": "Macro Gate",
        "IFS": "IFS",
        "IFSGrade": "IFS Grade",
        "Tradeability": "Tradeability",
        "TradeTier": "Trade Tier",
        "TradeGate": "Trade Gate",
        "TradeGateReason": "Trade Gate Reason",
        "ExecStatus": "Execution Status",
        "DistToEntryATR": "Jarak ke Entry (ATR)",
        "FillProb": "Prob Fill",
        "RR1": "RR1",
        "RR2": "RR2",
        "Lifecycle": "Lifecycle",
        "SetupType": "Setup Concept Raw",
        "SetupPriority": "Setup Priority",
        "SetupVariant": "Setup Variant",
        "EntryTrigger": "Entry Trigger",
        "BreakoutEntry": "Breakout Entry",
        "BreakoutStop": "Breakout Stop",
        "BreakoutTP1": "Breakout TP1",
        "BreakoutTP2": "Breakout TP2",
        "BreakoutZoneLow": "Breakout Zone Low",
        "BreakoutZoneHigh": "Breakout Zone High",
        "BreakoutTrigger": "Breakout Trigger",
        "BreakoutStatus": "Breakout Status",
        "BreakoutReason": "Breakout Reason",
        "BreakoutLifecycle": "Breakout Lifecycle",
        "BreakoutValidity": "Breakout Validity",
        "PullbackEntry": "Pullback Entry",
        "PullbackStop": "Pullback Stop",
        "PullbackTP1": "Pullback TP1",
        "PullbackTP2": "Pullback TP2",
        "PullbackZoneLow": "Pullback Zone Low",
        "PullbackZoneHigh": "Pullback Zone High",
        "PullbackTrigger": "Pullback Trigger",
        "PullbackStatus": "Pullback Status",
        "PullbackReason": "Pullback Reason",
        "PullbackLifecycle": "Pullback Lifecycle",
        "PullbackValidity": "Pullback Validity",
        "UnicornEntry": "Unicorn Entry",
        "UnicornStop": "Unicorn Stop",
        "UnicornTP1": "Unicorn TP1",
        "UnicornTP2": "Unicorn TP2",
        "UnicornZoneLow": "Unicorn Zone Low",
        "UnicornZoneHigh": "Unicorn Zone High",
        "UnicornTrigger": "Unicorn Trigger",
        "UnicornStatus": "Unicorn Status",
        "UnicornReason": "Unicorn Reason",
        "UnicornLifecycle": "Unicorn Lifecycle",
        "UnicornValidity": "Unicorn Validity",
        "SniperEntry": "Sniper Entry",
        "SniperStop": "Sniper Stop",
        "SniperTP1": "Sniper TP1",
        "SniperTP2": "Sniper TP2",
        "SniperZoneLow": "Sniper Zone Low",
        "SniperZoneHigh": "Sniper Zone High",
        "SniperTrigger": "Sniper Trigger",
        "SniperStatus": "Sniper Status",
        "SniperReason": "Sniper Reason",
        "SniperLifecycle": "Sniper Lifecycle",
        "SniperValidity": "Sniper Validity",
        "ReversalEntry": "Reversal Entry",
        "ReversalStop": "Reversal Stop",
        "ReversalTP1": "Reversal TP1",
        "ReversalTP2": "Reversal TP2",
        "ReversalZoneLow": "Reversal Zone Low",
        "ReversalZoneHigh": "Reversal Zone High",
        "ReversalTrigger": "Reversal Trigger",
        "ReversalStatus": "Reversal Status",
        "ReversalReason": "Reversal Reason",
        "ReversalLifecycle": "Reversal Lifecycle",
        "ReversalValidity": "Reversal Validity",
        "EntryZoneLow": "Entry Zone Low",
        "EntryZoneHigh": "Entry Zone High",
        "EntryZoneLabel": "Entry Zone",
        "EntryZoneRole": "Zone Role",
        "ProjectedFirstLeg": "Arah Awal",
        "ProjectedRebound": "Arah Rebound",
        "EntryProjection": "Entry Projection",
        "ProjectedEntry": "Zona Entry Proyeksi",
        "ProjectedStop": "Stop Loss Proyeksi",
        "ProjectedTP1": "Target Proyeksi 1",
        "ProjectedTP2": "Target Proyeksi 2",
        "ValidityReason": "Alasan Validitas",
        "NextAction": "Aksi Berikutnya",
        "Validity": "Valid",
        "Age Bars": "Age Bars",
    }
    out = out.rename(columns=rename_map)

    # Normalize key risk/reward and fill-probability columns for all concept views.
    alias_map = {
        "ProjectedEntry": "Projected Entry",
        "FillProb": "Prob Fill",
        "RR1": "RR1",
        "RR2": "RR2",
    }
    for src, dst in alias_map.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    if scope == "UNICORN":
        wanted = [
            "Ticker", "Setup Concept", "Decision", "Score", "Tradeability", "Trade Tier", "Trade Gate",
            "Valid", "Execution Status", "Unicorn Signal", "Unicorn Valid", "Unicorn Status", "Unicorn State",
            "FVG", "FVG Age", "FVG Status",
            "Unicorn Entry", "Unicorn Stop", "Unicorn TP1", "Unicorn TP2",
            "RR1", "RR2", "Prob Fill", "Projected Entry",
            "Unicorn Zone Low", "Unicorn Zone High", "Unicorn Trigger",
            "Unicorn Lifecycle", "Unicorn Validity", "Unicorn Reason",
            "Trade Gate Reason",
        ]
    elif scope == "SNIPER":
        wanted = [
            "Ticker", "Setup Concept", "Decision", "Score", "Tradeability", "Trade Tier", "Trade Gate",
            "Valid", "Execution Status", "Sniper Signal", "Sniper Valid", "Sniper Status", "Sniper State",
            "FVG", "FVG Age", "FVG Status",
            "Sniper Entry", "Sniper Stop", "Sniper TP1", "Sniper TP2",
            "RR1", "RR2", "Prob Fill", "Projected Entry",
            "Sniper Zone Low", "Sniper Zone High", "Sniper Trigger",
            "Sniper Lifecycle", "Sniper Validity", "Sniper Reason",
            "Trade Gate Reason",
        ]
    elif scope == "BREAKOUT":
        wanted = [
            "Ticker", "Setup Concept", "Decision", "Score", "Tradeability", "Trade Tier", "Trade Gate",
            "Valid", "Execution Status", "Breakout Signal", "Breakout Valid", "Breakout Status",
            "Breakout Entry", "Breakout Stop", "Breakout TP1", "Breakout TP2",
            "RR1", "RR2", "Prob Fill", "Projected Entry",
            "Breakout Zone Low", "Breakout Zone High", "Breakout Trigger",
            "Breakout Lifecycle", "Breakout Validity", "Breakout Reason",
            "Trade Gate Reason",
        ]
    elif scope == "PULLBACK":
        wanted = [
            "Ticker", "Setup Concept", "Decision", "Score", "Tradeability", "Trade Tier", "Trade Gate",
            "Valid", "Execution Status", "Pullback Signal", "Pullback Valid", "Pullback Status",
            "Pullback Entry", "Pullback Stop", "Pullback TP1", "Pullback TP2",
            "RR1", "RR2", "Prob Fill", "Projected Entry",
            "Pullback Zone Low", "Pullback Zone High", "Pullback Trigger",
            "Pullback Lifecycle", "Pullback Validity", "Pullback Reason",
            "Trade Gate Reason",
        ]
    elif scope == "REVERSAL":
        wanted = [
            "Ticker", "Setup Concept", "Decision", "Score", "Tradeability", "Trade Tier", "Trade Gate",
            "Valid", "Execution Status", "Reversal Signal", "Reversal Valid", "Reversal Status",
            "Reversal Entry", "Reversal Stop", "Reversal TP1", "Reversal TP2",
            "RR1", "RR2", "Prob Fill", "Projected Entry",
            "Reversal Zone Low", "Reversal Zone High", "Reversal Trigger",
            "Reversal Lifecycle", "Reversal Validity", "Reversal Reason",
            "Trade Gate Reason",
        ]
    else:
        wanted = [
            "Ticker", "Setup Concept", "Decision", "Score", "Market Struct", "Trend", "Momentum", "Risk",
            "Tradeability", "Trade Tier", "Trade Gate", "Valid", "Execution Status", "Zona Entry Proyeksi", "Stop Loss Proyeksi",
            "Target Proyeksi 1", "Target Proyeksi 2", "RR1", "RR2", "Entry Zone Low", "Entry Zone High", "Zone Role",
            "Arah Awal", "Arah Rebound", "Entry Projection", "Lifecycle", "Aksi Berikutnya",
            "Alasan Validitas", "Trade Gate Reason", "Execution Status", "Prob Fill", "Jarak ke Entry (ATR)", "Setup Variant", "Entry Trigger",
        ]

    available = [c for c in wanted if c in out.columns]
    return out.loc[:, available]



def _build_concept_view(df: pd.DataFrame, scope: str = "ALL") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    scope = str(scope or "ALL").strip().upper()
    mask = pd.Series(True, index=df.index)

    def _txt_contains(col: str, needle: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(False, index=df.index)
        series = df[col].fillna("").astype(str).str.upper()
        return series.str.contains(str(needle).upper(), regex=False, na=False)

    exec_ready = pd.Series(False, index=df.index)
    if "ExecStatus" in df.columns:
        exec_ready = df["ExecStatus"].fillna("").astype(str).str.upper().eq("EXECUTION_READY")
    fill_ready = pd.Series(False, index=df.index)
    if "FillProb" in df.columns:
        fill_ready = pd.to_numeric(df["FillProb"], errors="coerce").fillna(-1) >= 90.0

    setup_priority_ok = pd.Series(False, index=df.index)
    if "SetupPriority" in df.columns:
        setup_priority_ok = pd.to_numeric(df["SetupPriority"], errors="coerce").fillna(0) > 0

    concept_any = (
        _txt_contains("SetupType", "UNICORN")
        | _txt_contains("SetupType", "SNIPER")
        | _txt_contains("SetupType", "BREAKOUT")
        | _txt_contains("SetupType", "PULLBACK")
        | _txt_contains("SetupType", "REVERSAL")
        | _txt_contains("setup_kind", "UNICORN")
        | _txt_contains("setup_kind", "SNIPER")
        | _txt_contains("setup_kind", "BREAKOUT")
        | _txt_contains("setup_kind", "PULLBACK")
        | _txt_contains("setup_kind", "REVERSAL")
    )

    if scope == "ALL":
        mask = (
            concept_any
            | setup_priority_ok
            | _setup_bool_series(df, "unicorn_setup_valid")
            | _setup_bool_series(df, "UnicornValid")
            | _setup_bool_series(df, "unicorn_sniper_valid")
            | _setup_bool_series(df, "SniperValid")
            | _setup_bool_series(df, "breakout_setup_valid")
            | _setup_bool_series(df, "BreakoutValid")
            | _setup_bool_series(df, "pullback_continuation_valid")
            | _setup_bool_series(df, "PullbackValid")
            | _setup_bool_series(df, "reversal_accumulation_valid")
            | _setup_bool_series(df, "ReversalValid")
            | exec_ready
            | fill_ready
        )
    elif scope == "UNICORN":
        mask = (
            (_txt_contains("SetupType", "UNICORN") | _txt_contains("setup_kind", "UNICORN"))
            | _setup_bool_series(df, "unicorn_setup_valid")
            | _setup_bool_series(df, "UnicornValid")
            | _setup_bool_series(df, "unicorn_setup_confirmed")
        ) & ~(
            _setup_bool_series(df, "unicorn_sniper_valid")
            | _setup_bool_series(df, "SniperValid")
            | _setup_bool_series(df, "unicorn_sniper_confirmed")
        )
    elif scope == "SNIPER":
        mask = (
            (_txt_contains("SetupType", "SNIPER") | _txt_contains("setup_kind", "SNIPER"))
            | _setup_bool_series(df, "unicorn_sniper_valid")
            | _setup_bool_series(df, "SniperValid")
            | _setup_bool_series(df, "unicorn_sniper_confirmed")
        )
    elif scope == "BREAKOUT":
        mask = (
            (_txt_contains("SetupType", "BREAKOUT") | _txt_contains("setup_kind", "BREAKOUT"))
            | _setup_bool_series(df, "breakout_setup_valid")
            | _setup_bool_series(df, "BreakoutValid")
            | _setup_bool_series(df, "breakout_confirmed")
            | _setup_bool_series(df, "Breakout")
        )
    elif scope == "PULLBACK":
        mask = (
            (_txt_contains("SetupType", "PULLBACK") | _txt_contains("setup_kind", "PULLBACK") | _txt_contains("SetupVariant", "CONTINUATION"))
            | _setup_bool_series(df, "pullback_continuation_valid")
            | _setup_bool_series(df, "PullbackValid")
        )
    elif scope == "REVERSAL":
        mask = (
            (_txt_contains("SetupType", "REVERSAL") | _txt_contains("setup_kind", "REVERSAL"))
            | _setup_bool_series(df, "reversal_accumulation_valid")
            | _setup_bool_series(df, "ReversalValid")
        )
    else:
        mask = pd.Series(True, index=df.index)

    view = df.loc[mask].copy() if isinstance(mask, pd.Series) else df.copy()
    if view.empty:
        return pd.DataFrame()
    return _format_concept_table(view, scope=scope)


def _recalibrate_global_scan_results(valid_results: list[dict]) -> list[dict]:
    """Annotate the scan results without changing the underlying decision engine.

    Top 20 and Deep Dive now share the same score / decision logic, so this
    function only preserves compatibility with the existing call site.
    """
    if not valid_results:
        return valid_results

    try:
        base_df = pd.DataFrame(
            {
                "Score": [float(_safe_float(r.get("score"), np.nan)) for r in valid_results],
                "IFS": [float(_safe_float(r.get("ifs_score"), np.nan)) for r in valid_results],
            }
        )
        if base_df.empty:
            return valid_results

        score_pct = base_df["Score"].rank(pct=True, method="average") * 100.0
        ifs_pct = base_df["IFS"].rank(pct=True, method="average") * 100.0

        for idx, r in enumerate(valid_results):
            raw_score = float(_safe_float(r.get("score"), np.nan))
            spct = float(score_pct.iloc[idx])
            ipct = float(ifs_pct.iloc[idx])

            # Keep the engine output intact; only add rank metadata for display.
            r["score_raw"] = raw_score
            r["score_pct"] = spct
            r["ifs_pct"] = ipct
            r["DecisionRaw"] = str(r.get("Decision", r.get("decision", "AVOID")))

            notes = r.get("notes", [])
            if not isinstance(notes, list):
                notes = [str(notes)] if pd.notna(notes) and str(notes).strip() else []
            notes.append(f"GlobalScanAnnotated:ScorePct={spct:.1f};IFSRankPct={ipct:.1f}")
            r["notes"] = notes

            valid_results[idx] = r

        return valid_results
    except Exception:
        return valid_results


def _save_trade_journal_from_scan(results: list[dict], account_id: str = "default") -> int:
    """Persist the latest scan results into the trade journal ledger."""
    if not results:
        return 0
    saved = 0
    scan_date = dt.datetime.utcnow().date().isoformat()
    for r in results:
        if not isinstance(r, dict) or not r.get("symbol"):
            continue
        entry_plan = r.get("entry_plan", {}) if isinstance(r.get("entry_plan", {}), dict) else {}
        lifecycle_json = {
            "decision": r.get("decision"),
            "score": r.get("score"),
            "ifs_score": r.get("ifs_score"),
            "tradeability_score": r.get("tradeability_score"),
            "setup_lifecycle_stage": r.get("setup_lifecycle_stage"),
            "setup_validity_ok": r.get("setup_validity_ok"),
            "setup_validity_reason": r.get("setup_validity_reason"),
            "setup_next_action": r.get("setup_next_action"),
            "entry_plan": {
                "entry_price": entry_plan.get("entry_price_plan", r.get("entry_price_plan")),
                "stop_price": entry_plan.get("stop_loss_plan", r.get("stop_loss_plan")),
                "target_1": entry_plan.get("target_1", r.get("target_1")),
                "target_2": entry_plan.get("target_2", r.get("target_2")),
            },
        }
        try:
            pe.upsert_trade_journal(
                symbol=str(r.get("symbol")),
                scan_date=scan_date,
                setup_stage=str(r.get("setup_lifecycle_stage") or r.get("setup_kind") or "UNKNOWN"),
                validity_ok=bool(r.get("setup_validity_ok", False)),
                validity_reason=str(r.get("setup_validity_reason", r.get("tradeability_gate_reason", "n/a"))),
                next_action=str(r.get("setup_next_action", "WAIT")),
                decision=str(r.get("decision", "AVOID")),
                setup_kind=str(r.get("setup_kind", "")),
                score=float(r.get("score", np.nan)) if pd.notna(r.get("score", np.nan)) else None,
                ifs_score=float(r.get("ifs_score", np.nan)) if pd.notna(r.get("ifs_score", np.nan)) else None,
                catalyst_score=float(r.get("catalyst_score", np.nan)) if pd.notna(r.get("catalyst_score", np.nan)) else None,
                tradeability_score=float(r.get("tradeability_score", np.nan)) if pd.notna(r.get("tradeability_score", np.nan)) else None,
                entry_price=float(entry_plan.get("entry_price_plan", r.get("entry_price_plan", np.nan))) if pd.notna(entry_plan.get("entry_price_plan", r.get("entry_price_plan", np.nan))) else None,
                stop_price=float(entry_plan.get("stop_loss_plan", r.get("stop_loss_plan", np.nan))) if pd.notna(entry_plan.get("stop_loss_plan", r.get("stop_loss_plan", np.nan))) else None,
                target_1=float(entry_plan.get("target_1", r.get("target_1", np.nan))) if pd.notna(entry_plan.get("target_1", r.get("target_1", np.nan))) else None,
                target_2=float(entry_plan.get("target_2", r.get("target_2", np.nan))) if pd.notna(entry_plan.get("target_2", r.get("target_2", np.nan))) else None,
                risk_reward_1=float(r.get("setup_rr_1", r.get("risk_reward_1", np.nan))) if pd.notna(r.get("setup_rr_1", r.get("risk_reward_1", np.nan))) else None,
                risk_reward_2=float(r.get("setup_rr_2", r.get("risk_reward_2", np.nan))) if pd.notna(r.get("setup_rr_2", r.get("risk_reward_2", np.nan))) else None,
                value_traded_20d=float(r.get("avg_value_traded_20d", np.nan)) if pd.notna(r.get("avg_value_traded_20d", np.nan)) else None,
                spread_proxy_20d=float(r.get("spread_proxy_20d", np.nan)) if pd.notna(r.get("spread_proxy_20d", np.nan)) else None,
                gap_proxy_20d=float(r.get("gap_proxy_20d", np.nan)) if pd.notna(r.get("gap_proxy_20d", np.nan)) else None,
                notes=str(r.get("notes", "")),
                lifecycle_json=lifecycle_json,
                account_id=account_id,
                source="scanner_scan",
            )
            saved += 1
        except Exception:
            continue
    return saved


# =========================================================
# Universe loading
# =========================================================
if universe_mode == "Paste tickers":
    universe = parse_universe_text(paste_text)
elif universe_mode == "Upload CSV":
    universe = load_universe_tickers_from_csv(uploaded_file) if uploaded_file is not None else []
else:
    local_file = Path("midcap_universe.csv")
    universe = load_universe_tickers_from_csv(local_file) if local_file.exists() else []

# Keep the scan practical on mobile: dedupe and cap the active universe.
seen_universe = set()
_clean_universe = []
for sym in universe:
    sym = str(sym or "").strip().upper()
    if not sym or sym in seen_universe:
        continue
    seen_universe.add(sym)
    _clean_universe.append(sym)
universe = _clean_universe[: int(scan_limit)] if scan_limit else _clean_universe

if "global_scan_results" not in st.session_state:
    st.session_state.global_scan_results = []
if "global_watch_df" not in st.session_state:
    st.session_state.global_watch_df = pd.DataFrame()
if "global_watch_df_raw" not in st.session_state:
    st.session_state.global_watch_df_raw = pd.DataFrame()
if "global_unicorn_df" not in st.session_state:
    st.session_state.global_unicorn_df = pd.DataFrame()
if "global_unicorn_df_raw" not in st.session_state:
    st.session_state.global_unicorn_df_raw = pd.DataFrame()
if "global_breakout_df" not in st.session_state:
    st.session_state.global_breakout_df = pd.DataFrame()
if "global_breakout_df_raw" not in st.session_state:
    st.session_state.global_breakout_df_raw = pd.DataFrame()
if "global_reversal_df" not in st.session_state:
    st.session_state.global_reversal_df = pd.DataFrame()
if "global_reversal_df_raw" not in st.session_state:
    st.session_state.global_reversal_df_raw = pd.DataFrame()
if "global_pullback_df" not in st.session_state:
    st.session_state.global_pullback_df = pd.DataFrame()
if "global_pullback_df_raw" not in st.session_state:
    st.session_state.global_pullback_df_raw = pd.DataFrame()
if "global_valid_results" not in st.session_state:
    st.session_state.global_valid_results = []

# Shared analysis defaults are supplied directly by the widgets below so
# Streamlit does not warn about session-state collisions on the same keys.

flow_val = map_flow_to_score("Netral")
GLOBAL_BENCHMARK_SYMBOL = "^JKSE"


def process_symbol(symbol: str, scan_context: dict):
    try:
        months_local = int(scan_context.get("months", months))
        min_history_bars_local = int(scan_context.get("min_history_bars", min_history_bars))
        min_avg_volume_local = float(scan_context.get("min_avg_volume", min_avg_volume))
        min_price_local = float(scan_context.get("min_price", min_price))
        max_price_local = float(scan_context.get("max_price", max_price))
        strategy_mode_local = str(scan_context.get("strategy_mode", "Balanced"))
        bandarmology_mode_local = str(scan_context.get("bandarmology_mode", "Netral"))
        benchmark_symbol_local = str(scan_context.get("benchmark_symbol", GLOBAL_BENCHMARK_SYMBOL)).strip() or GLOBAL_BENCHMARK_SYMBOL
        show_benchmark_local = bool(scan_context.get("show_benchmark", True))
        entry_buffer_atr_local = float(scan_context.get("entry_buffer_atr", 0.25))
        stop_loss_atr_local = float(scan_context.get("stop_loss_atr", 1.8))
        take_profit_1_atr_local = float(scan_context.get("take_profit_1_atr", 2.2))
        take_profit_2_atr_local = float(scan_context.get("take_profit_2_atr", 3.8))
        macro_context = scan_context.get("macro_context")

        d = load_ticker_data(symbol, months_local)
        if d.empty or len(d) < min_history_bars_local:
            return {"valid": False, "symbol": symbol, "reason": "Data historis tidak mencukupi"}

        flow_val_local = map_flow_to_score(bandarmology_mode_local)

        fundamental = compute_fundamental_grade(symbol)
        future_context = compute_future_fundamental_grade(symbol, d, macro_context)
        res = score_stock_smc(
            d,
            flow_used=True,
            flow_val=flow_val_local,
            min_avg_volume=min_avg_volume_local,
            min_price=min_price_local,
            max_price=max_price_local,
            mode=strategy_mode_local,
            min_history_bars=min_history_bars_local,
            macro_context=macro_context,
            future_fundamental_context=future_context,
        )

        res["symbol"] = symbol
        res["fundamental_score"] = fundamental.get("fundamental_score", np.nan)
        res["fundamental_grade"] = fundamental.get("fundamental_grade", "n/a")
        res["expected_revenue_growth_next_q"] = future_context.get("expected_revenue_growth_next_q", np.nan)
        res["expected_eps_growth_next_q"] = future_context.get("expected_eps_growth_next_q", np.nan)
        res["expected_margin_next_q"] = future_context.get("expected_margin_next_q", np.nan)

        ifs_context = compute_institutional_forward_score(
            symbol=symbol,
            price_df=d,
            bench_df=(macro_context.get("benchmark_df") if isinstance(macro_context, dict) else None),
            current_fundamental=fundamental,
            future_context=future_context,
            technical_context=res,
        )
        res["ifs_score"] = ifs_context.get("ifs_score", np.nan)
        res["ifs_grade"] = ifs_context.get("ifs_grade", "n/a")
        res["ifs_breakdown"] = ifs_context.get("ifs_breakdown", {})
        res["ifs_detail"] = ifs_context.get("ifs_detail", {})

        # Build the same trade plan that the deep dive uses, so Top 20 and detail stay aligned.
        res["entry_plan"] = build_entry_plan(
            res,
            entry_buffer_atr=entry_buffer_atr_local,
            stop_loss_atr=stop_loss_atr_local,
            target_1_atr=take_profit_1_atr_local,
            target_2_atr=take_profit_2_atr_local,
        )
        res.update(res["entry_plan"])

        return res
    except Exception as e:
        return {"valid": False, "symbol": symbol, "reason": str(e)}

def run_deep_dive_analysis(
    ticker_input: str,
    strategy_mode: str,
    bandarmology_mode: str,
    benchmark_symbol_local: str,
    show_benchmark_local: bool,
    entry_buffer_atr_local: float,
    stop_loss_atr_local: float,
    take_profit_1_atr_local: float,
    take_profit_2_atr_local: float,
) -> dict:
    """Run a single-ticker deep dive and return a reusable analysis bundle."""
    deep_ticker = normalize_ticker(ticker_input)
    flow_val_local = map_flow_to_score(bandarmology_mode)

    stock_df = load_ticker_data(deep_ticker, months)
    bench_df = load_ticker_data(benchmark_symbol_local, months) if benchmark_symbol_local else pd.DataFrame()

    macro_context = None
    if show_benchmark_local and benchmark_symbol_local:
        macro_context = _load_macro_context(benchmark_symbol_local, months)

    if stock_df.empty or len(stock_df) < min_history_bars:
        return {
            "symbol": deep_ticker,
            "stock_df": stock_df,
            "bench_df": bench_df,
            "macro_context": macro_context,
            "stock_res": None,
            "fundamental": None,
            "future_context": None,
            "ifs_context": None,
            "entry_plan": None,
            "error": "Data ticker tidak cukup atau gagal diunduh.",
        }

    try:
        future_fundamental_context = compute_future_fundamental_grade(deep_ticker, stock_df, macro_context)
    except Exception as exc:
        future_fundamental_context = {
            "current_fundamental_score": np.nan,
            "current_fundamental_grade": "n/a",
            "future_fundamental_score": np.nan,
            "future_fundamental_grade": "n/a",
            "future_fundamental_direction": "n/a",
            "future_fundamental_confidence": np.nan,
            "future_growth_proxy": np.nan,
            "future_fundamental_momentum_score": np.nan,
            "future_seasonal_anomaly_score": np.nan,
            "future_inflection_score": np.nan,
            "future_cash_flow_proxy": np.nan,
            "future_balance_quality": np.nan,
            "future_price_proxy": np.nan,
            "future_cycle_support": np.nan,
            "future_macro_score": np.nan,
            "future_macro_adjusted_score": np.nan,
            "future_macro_gate_ok": False,
            "future_macro_gate_reason": f"future_fundamental_error: {type(exc).__name__}",
            "expected_revenue_growth_next_q": np.nan,
            "expected_eps_growth_next_q": np.nan,
            "expected_margin_next_q": np.nan,
            "future_moat_reason": f"future_fundamental_error: {type(exc).__name__}",
            "future_reliability": np.nan,
            "future_time_to_top": np.nan,
            "future_time_to_bottom": np.nan,
            "future_phase": "Unknown",
        }
    try:
        stock_res = score_stock_smc(
            stock_df,
            flow_used=True,
            flow_val=flow_val_local,
            min_avg_volume=min_avg_volume,
            min_price=min_price,
            max_price=max_price,
            mode=strategy_mode,
            min_history_bars=min_history_bars,
            macro_context=macro_context,
            future_fundamental_context=future_fundamental_context,
        )
    except Exception as exc:
        return {
            "symbol": deep_ticker,
            "stock_df": stock_df,
            "bench_df": bench_df,
            "macro_context": macro_context,
            "stock_res": None,
            "fundamental": None,
            "future_context": future_fundamental_context,
            "ifs_context": None,
            "entry_plan": None,
            "error": f"score_stock_smc failed: {type(exc).__name__}: {exc}",
        }

    try:
        fundamental = compute_fundamental_grade(deep_ticker)
    except Exception:
        fundamental = {}
    stock_res["peg_ratio"] = fundamental.get("peg_ratio", np.nan)
    stock_res["trailing_pe"] = fundamental.get("trailing_pe", np.nan)
    stock_res["forward_pe"] = fundamental.get("forward_pe", np.nan)
    stock_res["revenue_growth"] = fundamental.get("revenue_growth", np.nan)
    stock_res["earnings_growth"] = fundamental.get("earnings_growth", np.nan)
    stock_res["profit_margins"] = fundamental.get("profit_margins", np.nan)
    stock_res["future_fundamental_score"] = future_fundamental_context.get("future_fundamental_score", np.nan)
    stock_res["future_fundamental_grade"] = future_fundamental_context.get("future_fundamental_grade", "n/a")
    stock_res["future_fundamental_direction"] = future_fundamental_context.get("future_fundamental_direction", "n/a")
    stock_res["future_fundamental_confidence"] = future_fundamental_context.get("future_fundamental_confidence", np.nan)
    stock_res["future_fundamental_phase"] = future_fundamental_context.get("future_phase", "Unknown")
    stock_res["future_fundamental_reason"] = future_fundamental_context.get("future_moat_reason", "n/a")
    stock_res["expected_revenue_growth_next_q"] = future_fundamental_context.get("expected_revenue_growth_next_q", np.nan)
    stock_res["expected_eps_growth_next_q"] = future_fundamental_context.get("expected_eps_growth_next_q", np.nan)
    stock_res["expected_margin_next_q"] = future_fundamental_context.get("expected_margin_next_q", np.nan)

    try:
        entry_plan = build_entry_plan(
            stock_res,
            entry_buffer_atr=entry_buffer_atr_local,
            stop_loss_atr=stop_loss_atr_local,
            target_1_atr=take_profit_1_atr_local,
            target_2_atr=take_profit_2_atr_local,
        )
    except Exception as exc:
        entry_plan = {
            "entry_valid": False,
            "entry_reason": f"build_entry_plan failed: {type(exc).__name__}",
        }
    stock_res["entry_plan"] = entry_plan
    stock_res.update(entry_plan)

    try:
        ifs_context = compute_institutional_forward_score(
            symbol=deep_ticker,
            price_df=stock_df,
            bench_df=bench_df,
            current_fundamental=fundamental,
            future_context=future_fundamental_context,
            technical_context=stock_res,
        )
    except Exception as exc:
        ifs_context = {
            "ifs_score": np.nan,
            "ifs_grade": "n/a",
            "ifs_breakdown": {},
            "ifs_detail": {},
            "error": f"compute_institutional_forward_score failed: {type(exc).__name__}: {exc}",
        }
    stock_res["ifs_score"] = ifs_context.get("ifs_score", np.nan)
    stock_res["ifs_grade"] = ifs_context.get("ifs_grade", "n/a")
    stock_res["ifs_breakdown"] = ifs_context.get("ifs_breakdown", {})
    stock_res["ifs_detail"] = ifs_context.get("ifs_detail", {})

    return {
        "symbol": deep_ticker,
        "stock_df": stock_df,
        "bench_df": bench_df,
        "macro_context": macro_context,
        "stock_res": stock_res,
        "fundamental": fundamental,
        "future_context": future_fundamental_context,
        "ifs_context": ifs_context,
        "entry_plan": entry_plan,
    }

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Market Structure", "🏦 Institutional Forward Score", "🧪 Walk-Forward Lab", "⬇️ OHLCV Downloader", "💼 Live Portfolio"])

with tab1:
    st.subheader("Market Structure + Tradeability Top 20")
    st.caption("Fokus pada trend, momentum, cycle, risk, dan apakah setup cukup tradeable untuk uang real.")

    if run_global_scan:
        if not universe:
            st.error("Universe kosong. Isi tickers di sidebar terlebih dahulu.")
        else:
            scan_strategy_mode = str(st.session_state.get("deep_strategy_mode", "Balanced"))
            scan_bandarmology_mode = str(st.session_state.get("deep_bandarmology_mode", "Netral"))
            scan_benchmark_symbol = str(st.session_state.get("deep_benchmark_symbol", GLOBAL_BENCHMARK_SYMBOL)).strip() or GLOBAL_BENCHMARK_SYMBOL
            scan_show_benchmark = bool(st.session_state.get("deep_show_benchmark", True))
            entry_buffer_atr_local = float(st.session_state.get("deep_entry_buffer_atr", 0.25))
            stop_loss_atr_local = float(st.session_state.get("deep_stop_loss_atr", 1.8))
            take_profit_1_atr_local = float(st.session_state.get("deep_take_profit_1_atr", 2.2))
            take_profit_2_atr_local = float(st.session_state.get("deep_take_profit_2_atr", 3.8))
            macro_context = None
            if scan_show_benchmark and scan_benchmark_symbol:
                macro_context = _load_macro_context(scan_benchmark_symbol, months)

            scan_context = {
                "months": months,
                "min_history_bars": min_history_bars,
                "min_avg_volume": min_avg_volume,
                "min_price": min_price,
                "max_price": max_price,
                "strategy_mode": scan_strategy_mode,
                "bandarmology_mode": scan_bandarmology_mode,
                "benchmark_symbol": scan_benchmark_symbol,
                "show_benchmark": scan_show_benchmark,
                "entry_buffer_atr": entry_buffer_atr_local,
                "stop_loss_atr": stop_loss_atr_local,
                "take_profit_1_atr": take_profit_1_atr_local,
                "take_profit_2_atr": take_profit_2_atr_local,
                "macro_context": macro_context,
            }

            st.write(f"⚙️ Memproses analisis struktural pada **{len(universe)}** emiten...")
            progress = st.progress(0)
            status = st.empty()
            results = []

            with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(process_symbol, sym, scan_context): sym for sym in universe}
                done = 0
                total = len(futures)
                for fut in cf.as_completed(futures):
                    done += 1
                    progress.progress(done / total)
                    status.caption(f"Selesai mengurai: {done}/{total} -> {futures[fut]}")
                    results.append(fut.result())

            progress.empty()
            status.empty()

            st.session_state.global_scan_results = results
            scan_rows = [r for r in results if isinstance(r, dict) and _safe_text(r.get("symbol") or r.get("Ticker") or r.get("ticker") or "")]
            valid_results = [
                r for r in scan_rows
                if bool(r.get("valid", False))
                or bool(r.get("setup_validity_ok", False))
                or str(r.get("Decision", r.get("decision", ""))).upper() in {"BUY", "STRONG BUY", "WATCHLIST"}
                or bool(r.get("breakout_confirmed", False))
                or bool(r.get("unicorn_setup_valid", False))
                or bool(r.get("unicorn_sniper_valid", False))
                or bool(r.get("pullback_continuation_valid", False))
                or bool(r.get("reversal_accumulation_valid", False))
            ]
            valid_results = _recalibrate_global_scan_results(valid_results) or []
            st.session_state.global_valid_results = valid_results
            st.session_state.global_scan_rows = scan_rows

            if not scan_rows:
                st.session_state.global_watch_df_raw = pd.DataFrame()
                st.session_state.global_watch_df = pd.DataFrame()
                st.session_state.global_unicorn_df_raw = pd.DataFrame()
                st.session_state.global_unicorn_df = pd.DataFrame()
                st.session_state.global_breakout_df_raw = pd.DataFrame()
                st.session_state.global_breakout_df = pd.DataFrame()
                st.session_state.global_pullback_df_raw = pd.DataFrame()
                st.session_state.global_pullback_df = pd.DataFrame()
                st.session_state.global_reversal_df_raw = pd.DataFrame()
                st.session_state.global_reversal_df = pd.DataFrame()
                st.session_state.global_eligible_results = []
                st.session_state.global_unicorn_results = []
                st.warning("Tidak ada hasil scan yang bisa diproses dari universe ini.")
            else:
                watch_rows = [_build_watch_row(r) for r in scan_rows]

                watch_df_raw = pd.DataFrame(watch_rows)
                watch_df = _build_watch_df(watch_rows, ascending=(ranking_sort_mode == "Ascending"))

                unicorn_df_raw = watch_df_raw[
                    (
                        _setup_bool_series(watch_df_raw, "unicorn_setup_valid")
                        | _setup_bool_series(watch_df_raw, "UnicornValid")
                    )
                    & ~(
                        _setup_bool_series(watch_df_raw, "unicorn_sniper_valid")
                        | _setup_bool_series(watch_df_raw, "SniperValid")
                    )
                ].copy() if not watch_df_raw.empty else pd.DataFrame()
                sniper_df_raw = watch_df_raw[
                    _setup_bool_series(watch_df_raw, "unicorn_sniper_valid") | _setup_bool_series(watch_df_raw, "SniperValid")
                ].copy() if not watch_df_raw.empty else pd.DataFrame()
                breakout_df_raw = watch_df_raw[
                    _setup_bool_series(watch_df_raw, "breakout_setup_valid") | _setup_bool_series(watch_df_raw, "BreakoutValid") | _setup_bool_series(watch_df_raw, "breakout_confirmed") | _setup_bool_series(watch_df_raw, "Breakout")
                ].copy() if not watch_df_raw.empty else pd.DataFrame()
                pullback_df_raw = watch_df_raw[
                    _setup_bool_series(watch_df_raw, "pullback_continuation_valid") | _setup_bool_series(watch_df_raw, "PullbackValid")
                ].copy() if not watch_df_raw.empty else pd.DataFrame()
                reversal_df_raw = watch_df_raw[
                    _setup_bool_series(watch_df_raw, "reversal_accumulation_valid") | _setup_bool_series(watch_df_raw, "ReversalValid")
                ].copy() if not watch_df_raw.empty else pd.DataFrame()
                unicorn_df = _build_watch_df(unicorn_df_raw.to_dict("records"), ascending=(ranking_sort_mode == "Ascending")) if not unicorn_df_raw.empty else pd.DataFrame()
                sniper_df = _build_watch_df(sniper_df_raw.to_dict("records"), ascending=(ranking_sort_mode == "Ascending")) if not sniper_df_raw.empty else pd.DataFrame()
                breakout_df = _build_watch_df(breakout_df_raw.to_dict("records"), ascending=(ranking_sort_mode == "Ascending")) if not breakout_df_raw.empty else pd.DataFrame()
                pullback_df = _build_watch_df(pullback_df_raw.to_dict("records"), ascending=(ranking_sort_mode == "Ascending")) if not pullback_df_raw.empty else pd.DataFrame()
                reversal_df = _build_watch_df(reversal_df_raw.to_dict("records"), ascending=(ranking_sort_mode == "Ascending")) if not reversal_df_raw.empty else pd.DataFrame()

                st.session_state.global_watch_df_raw = watch_df_raw
                st.session_state.global_watch_df = watch_df
                st.session_state.global_unicorn_df_raw = unicorn_df_raw
                st.session_state.global_unicorn_df = unicorn_df
                st.session_state.global_sniper_df_raw = sniper_df_raw
                st.session_state.global_sniper_df = sniper_df
                st.session_state.global_breakout_df_raw = breakout_df_raw
                st.session_state.global_breakout_df = breakout_df
                st.session_state.global_pullback_df_raw = pullback_df_raw
                st.session_state.global_pullback_df = pullback_df
                st.session_state.global_reversal_df_raw = reversal_df_raw
                st.session_state.global_reversal_df = reversal_df
                st.session_state.global_eligible_results = valid_results or []
                st.session_state.global_unicorn_results = [
                    r for r in (valid_results or [])
                    if (
                        str(r.get("Decision", r.get("decision", ""))).upper() in {"BUY", "STRONG BUY"}
                        or str(r.get("setup_lifecycle_stage", "")).upper() in {"ENTRY_ZONE", "ENTRY_TRIGGERED", "ENTRY_WATCH"}
                        or str(r.get("breakout_setup_status", "")).upper() == "ENTRY"
                        or str(r.get("unicorn_setup_status", "")).upper() == "ENTRY"
                        or str(r.get("pullback_continuation_status", "")).upper() == "ENTRY"
                        or str(r.get("reversal_accumulation_status", "")).upper() == "ENTRY"
                        or str(r.get("ExecStatus", r.get("execution_status", ""))).upper() == "EXECUTION_READY"
                    )
                    and (
                        bool(r.get("breakout_confirmed", False))
                        or (bool(r.get("unicorn_setup_valid", False)) and not bool(r.get("unicorn_sniper_valid", False)))
                        or bool(r.get("pullback_continuation_valid", False))
                        or bool(r.get("reversal_accumulation_valid", False))
                        or str(r.get("setup_validity_ok", "")).lower() in {"true", "1", "yes"}
                        or str(r.get("ExecStatus", r.get("execution_status", ""))).upper() == "EXECUTION_READY"
                    )
                ]

                watch_view = _build_concept_view(watch_df, "ALL")
                unicorn_view = _build_concept_view(unicorn_df, "UNICORN")
                sniper_view = _build_concept_view(sniper_df, "SNIPER")
                breakout_view = _build_concept_view(breakout_df, "BREAKOUT")
                pullback_df_raw = st.session_state.global_pullback_df_raw if "global_pullback_df_raw" in st.session_state else pd.DataFrame()
                pullback_df = st.session_state.global_pullback_df if "global_pullback_df" in st.session_state else pd.DataFrame()
                pullback_view = _build_concept_view(pullback_df, "PULLBACK")
                reversal_df_raw = st.session_state.global_reversal_df_raw if "global_reversal_df_raw" in st.session_state else pd.DataFrame()
                reversal_df = st.session_state.global_reversal_df if "global_reversal_df" in st.session_state else pd.DataFrame()
                reversal_view = _build_concept_view(reversal_df, "REVERSAL")
                setup_priority_ok = (
                    pd.to_numeric(watch_df_raw["SetupPriority"], errors="coerce").fillna(0) > 0
                    if "SetupPriority" in watch_df_raw.columns
                    else pd.Series(False, index=watch_df_raw.index)
                )
                setup_candidate_raw = watch_df_raw[
                    setup_priority_ok
                    | _setup_bool_series(watch_df_raw, "unicorn_setup_valid")
                    | _setup_bool_series(watch_df_raw, "UnicornValid")
                    | _setup_bool_series(watch_df_raw, "unicorn_sniper_valid")
                    | _setup_bool_series(watch_df_raw, "SniperValid")
                    | _setup_bool_series(watch_df_raw, "breakout_setup_valid")
                    | _setup_bool_series(watch_df_raw, "BreakoutValid")
                    | _setup_bool_series(watch_df_raw, "breakout_confirmed")
                    | _setup_bool_series(watch_df_raw, "Breakout")
                    | _setup_bool_series(watch_df_raw, "pullback_continuation_valid")
                    | _setup_bool_series(watch_df_raw, "PullbackValid")
                    | _setup_bool_series(watch_df_raw, "reversal_accumulation_valid")
                    | _setup_bool_series(watch_df_raw, "ReversalValid")
                    | (watch_df_raw["ExecStatus"].fillna("").astype(str).str.upper().eq("EXECUTION_READY") if "ExecStatus" in watch_df_raw.columns else pd.Series(False, index=watch_df_raw.index))
                    | (pd.to_numeric(watch_df_raw["FillProb"], errors="coerce").fillna(-1) >= 90.0 if "FillProb" in watch_df_raw.columns else pd.Series(False, index=watch_df_raw.index))
                    | (pd.to_numeric(watch_df_raw["SetupPriority"], errors="coerce").fillna(0).astype(float).gt(0) if "SetupPriority" in watch_df_raw.columns else pd.Series(False, index=watch_df_raw.index))
                ].copy() if not watch_df_raw.empty else pd.DataFrame()
                setup_candidate_view = _build_concept_view(_build_watch_df(setup_candidate_raw.to_dict("records"), ascending=(ranking_sort_mode == "Ascending")), "ALL") if not setup_candidate_raw.empty else pd.DataFrame()

                st.caption(
                    f"Scan rows: {len(watch_df)} | Valid technical rows: {len(valid_results)} | Setup candidates: {len(setup_candidate_view)} | Unicorn: {len(unicorn_view)} | Sniper: {len(sniper_view)} | Breakout: {len(breakout_view)} | Pullback: {len(pullback_view)} | Reversal: {len(reversal_view)} | Ranking order: {ranking_sort_mode}"
                )

                priority_df = _build_watch_df(watch_rows, ascending=False)
                exec_ready_mask = (
                    (priority_df["ExecStatus"].fillna("").astype(str).str.upper().eq("EXECUTION_READY") if "ExecStatus" in priority_df.columns else pd.Series(False, index=priority_df.index))
                    | (pd.to_numeric(priority_df["FillProb"], errors="coerce").fillna(-1) >= 90.0 if "FillProb" in priority_df.columns else pd.Series(False, index=priority_df.index))
                ) if not priority_df.empty else pd.Series(dtype=bool)
                top20_priority = priority_df[exec_ready_mask].head(20).copy() if not priority_df.empty else pd.DataFrame()
                journal_col1, journal_col2 = st.columns([1, 2])
                with journal_col1:
                    if st.button("Save current scan to journal", type="secondary", key="save_current_scan_to_journal_top3"):
                        saved = _save_trade_journal_from_scan(st.session_state.get("global_scan_rows", []), account_id="default")
                        st.success(f"{saved} snapshot jurnal tersimpan.")
                with journal_col2:
                    st.caption("Trade journal menyimpan lifecycle, validity, tradeability, entry projection, stop, dan target untuk review manual Stockbit.")
                st.subheader("🔥 Top 3 High-Conviction Setups")
                top3 = priority_df[
                    priority_df["Decision"].isin(["BUY", "STRONG BUY"])
                    & (priority_df.get("TradeGate", "OK") == "OK")
                    & (priority_df.get("Validity", "NO") == "YES")
                    & exec_ready_mask
                ].head(3)
                if not top3.empty:
                    if mobile_mode or len(top3) == 1:
                        for row in top3.itertuples():
                            with st.container():
                                entry_value = getattr(row, "ProjectedEntry", getattr(row, "Entry", np.nan))
                                stop_value = getattr(row, "Stop", np.nan)
                                st.metric(
                                    label=f"🌟 {row.Ticker} ({row.Decision})",
                                    value=f"Rp {entry_value:,.0f}" if pd.notna(entry_value) else f"Rp {stop_value:,.0f}",
                                    delta=f"IFS: {row.IFS}",
                                )
                                st.markdown(_build_setup_summary(row))
                                st.markdown("---")
                    else:
                        cols = st.columns(len(top3))
                        for idx, row in enumerate(top3.itertuples()):
                            with cols[idx]:
                                entry_value = getattr(row, "ProjectedEntry", getattr(row, "Entry", np.nan))
                                stop_value = getattr(row, "Stop", np.nan)
                                st.metric(
                                    label=f"🌟 {row.Ticker} ({row.Decision})",
                                    value=f"Rp {entry_value:,.0f}" if pd.notna(entry_value) else f"Rp {stop_value:,.0f}",
                                    delta=f"IFS: {row.IFS}",
                                )
                                st.markdown(_build_setup_summary(row))
                else:
                    st.info("Belum ada kandidat BUY/STRONG BUY yang lolos tradeability gate pada universe saat ini.")

                actionable_df = priority_df[
                    priority_df["Decision"].isin(["BUY", "STRONG BUY"])
                    & (priority_df.get("TradeGate", "OK") == "OK")
                    & (priority_df.get("Validity", "NO") == "YES")
                    & exec_ready_mask
                ].copy()
                if not actionable_df.empty:
                    st.markdown("---")
                    st.subheader("📱 HP Trade Ticket")
                    st.caption("Pilih setup lalu salin ticket ini ke Stockbit notes / checklist manual Anda sebelum entry.")
                    ticket_options = actionable_df["Ticker"].astype(str).tolist()
                    selected_setup = st.selectbox("Pilih ticker", ticket_options, key="hp_stockbit_ticket_picker")
                    ticket_row = actionable_df[actionable_df["Ticker"].astype(str) == str(selected_setup)].iloc[0]
                    ticket_text = _build_stockbit_ticket(ticket_row)
                    st.text_area("Stockbit ticket", value=ticket_text, height=280, key="hp_stockbit_ticket_text")
                    st.download_button(
                        "Download ticket .txt",
                        data=ticket_text.encode("utf-8"),
                        file_name=f"{selected_setup}_stockbit_ticket.txt",
                        mime="text/plain",
                    )

                st.markdown("---")
                st.subheader("🏆 Market Structure Ranking (Top 20)")
                if not top20_priority.empty:
                    st.dataframe(_build_watch_df(top20_priority.to_dict("records"), ascending=(ranking_sort_mode == "Ascending")), width="stretch", hide_index=True)
                else:
                    st.dataframe(watch_view.head(20), width="stretch", hide_index=True)

                st.markdown("---")
                st.subheader("🎯 All Setup Candidates")
                if setup_candidate_view.empty:
                    st.info("Tidak ada setup candidate yang lolos pada universe ini.")
                else:
                    st.dataframe(setup_candidate_view.head(20), width="stretch", hide_index=True)

                st.markdown("---")
                st.subheader("🧲 Pullback Candidates")
                if pullback_view.empty:
                    st.info("Tidak ada kandidat Pullback Continuation pada universe ini.")
                else:
                    st.dataframe(pullback_view.head(20), width="stretch", hide_index=True)

                st.markdown("---")
                st.subheader("🚀 Breakout Candidates")
                if breakout_view.empty:
                    st.info("Tidak ada kandidat Breakout pada universe ini.")
                else:
                    st.dataframe(breakout_view.head(20), width="stretch", hide_index=True)

                st.markdown("---")
                st.subheader("🦄 Unicorn Candidates")
                if unicorn_view.empty:
                    st.info("Tidak ada kandidat Unicorn pada universe ini.")
                else:
                    st.dataframe(unicorn_view.head(20), width="stretch", hide_index=True)

                st.markdown("---")
                st.subheader("🎯 Sniper Candidates")
                if sniper_view.empty:
                    st.info("Tidak ada kandidat Sniper pada universe ini.")
                else:
                    st.dataframe(sniper_view.head(20), width="stretch", hide_index=True)

                st.markdown("---")
                st.subheader("🔄 Reversal Candidates")
                if reversal_view.empty:
                    st.info("Tidak ada kandidat Reversal Accumulation pada universe ini.")
                else:
                    st.dataframe(reversal_view.head(20), width="stretch", hide_index=True)

                journal_col1, journal_col2 = st.columns([1, 2])
                with journal_col1:
                    if st.button("Save current scan to journal", type="secondary", key="save_current_scan_to_journal_breakout"):
                        saved = _save_trade_journal_from_scan(st.session_state.get("global_valid_results", []), account_id="default")
                        st.success(f"{saved} snapshot jurnal tersimpan.")
                with journal_col2:
                    st.caption("Trade journal menyimpan lifecycle, validity, tradeability, entry projection, stop, dan target untuk review manual Stockbit.")

                st.markdown("---")
                st.subheader("🧾 Trade Journal Snapshot")
                journal_df = pe.list_trade_journal(limit=25)
                if journal_df.empty:
                    st.info("Belum ada trade journal tersimpan. Klik tombol save untuk menyalin scan ini ke ledger.")
                else:
                    st.dataframe(journal_df.head(10), width="stretch", hide_index=True)
    else:
        if not st.session_state.global_watch_df_raw.empty:
            watch_df = _build_watch_df(st.session_state.global_watch_df_raw.to_dict("records"), ascending=(ranking_sort_mode == "Ascending"))
            watch_view = _build_concept_view(watch_df, "ALL")
            unicorn_raw = st.session_state.global_unicorn_df_raw
            unicorn_df = _build_watch_df(unicorn_raw.to_dict("records"), ascending=(ranking_sort_mode == "Ascending")) if not unicorn_raw.empty else pd.DataFrame()
            unicorn_view = _build_concept_view(unicorn_df, "UNICORN")
            sniper_raw = st.session_state.global_sniper_df_raw if "global_sniper_df_raw" in st.session_state else pd.DataFrame()
            sniper_df = _build_watch_df(sniper_raw.to_dict("records"), ascending=(ranking_sort_mode == "Ascending")) if not sniper_raw.empty else pd.DataFrame()
            sniper_view = _build_concept_view(sniper_df, "SNIPER")
            breakout_raw = st.session_state.global_breakout_df_raw
            breakout_df = _build_watch_df(breakout_raw.to_dict("records"), ascending=(ranking_sort_mode == "Ascending")) if not breakout_raw.empty else pd.DataFrame()
            breakout_view = _build_concept_view(breakout_df, "BREAKOUT")

            st.caption(
                f"Valid results: {len(watch_df)} | Unicorn: {len(unicorn_view)} | Sniper: {len(sniper_view)} | Breakout: {len(breakout_view)} | Ranking order: {ranking_sort_mode}"
            )
            st.subheader("🏆 Market Structure Ranking (Top 20)")
            st.dataframe(watch_view.head(20), width="stretch", hide_index=True)
            st.markdown("---")
            st.subheader("🦄 Unicorn Candidates")
            if unicorn_view.empty:
                st.info("Tidak ada kandidat Unicorn pada universe ini.")
            else:
                st.dataframe(unicorn_view.head(20), width="stretch", hide_index=True)
            st.markdown("---")
            st.subheader("🎯 Sniper Candidates")
            if sniper_view.empty:
                st.info("Tidak ada kandidat Sniper pada universe ini.")
            else:
                st.dataframe(sniper_view.head(20), width="stretch", hide_index=True)
            st.markdown("---")
            st.subheader("🚀 Breakout Candidates")
            if breakout_view.empty:
                st.info("Tidak ada kandidat Breakout pada universe ini.")
            else:
                st.dataframe(breakout_view.head(20), width="stretch", hide_index=True)
            st.markdown("---")
            st.subheader("🧾 Trade Journal Snapshot")
            journal_df = pe.list_trade_journal(limit=25)
            if journal_df.empty:
                st.info("Belum ada trade journal tersimpan.")
            else:
                st.dataframe(journal_df.head(10), width="stretch", hide_index=True)
            st.info("Klik **Run global scan** di sidebar untuk memperbarui ranking.")
        else:
            st.info("Klik **Run global scan** di sidebar untuk mulai scan universe.")

with tab2:
    st.subheader("🏦 Institutional Forward Score")
    st.caption("Dibagi menjadi overview, factor breakdown, smart money, forward fundamental proxy, entry plan, dan detail saham. Confidence akan ikut turun bila coverage data tipis.")

    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    with c1:
        ticker_input = st.text_input("Ticker saham", value="BMRI", key="deep_ticker_input")
    with c2:
        strategy_mode = st.selectbox(
            "Strategy mode",
            ["Conservative", "Balanced", "Aggressive"],
            index=1,
            key="deep_strategy_mode",
        )
    with c3:
        bandarmology_mode = st.selectbox(
            "Bandarmology",
            ["Big Akumulasi", "Small Akumulasi", "Netral", "Small Distribusi", "Big Distribusi"],
            index=2,
            key="deep_bandarmology_mode",
        )

    with st.expander("⚙️ Deep Dive Settings", expanded=True):
        d1, d2, d3, d4 = st.columns([1, 1, 1, 1])
        with d1:
            benchmark_symbol_local = st.text_input("Benchmark IHSG symbol", value="^JKSE", key="deep_benchmark_symbol")
        with d2:
            show_benchmark_local = st.checkbox("Tampilkan benchmark vs saham", value=True, key="deep_show_benchmark")
        with d3:
            entry_buffer_atr_local = st.slider("Entry buffer (x ATR)", 0.10, 1.00, 0.25, 0.05, key="deep_entry_buffer_atr")
        with d4:
            stop_loss_atr_local = st.slider("Stop Loss (x ATR)", 1.0, 5.0, 1.8, 0.1, key="deep_stop_loss_atr")

        d5, d6 = st.columns([1, 1])
        with d5:
            take_profit_1_atr_local = st.slider("Take Profit 1 (x ATR)", 1.0, 6.0, 2.2, 0.1, key="deep_take_profit_1_atr")
        with d6:
            take_profit_2_atr_local = st.slider("Take Profit 2 (x ATR)", 2.0, 8.0, 3.8, 0.1, key="deep_take_profit_2_atr")

        analyze_btn = st.button("Analyze ticker", type="primary", key="deep_analyze_btn")

    analysis_bundle = {}
    if analyze_btn:
        deep_ticker = normalize_ticker(ticker_input)
        analysis_bundle = run_deep_dive_analysis(
            ticker_input=ticker_input,
            strategy_mode=strategy_mode,
            bandarmology_mode=bandarmology_mode,
            benchmark_symbol_local=benchmark_symbol_local,
            show_benchmark_local=show_benchmark_local,
            entry_buffer_atr_local=entry_buffer_atr_local,
            stop_loss_atr_local=stop_loss_atr_local,
            take_profit_1_atr_local=take_profit_1_atr_local,
            take_profit_2_atr_local=take_profit_2_atr_local,
        )
        st.session_state.ifs_analysis = analysis_bundle
    else:
        analysis_bundle = st.session_state.get("ifs_analysis", {})

    stock_res = analysis_bundle.get("stock_res")
    ifs_context = analysis_bundle.get("ifs_context")
    fundamental = analysis_bundle.get("fundamental")
    future_context = analysis_bundle.get("future_context")
    deep_ticker = analysis_bundle.get("symbol", normalize_ticker(ticker_input))
    st.session_state["deep_selected_symbol"] = deep_ticker
    stock_df = analysis_bundle.get("stock_df", pd.DataFrame())
    bench_df = analysis_bundle.get("bench_df", pd.DataFrame())
    macro_context = analysis_bundle.get("macro_context")
    entry_plan = _effective_entry_plan(analysis_bundle.get("stock_res", {}) if isinstance(analysis_bundle, dict) else {})

    sub_overview, sub_factor, sub_smart, sub_forward, sub_entry, sub_news, sub_detail = st.tabs(
        ["Overview", "Factor Breakdown", "Smart Money", "Forward Fundamental", "Entry Plan", "News Catalyst", "Detail Saham"]
    )

    with sub_entry:
        st.subheader("Entry Plan")
        if ifs_context is not None and stock_res is not None:
            plan = _effective_entry_plan(stock_res)
            cols = st.columns(4)
            cols[0].metric("Signal", stock_res.get("decision", "n/a"), f'Confidence {ifs_context.get("ifs_detail", {}).get("future_confidence", np.nan):.0f}%')
            cols[1].metric("Projected Entry", f'Rp {plan.get("entry_price_plan", np.nan):,.0f}' if pd.notna(plan.get("entry_price_plan", np.nan)) else "n/a")
            cols[2].metric("Stop Loss", f'Rp {plan.get("stop_loss_plan", np.nan):,.0f}' if pd.notna(plan.get("stop_loss_plan", np.nan)) else "n/a")
            cols[3].metric("Trigger", plan.get("entry_trigger", "n/a"), plan.get("plan_reason", "n/a"))

            trade_cols = st.columns(4)
            trade_cols[0].metric("Tradeability", f'{plan.get("tradeability_score", np.nan):.2f}' if pd.notna(plan.get("tradeability_score", np.nan)) else "n/a", plan.get("execution_status", "n/a"))
            trade_cols[1].metric("Trade Gate", "OK" if plan.get("tradeability_ok", False) else "BLOCK", plan.get("tradeability_reason", "n/a"))
            trade_cols[2].metric("Lifecycle", plan.get("setup_lifecycle_stage", "n/a"), plan.get("setup_validity_reason", "n/a"))
            trade_cols[3].metric("Next Action", plan.get("setup_next_action", "n/a"), "Valid" if plan.get("setup_validity_ok", False) else "Invalid")

            entry_table = pd.DataFrame(
                [
                    {"Metric": "Decision", "Value": stock_res.get("decision", "n/a")},
                    {"Metric": "Tradeability Score", "Value": f'{plan.get("tradeability_score", np.nan):.2f}' if pd.notna(plan.get("tradeability_score", np.nan)) else "n/a"},
                    {"Metric": "Trade Gate", "Value": "OK" if plan.get("tradeability_ok", False) else "BLOCK"},
                    {"Metric": "Lifecycle Stage", "Value": plan.get("setup_lifecycle_stage", "n/a")},
                    {"Metric": "Setup Valid", "Value": "YES" if plan.get("setup_validity_ok", False) else "NO"},
                    {"Metric": "Validity Reason", "Value": plan.get("setup_validity_reason", "n/a")},
                    {"Metric": "Next Action", "Value": plan.get("setup_next_action", "n/a")},
                    {"Metric": "Entry Valid", "Value": "YES" if plan.get("entry_valid", False) else "NO"},
                    {"Metric": "Execution Status", "Value": plan.get("execution_status", "n/a")},
                    {"Metric": "Entry Mode", "Value": plan.get("entry_mode", "n/a")},
                    {"Metric": "Entry Zone Low", "Value": f'Rp {plan.get("entry_zone_low", np.nan):,.0f}' if pd.notna(plan.get("entry_zone_low", np.nan)) else "n/a"},
                    {"Metric": "Entry Zone High", "Value": f'Rp {plan.get("entry_zone_high", np.nan):,.0f}' if pd.notna(plan.get("entry_zone_high", np.nan)) else "n/a"},
                    {"Metric": "Entry Trigger", "Value": plan.get("entry_trigger", "n/a")},
                    {"Metric": "Entry Zone Role", "Value": plan.get("entry_zone_role", "n/a")},
                    {"Metric": "Projected First Leg", "Value": plan.get("projected_first_leg", "n/a")},
                    {"Metric": "Projected Rebound", "Value": plan.get("projected_rebound_leg", "n/a")},
                    {"Metric": "Entry Projection", "Value": plan.get("entry_projection_summary", "n/a")},
                    {"Metric": "Stop Loss", "Value": f'Rp {plan.get("stop_loss_plan", np.nan):,.0f}' if pd.notna(plan.get("stop_loss_plan", np.nan)) else "n/a"},
                    {"Metric": "Target 1", "Value": f'Rp {plan.get("target_1", np.nan):,.0f}' if pd.notna(plan.get("target_1", np.nan)) else "n/a"},
                    {"Metric": "Target 2", "Value": f'Rp {plan.get("target_2", np.nan):,.0f}' if pd.notna(plan.get("target_2", np.nan)) else "n/a"},
                    {"Metric": "Risk / Share", "Value": f'Rp {plan.get("risk_per_share", np.nan):,.0f}' if pd.notna(plan.get("risk_per_share", np.nan)) else "n/a"},
                    {"Metric": "RR1", "Value": f'{plan.get("risk_reward_1", np.nan):.2f}' if pd.notna(plan.get("risk_reward_1", np.nan)) else "n/a"},
                    {"Metric": "RR2", "Value": f'{plan.get("risk_reward_2", np.nan):.2f}' if pd.notna(plan.get("risk_reward_2", np.nan)) else "n/a"},
                    {"Metric": "Upside TP1", "Value": f'{plan.get("upside_to_t1_pct", np.nan):.2f}%' if pd.notna(plan.get("upside_to_t1_pct", np.nan)) else "n/a"},
                    {"Metric": "Upside TP2", "Value": f'{plan.get("upside_to_t2_pct", np.nan):.2f}%' if pd.notna(plan.get("upside_to_t2_pct", np.nan)) else "n/a"},
                    {"Metric": "Plan Reason", "Value": plan.get("plan_reason", "n/a")},
                ]
            )
            st.dataframe(entry_table, width="stretch", hide_index=True)
        else:
            st.info("Klik Analyze ticker untuk melihat entry plan otomatis.")


    with sub_news:
        st.subheader("News Catalyst")
        st.caption("Berita diklasifikasikan oleh modul Catalyst NLP: PASS / WATCH / REJECT berdasarkan relevansi struktural, bukan sekadar sentimen pasar.")

        news_default_symbol = st.session_state.get("deep_selected_symbol") or deep_ticker or normalize_ticker(ticker_input)
        if "news_catalyst_symbol" not in st.session_state:
            st.session_state["news_catalyst_symbol"] = news_default_symbol if news_default_symbol else "BBCA.JK"

        news_symbol = st.text_input(
            "Ticker untuk news scan",
            key="news_catalyst_symbol",
        )

        news_source_mode = st.radio(
            "Sumber berita",
            ["Yahoo Finance", "Indonesia News", "Paste manual"],
            horizontal=True,
            key="news_catalyst_source_mode",
        )

        news_limit = st.slider("Maksimal item berita", 5, 25, 10, key="news_catalyst_limit")
        news_paste = st.text_area(
            "Paste berita manual (format: judul | sumber | ringkasan). Satu item per baris.",
            height=170,
            placeholder="Contoh:\nLaba BBCA tumbuh 12% | Reuters | Emiten membukukan pertumbuhan laba...\nBI tahan suku bunga | BI | Kebijakan moneter tetap akomodatif...",
            key="news_catalyst_manual_input",
        )

        fetch_col1, fetch_col2 = st.columns([1, 1])
        fetch_clicked = fetch_col1.button("Refresh news catalyst", type="primary", key="news_catalyst_fetch_btn")
        use_cached = fetch_col2.checkbox("Gunakan cache terakhir", value=True, key="news_catalyst_use_cache")

        cache = st.session_state.get("news_catalyst_cache", {})
        should_fetch = False
        if news_source_mode in ("Yahoo Finance", "Indonesia News"):
            cache_symbol = cache.get("symbol")
            cache_mode = cache.get("source_mode")
            if fetch_clicked:
                should_fetch = True
            elif use_cached and cache_symbol == news_symbol and cache_mode == news_source_mode and cache.get("raw_news"):
                should_fetch = False
            elif cache_symbol != news_symbol or cache_mode != news_source_mode or not cache.get("raw_news"):
                should_fetch = True

        if news_source_mode in ("Yahoo Finance", "Indonesia News"):
            if should_fetch:
                with st.spinner(f"Mengambil news untuk {news_symbol}..."):
                    if news_source_mode == "Indonesia News":
                        raw_news = fetch_indonesia_news_items(news_symbol, news_limit)
                    else:
                        raw_news = fetch_ticker_news_items(news_symbol, news_limit)
                    scored_news = filter_news_items(raw_news, symbol=news_symbol) if raw_news else []
                    cache = {
                        "symbol": news_symbol,
                        "source_mode": news_source_mode,
                        "raw_news": raw_news,
                        "scored_news": scored_news,
                    }
                    st.session_state.news_catalyst_cache = cache
            scored_news = cache.get("scored_news", []) if cache.get("symbol") == news_symbol and cache.get("source_mode") == news_source_mode else []
            raw_news = cache.get("raw_news", []) if cache.get("symbol") == news_symbol and cache.get("source_mode") == news_source_mode else []

            if not raw_news:
                if news_source_mode == "Indonesia News":
                    fallback_news = fetch_indonesia_news_items(news_symbol, news_limit)
                else:
                    fallback_news = fetch_ticker_news_search_items(news_symbol, news_limit)
                if fallback_news:
                    raw_news = fallback_news
                    scored_news = filter_news_items(raw_news, symbol=news_symbol)
                    cache = {
                        "symbol": news_symbol,
                        "source_mode": news_source_mode,
                        "raw_news": raw_news,
                        "scored_news": scored_news,
                        "fallback": "indonesia_google_news" if news_source_mode == "Indonesia News" else "yahoo_search",
                    }
                    st.session_state.news_catalyst_cache = cache

        else:
            raw_news = _parse_manual_news_lines(news_paste)
            scored_news = filter_news_items(raw_news, symbol=news_symbol) if raw_news else []
            if fetch_clicked:
                cache = {
                    "symbol": news_symbol,
                    "source_mode": "Paste manual",
                    "raw_news": raw_news,
                    "scored_news": scored_news,
                }
                st.session_state.news_catalyst_cache = cache

        if news_source_mode == "Paste manual" and fetch_clicked:
            raw_news = _parse_manual_news_lines(news_paste)
            scored_news = filter_news_items(raw_news, symbol=news_symbol) if raw_news else []
            cache = {
                "symbol": news_symbol,
                "source_mode": "Paste manual",
                "raw_news": raw_news,
                "scored_news": scored_news,
            }
            st.session_state.news_catalyst_cache = cache

        scored_news = scored_news if isinstance(scored_news, list) else []
        raw_news = raw_news if isinstance(raw_news, list) else []

        if not scored_news:
            st.info("Belum ada item news catalyst. Klik refresh untuk Indonesia News / Yahoo Finance, atau paste berita manual untuk dianalisis.")
        else:
            pass_count = sum(1 for item in scored_news if str(getattr(item, "decision", "")).upper() == "PASS")
            watch_count = sum(1 for item in scored_news if str(getattr(item, "decision", "")).upper() == "WATCH")
            reject_count = sum(1 for item in scored_news if str(getattr(item, "decision", "")).upper() == "REJECT")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Items", len(scored_news))
            m2.metric("PASS", pass_count)
            m3.metric("WATCH", watch_count)
            m4.metric("REJECT", reject_count)

            news_rows = []
            for idx, item in enumerate(scored_news):
                raw_item = raw_news[idx] if idx < len(raw_news) and isinstance(raw_news[idx], dict) else {}
                decision = str(getattr(item, "decision", "WATCH")).upper()
                news_rows.append(
                    {
                        "Decision": _news_decision_label(decision),
                        "Category": getattr(item, "category", "unknown"),
                        "Confidence": int(getattr(item, "confidence", 50)),
                        "Horizon": getattr(item, "impact_horizon", "days"),
                        "Materiality": getattr(item, "materiality", "medium"),
                        "Source Quality": getattr(item, "source_quality", "unknown"),
                        "Source Tier": getattr(item, "source_tier", getattr(item, "source_quality", "unknown")),
                        "Freshness": getattr(item, "freshness_score", 0),
                        "Relevance": getattr(item, "relevance_score", 0),
                        "Title": raw_item.get("title", ""),
                        "Source": raw_item.get("source", ""),
                        "Published": raw_item.get("published_at", ""),
                        "Link": raw_item.get("link", ""),
                        "Summary": raw_item.get("summary", ""),
                        "Reasons": ", ".join(getattr(item, "reasons", [])) if getattr(item, "reasons", None) else "",
                        "Red Flags": ", ".join(getattr(item, "red_flags", [])) if getattr(item, "red_flags", None) else "",
                        "Tags": ", ".join(getattr(item, "tags", [])) if getattr(item, "tags", None) else "",
                    }
                )

            news_df = pd.DataFrame(news_rows)
            if not news_df.empty:
                st.dataframe(
                    news_df[
                        [
                            "Decision",
                            "Confidence",
                            "Category",
                            "Horizon",
                            "Materiality",
                            "Source Quality",
                            "Source Tier",
                            "Freshness",
                            "Relevance",
                            "Published",
                            "Title",
                            "Source",
                            "Link",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

                st.markdown("#### Detail per item")
                for i, row in news_df.iterrows():
                    title = row.get("Title", "")
                    badge = row.get("Decision", "WATCH")
                    confidence = row.get("Confidence", 50)
                    with st.expander(f"{badge} · {confidence}% · {title}", expanded=False):
                        st.write(f"**Category:** {row.get('Category', 'unknown')}")
                        st.write(f"**Horizon:** {row.get('Horizon', 'days')}")
                        st.write(f"**Materiality:** {row.get('Materiality', 'medium')}")
                        st.write(f"**Source quality:** {row.get('Source Quality', 'unknown')}")
                        st.write(f"**Source tier:** {row.get('Source Tier', 'unknown')}")
                        st.write(f"**Freshness score:** {row.get('Freshness', 0)}")
                        st.write(f"**Relevance score:** {row.get('Relevance', 0)}")
                        if row.get("Published"):
                            st.write(f"**Published:** {row.get('Published')}")
                        if row.get("Source"):
                            st.write(f"**Source:** {row.get('Source')}")
                        if row.get("Link"):
                            st.write(f"**Link:** {row.get('Link')}")
                        if row.get("Summary"):
                            st.write(f"**Summary:** {row.get('Summary')}")
                        if row.get("Reasons"):
                            st.write(f"**Reasons:** {row.get('Reasons')}")
                        if row.get("Red Flags"):
                            st.write(f"**Red flags:** {row.get('Red Flags')}")
                        if row.get("Tags"):
                            st.write(f"**Tags:** {row.get('Tags')}")

    with sub_detail:
        if analyze_btn:
            deep_ticker = normalize_ticker(ticker_input)
            flow_val_local = map_flow_to_score(bandarmology_mode)

            stock_df = load_ticker_data(deep_ticker, months)
            bench_df = load_ticker_data(benchmark_symbol_local, months) if benchmark_symbol_local else pd.DataFrame()

            macro_context = None
            if show_benchmark_local and not bench_df.empty and len(bench_df) >= min_history_bars:
                macro_context = build_macro_liquidity_gate(bench_df.copy(), benchmark_symbol_local)

            if stock_df.empty or len(stock_df) < min_history_bars:
                st.error("Data ticker tidak cukup atau gagal diunduh.")
            else:
                future_fundamental_context = compute_future_fundamental_grade(deep_ticker, stock_df, macro_context)

                stock_res = score_stock_smc(
                    stock_df,
                    flow_used=True,
                    flow_val=flow_val_local,
                    min_avg_volume=min_avg_volume,
                    min_price=min_price,
                    max_price=max_price,
                    mode=strategy_mode,
                    min_history_bars=min_history_bars,
                    macro_context=macro_context,
                    future_fundamental_context=future_fundamental_context,
                )

                if not stock_res.get("valid", False):
                    st.warning(stock_res.get("reason", "Analisis teknikal tidak valid."))
                    st.stop()

                stock = stock_res.get("df", pd.DataFrame()).copy()
                stock_last = stock_res.get("last")
                fundamental = compute_fundamental_grade(deep_ticker)
                stock_res["peg_ratio"] = fundamental.get("peg_ratio", np.nan)
                stock_res["trailing_pe"] = fundamental.get("trailing_pe", np.nan)
                stock_res["forward_pe"] = fundamental.get("forward_pe", np.nan)
                stock_res["revenue_growth"] = fundamental.get("revenue_growth", np.nan)
                stock_res["earnings_growth"] = fundamental.get("earnings_growth", np.nan)
                stock_res["profit_margins"] = fundamental.get("profit_margins", np.nan)
                stock_res["future_fundamental_score"] = future_fundamental_context.get("future_fundamental_score", np.nan)
                stock_res["future_fundamental_grade"] = future_fundamental_context.get("future_fundamental_grade", "n/a")
                stock_res["future_fundamental_direction"] = future_fundamental_context.get("future_fundamental_direction", "n/a")
                stock_res["future_fundamental_confidence"] = future_fundamental_context.get("future_fundamental_confidence", np.nan)
                stock_res["future_fundamental_phase"] = future_fundamental_context.get("future_phase", "Unknown")
                stock_res["future_fundamental_reason"] = future_fundamental_context.get("future_moat_reason", "n/a")
                ifs_context = compute_institutional_forward_score(
                    symbol=deep_ticker,
                    price_df=stock_df,
                    bench_df=bench_df,
                    current_fundamental=fundamental,
                    future_context=future_fundamental_context,
                    technical_context=stock_res,
                )
                entry_plan = build_entry_plan(
                    stock_res,
                    entry_buffer_atr=entry_buffer_atr_local,
                    stop_loss_atr=stop_loss_atr_local,
                    target_1_atr=take_profit_1_atr_local,
                    target_2_atr=take_profit_2_atr_local,
                )
                stock_res["entry_plan"] = entry_plan
                stock_res.update(entry_plan)
                stock_res["ifs_score"] = ifs_context.get("ifs_score", np.nan)
                stock_res["ifs_grade"] = ifs_context.get("ifs_grade", "n/a")
                stock_res["ifs_breakdown"] = ifs_context.get("ifs_breakdown", {})
                stock_res["ifs_detail"] = ifs_context.get("ifs_detail", {})
                st.session_state.ifs_analysis = {
                    "symbol": deep_ticker,
                    "stock_df": stock_df,
                    "bench_df": bench_df,
                    "macro_context": macro_context,
                    "stock_res": stock_res,
                    "fundamental": fundamental,
                    "future_context": future_fundamental_context,
                    "ifs_context": ifs_context,
                    "entry_plan": entry_plan,
                    "strategy_mode": strategy_mode,
                    "bandarmology_mode": bandarmology_mode,
                    "ticker_input": ticker_input,
                }
                bench = pd.DataFrame()
                bench_cycle = None
                if macro_context is not None:
                    bench = bench_df.copy()
                    bench_cycle = macro_context.get("cycle_tuple")

                stock_status = "Near Bottom" if stock_res["time_to_bottom"] <= 4 else "Mid-Cycle Moving"
                bench_status = "n/a"
                if macro_context is not None:
                    bench_status = "Near Bottom" if macro_context.get("macro_time_to_bottom", 999) <= 4 else "Mid-Cycle Moving"
                stock_top = stock_res.get("time_to_top", np.nan)
                stock_phase_age = stock_res.get("phase_age_bars", np.nan)
                stock_phase_age_pct = stock_res.get("phase_age_pct", np.nan)
                macro_context = macro_context or build_macro_liquidity_gate(pd.DataFrame(), benchmark_symbol_local)
                bench_top = macro_context.get("macro_time_to_top", np.nan) if macro_context is not None else np.nan
                bench_phase_age = macro_context.get("macro_phase_age_bars", np.nan) if macro_context is not None else np.nan
                bench_phase_age_pct = macro_context.get("macro_phase_age_pct", np.nan) if macro_context is not None else np.nan

                st.markdown(
                    """
                    <div style="margin-top: 0.25rem;">
                        <h2 style="margin-bottom:0.25rem;">⏳ Trader Time Analysis Model</h2>
                        <div style="font-size:1.05rem; opacity:0.9;">
                            Mengukur frekuensi dominan dan estimasi waktu pembalikan tren berlandaskan struktur matematika siklus bursa.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                stock_period = stock_res["dominant_period"]
                stock_ttb = stock_res["time_to_bottom"]
                stock_cycle_info = stock_res.get("cycle_info", {})
                bench_period = bench_cycle[0] if bench_cycle is not None else None
                bench_ttb = bench_cycle[1] if bench_cycle is not None else None
                bench_cycle_info = bench_cycle[3] if bench_cycle is not None and len(bench_cycle) > 3 else {}

                stock_html = f"""
                <div style="background:linear-gradient(180deg, rgba(235,244,255,1) 0%, rgba(225,235,250,1) 100%); padding:22px; border-radius:18px; border:1px solid rgba(0,0,0,0.05); box-shadow:0 8px 24px rgba(0,0,0,0.04);">
                    <div style="font-size:1.15rem; font-weight:700; color:#173b6d; margin-bottom:18px;">Siklus Saham ({deep_ticker})</div>
                    <div style="font-size:1.02rem; color:#173b6d; line-height:2;">
                        <div>• <b>Periode Siklus Dominan:</b> {stock_period} Hari Bursa</div>
                        <div>• <b>Estimasi Sisa Waktu Menuju Bottom berikutnya:</b> {stock_ttb} Bar</div>
                        <div>• <b>Estimasi Menuju Top Berikutnya:</b> {stock_top} Bar</div>
                        <div>• <b>Phase Age:</b> {stock_phase_age} Bar ({stock_phase_age_pct:.0f}%)</div>
                        <div>• <b>Status Posisi Siklus:</b> {stock_status}</div>
                        <div>• <b>8-Phase Cycle:</b> {stock_res["phase"]} ({stock_res["phase_confidence"]:.0f}%)</div>
                        <div>• <b>FFT / Hilbert / Autocorr:</b> {stock_cycle_info.get("fft_period", "-")} / {stock_cycle_info.get("hilbert_period", "-")} / {stock_cycle_info.get("autocorr_period", "-")}</div>
                        <div>• <b>Weighted Composite:</b> {stock_cycle_info.get("weighted_period", stock_period)} bars</div>
                        <div>• <b>Cycle Reliability:</b> {stock_cycle_info.get("cycle_reliability", np.nan):.0f}%</div>
                        <div>• <b>Cycle Gate Reason:</b> {stock_cycle_info.get("cycle_gate_reason", "OK")}</div>
                        <div>• <b>Detrend Method:</b> {stock_cycle_info.get('detrend_method', 'HighPass+TailHilbert')}</div>
                        <div>• <b>Macro Gate:</b> {'ON' if stock_res.get('macro_gate_ok', True) else 'OFF'} ({stock_res.get('macro_phase', 'Unknown')})</div>
                        <div>• <b>Macro Gate Reason:</b> {stock_res.get('macro_gate_reason', 'OK')}</div>
                        <div>• <b>Trend Quality:</b> {'OK' if stock_res.get('trend_ok', False) else 'Weak'}</div>
                    </div>
                </div>
                """
                bench_html = f"""
                <div style="background:linear-gradient(180deg, rgba(255,248,230,1) 0%, rgba(248,238,210,1) 100%); padding:22px; border-radius:18px; border:1px solid rgba(0,0,0,0.05); box-shadow:0 8px 24px rgba(0,0,0,0.04);">
                    <div style="font-size:1.15rem; font-weight:700; color:#8a4b00; margin-bottom:18px;">Siklus Makro Komposit (IHSG)</div>
                    <div style="font-size:1.02rem; color:#8a4b00; line-height:2;">
                        <div>• <b>Periode Siklus Dominan:</b> {bench_period if bench_period is not None else '-'} Hari Bursa</div>
                        <div>• <b>Estimasi Sisa Waktu Menuju Bottom berikutnya:</b> {bench_ttb if bench_ttb is not None else '-'} Bar</div>
                        <div>• <b>Status Posisi Siklus Makro:</b> {bench_status}</div>
                        <div>• <b>FFT / Hilbert / Autocorr:</b> {bench_cycle_info.get("fft_period", "-")} / {bench_cycle_info.get("hilbert_period", "-")} / {bench_cycle_info.get("autocorr_period", "-")}</div>
                        <div>• <b>Weighted Composite:</b> {bench_cycle_info.get("weighted_period", bench_period if bench_period is not None else '-') } bars</div>
                        <div>• <b>Estimasi Menuju Top Berikutnya:</b> {bench_top} Bar</div>
                        <div>• <b>Phase Age:</b> {bench_phase_age} Bar ({bench_phase_age_pct:.0f}%)</div>
                        <div>• <b>Macro Score:</b> {macro_context.get('macro_score', np.nan):.0f}%</div>
                        <div>• <b>Macro Gate:</b> {'ON' if macro_context.get('macro_gate_ok', True) else 'OFF'}</div>
                        <div>• <b>Macro Gate Reason:</b> {macro_context.get('macro_gate_reason', 'OK')}</div>
                        <div>• <b>Detrend Method:</b> {bench_cycle_info.get('detrend_method', 'ZLEMA')}</div>
                        <div>• <b>Trend Lag:</b> {bench_cycle_info.get('trend_lag_bars', '-')} Bar</div>
                    </div>
                </div>
                """
                st.markdown(
                    f"""
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:18px; margin: 18px 0 8px 0;">
                        {stock_html}
                        {bench_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                ctop1, ctop2, ctop3, ctop4 = st.columns(4)
                ctop1.metric("Decision", stock_res["decision"])
                ctop2.metric("Score", f"{stock_res['score']:.2f}")
                ctop3.metric("Close", f"Rp {stock_res['close']:,.0f}")
                ctop4.metric("Phase", stock_res["phase"])

                ctop5, ctop6, ctop7, ctop8 = st.columns(4)
                ctop5.metric("Smart Money", f"{stock_res['smart_money_score']:.0f}")
                ctop6.metric("Fundamental", f"{fundamental.get('fundamental_score', np.nan):.0f}" if pd.notna(fundamental.get('fundamental_score', np.nan)) else "n/a")
                ctop7.metric("PEG", f"{fundamental.get('peg_ratio', np.nan):.2f}" if pd.notna(fundamental.get('peg_ratio', np.nan)) else "n/a")
                ctop8.metric("Grade", fundamental.get("fundamental_grade", "n/a"))

                cmid1, cmid2, cmid3, cmid4 = st.columns(4)
                cmid1.metric("Unicorn", stock_res.get("unicorn_setup_status", "n/a"))
                cmid2.metric("RSI14", f"{stock_res['rsi']:.2f}")
                cmid3.metric("ADX14", f"{stock_res['adx']:.2f}" if pd.notna(stock_res["adx"]) else "n/a")
                cmid4.metric("Phase Confidence", f"{stock_res['phase_confidence']:.0f}%")

                left, right = st.columns([1, 1])
                with left:
                    st.subheader("Time Analysis - Stock")
                    st.write(f"**Dominant cycle:** `{stock_res['dominant_period']} bars`")
                    st.write(f"**FFT / Hilbert / Autocorr:** `{stock_cycle_info.get('fft_period', '-')}` / `{stock_cycle_info.get('hilbert_period', '-')}` / `{stock_cycle_info.get('autocorr_period', '-')}`")
                    st.write(f"**Weighted composite:** `{stock_cycle_info.get('weighted_period', stock_res['dominant_period'])} bars`")
                    st.write(f"**Detrend method:** `{stock_cycle_info.get('detrend_method', 'ZLEMA')}` | lag `{stock_cycle_info.get('trend_lag_bars', '-')}` bars")
                    st.write(f"**Time to next bottom:** `{stock_res['time_to_bottom']} bars`")
                    st.write(f"**Time to next top:** `{stock_res.get('time_to_top', np.nan)} bars`")
                    st.write(f"**Phase age:** `{stock_res.get('phase_age_bars', np.nan)} bars` ({stock_res.get('phase_age_pct', np.nan):.0f}%)")
                    st.write(f"**Cycle status:** `{stock_status}`")
                    st.write(f"**8-Phase:** `{stock_res['phase']}`")
                    st.write(f"**Phase confidence:** `{stock_res['phase_confidence']:.0f}%`")
                    st.write(f"**Phase reason:** {stock_res['phase_reason']}")
                    st.write(f"**Reversal signals:** `{stock_res['reversal_hits']}`")
                    st.write(f"**OBV trend:** `{stock_res['obv_trend']}`")
                    st.write(f"**CMF20 / MFI14:** `{stock_res['cmf20']:.2f}` / `{stock_res['mfi14']:.2f}`")
                    st.write(f"**Stoch K/D:** `{stock_res['stoch_k']:.2f}` / `{stock_res['stoch_d']:.2f}`")
                    st.write(f"**PEG:** `{stock_res.get('peg_ratio', np.nan):.2f}`" if pd.notna(stock_res.get("peg_ratio", np.nan)) else "**PEG:** n/a")
                    st.write(f"**SMC:** FVG `{stock_res['fvg_present']}` | OB `{stock_res['ob_present']}` | Unicorn `{stock_res.get('unicorn_setup', False)}` | Sniper `{stock_res.get('unicorn_sniper', False)}`")
                    st.write(f"**Bandarmology input:** `{bandarmology_mode}`")

                with right:
                    st.subheader("Recommendation")
                    exec_status = str(stock_res.get("execution_status", _effective_entry_plan(stock_res).get("execution_status", "n/a")))
                    if stock_res["decision"] in {"BUY", "STRONG BUY"} and exec_status == "EXECUTION_READY":
                        st.success("Saham layak dibeli menurut filter saat ini.")
                        st.write(f"**Recommended entry:** `Rp {stock_res['entry_price']:,.0f}`")
                        st.write(f"**Recommended stoploss:** `Rp {stock_res['stop_price']:,.0f}`")
                        rr_risk = stock_res["entry_price"] - stock_res["stop_price"]
                        st.write(f"**Risk per share:** `Rp {rr_risk:,.0f}`")
                        tp_plan = _effective_entry_plan(stock_res)
                        tp1 = tp_plan.get("target_1", np.nan)
                        tp2 = tp_plan.get("target_2", np.nan)
                        if pd.notna(tp1) and pd.notna(tp2):
                            st.write(f"**Take profit T1/T2:** `Rp {tp1:,.0f}` / `Rp {tp2:,.0f}`")
                        elif pd.notna(tp1):
                            st.write(f"**Take profit target:** `Rp {tp1:,.0f}`")
                        else:
                            tp_price = stock_res["entry_price"] + take_profit_1_atr_local * float(stock_res["last"]["ATR14"])
                            st.write(f"**Take profit target:** `Rp {tp_price:,.0f}`")
                    elif exec_status == "WATCHLIST_ENTRY":
                        st.warning("Setup valid tetapi belum executable. Ini non-executable candidate.")
                        st.write(f"**Planned entry:** `Rp {stock_res['entry_price']:,.0f}`")
                        st.write(f"**Planned stoploss:** `Rp {stock_res['stop_price']:,.0f}`")
                        tp_plan = _effective_entry_plan(stock_res)
                        tp1 = tp_plan.get("target_1", np.nan)
                        tp2 = tp_plan.get("target_2", np.nan)
                        if pd.notna(tp1) and pd.notna(tp2):
                            st.write(f"**Take profit T1/T2:** `Rp {tp1:,.0f}` / `Rp {tp2:,.0f}`")
                        st.caption(f"Execution status: {exec_status} | {stock_res.get('execution_status_reason', 'n/a')}")
                    else:
                        st.warning("Belum layak beli. Tunggu reversal / struktur membaik.")
                        st.write("Entry/stoploss tidak ditampilkan karena belum memenuhi kriteria beli.")

                st.markdown("---")
                fig = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    row_heights=[0.45, 0.15, 0.20, 0.20],
                    subplot_titles=(
                        f"{deep_ticker} Price Action",
                        "Reversal / SMC / OBV Signals",
                        "Relative Strength vs Benchmark",
                        "Volume",
                    ),
                )

                fig.add_trace(
                    go.Candlestick(
                        x=stock.index,
                        open=stock["Open"],
                        high=stock["High"],
                        low=stock["Low"],
                        close=stock["Close"],
                        name="Price",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(go.Scatter(x=stock.index, y=stock["EMA20"], name="EMA20", mode="lines"), row=1, col=1)
                fig.add_trace(go.Scatter(x=stock.index, y=stock["EMA50"], name="EMA50", mode="lines"), row=1, col=1)
                fig.add_trace(go.Scatter(x=stock.index, y=stock["EMA200"], name="EMA200", mode="lines"), row=1, col=1)

                fvg_df = stock[stock["Bullish_FVG"]].tail(5)
                for idx, _ in fvg_df.iterrows():
                    loc = stock.index.get_loc(idx)
                    if loc >= 2:
                        fig.add_shape(
                            type="rect",
                            x0=idx,
                            x1=stock.index[-1],
                            y0=float(stock["High"].iloc[loc - 2]),
                            y1=float(stock["Low"].iloc[loc]),
                            fillcolor="rgba(0, 255, 0, 0.08)",
                            line=dict(width=0),
                            row=1,
                            col=1,
                        )

                ob_df = stock[stock["Bullish_OB"]].tail(5)
                for idx, _ in ob_df.iterrows():
                    loc = stock.index.get_loc(idx)
                    if loc >= 1:
                        fig.add_shape(
                            type="rect",
                            x0=stock.index[loc - 1],
                            x1=stock.index[-1],
                            y0=float(stock["Low"].iloc[loc - 1]),
                            y1=float(stock["High"].iloc[loc - 1]),
                            fillcolor="rgba(255, 165, 0, 0.10)",
                            line=dict(width=0),
                            row=1,
                            col=1,
                        )

                unicorn_df = stock[stock["Unicorn_Setup"]].tail(5)
                for idx, _ in unicorn_df.iterrows():
                    loc = stock.index.get_loc(idx)
                    if loc >= 2:
                        fig.add_shape(
                            type="rect",
                            x0=idx,
                            x1=stock.index[-1],
                            y0=float(stock["FVG_Bottom"].iloc[loc]),
                            y1=float(stock["FVG_Top"].iloc[loc]),
                            fillcolor="rgba(138, 43, 226, 0.10)",
                            line=dict(width=0),
                            row=1,
                            col=1,
                        )

                sig_names = [
                    "Bullish_Engulfing",
                    "Hammer",
                    "Inverted_Hammer",
                    "Morning_Star",
                    "EMA20_Reclaim",
                    "MACD_Bull_Cross",
                    "RSI_Bounce",
                    "Breakout_5D",
                ]
                for sig in sig_names:
                    y = stock["Low"] * (0.995 if sig in ["Hammer", "Inverted_Hammer"] else 1.005)
                    fig.add_trace(
                        go.Scatter(
                            x=stock.index,
                            y=np.where(stock[sig], y, np.nan),
                            mode="markers",
                            name=sig,
                        ),
                        row=2,
                        col=1,
                    )

                fig.add_trace(go.Scatter(x=stock.index, y=stock["OBV"], name="OBV", mode="lines"), row=2, col=1)

                if show_benchmark_local and not bench.empty:
                    rs_ratio = compute_relative_strength(stock["Close"], bench["Close"])
                    fig.add_trace(go.Scatter(x=rs_ratio.index, y=rs_ratio, name="Stock/Benchmark", mode="lines"), row=3, col=1)
                    fig.add_trace(go.Scatter(x=bench.index, y=bench["Close"], name=f"Benchmark {benchmark_symbol_local}", mode="lines"), row=3, col=1)
                else:
                    fig.add_trace(go.Scatter(x=stock.index, y=stock["RSI14"], name="RSI14", mode="lines"), row=3, col=1)

                fig.add_trace(go.Bar(x=stock.index, y=stock["Volume"], name="Daily Volume"), row=4, col=1)
                fig.add_trace(go.Scatter(x=stock.index, y=stock["VOL_SMA20"], name="Vol SMA20", mode="lines"), row=4, col=1)

                if np.isfinite(float(stock_last["Close"])):
                    fig.add_hline(y=float(stock_last["Close"]), line_width=1.2, line_dash="dash", annotation_text="Current", row=1, col=1)
                if stock_res["decision"] in {"BUY", "STRONG BUY"} and np.isfinite(stock_res["stop_price"]):
                    fig.add_hline(y=float(stock_res["stop_price"]), line_width=1.2, line_dash="dash", annotation_text="Stop", row=1, col=1)
                if stock_res["decision"] in {"BUY", "STRONG BUY"} and np.isfinite(stock_res["entry_price"]):
                    fig.add_hline(y=float(stock_res["entry_price"]), line_width=1.2, line_dash="dash", annotation_text="Entry", row=1, col=1)
                if stock_res["decision"] in {"BUY", "STRONG BUY"} and np.isfinite(stock_res.get("target_1", np.nan)):
                    fig.add_hline(y=float(stock_res["target_1"]), line_width=1.0, line_dash="dot", annotation_text="TP1", row=1, col=1)
                if stock_res["decision"] in {"BUY", "STRONG BUY"} and np.isfinite(stock_res.get("target_2", np.nan)):
                    fig.add_hline(y=float(stock_res["target_2"]), line_width=1.0, line_dash="dot", annotation_text="TP2", row=1, col=1)

                fig.update_layout(height=980, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=True)
                st.plotly_chart(fig, width="stretch")
                st.markdown("---")
                st.subheader("Entry Plan & Risk")
                st.caption(f"Execution status: {stock_res.get('execution_status', 'n/a')} | {stock_res.get('execution_status_reason', 'n/a')}")
                plan_cols = st.columns(4)
                plan = _effective_entry_plan(stock_res)
                plan_eta = _estimate_entry_eta(stock_res, plan)
                plan_cols[0].metric(
                    "Projected Entry",
                    f"Rp {plan.get('entry_price_plan', np.nan):,.0f}" if pd.notna(plan.get("entry_price_plan", np.nan)) else "n/a",
                    f"Zone {plan.get('entry_zone_low', np.nan):,.0f} - {plan.get('entry_zone_high', np.nan):,.0f}" if pd.notna(plan.get("entry_zone_low", np.nan)) and pd.notna(plan.get("entry_zone_high", np.nan)) else "Zone n/a",
                )
                plan_cols[1].metric(
                    "Stop Loss",
                    f"Rp {plan.get('stop_loss_plan', np.nan):,.0f}" if pd.notna(plan.get("stop_loss_plan", np.nan)) else "n/a",
                    f"Risk / sh. Rp {plan.get('risk_per_share', np.nan):,.0f}" if pd.notna(plan.get("risk_per_share", np.nan)) else "Risk n/a",
                )
                plan_cols[2].metric(
                    "Target 1",
                    f"Rp {plan.get('target_1', np.nan):,.0f}" if pd.notna(plan.get("target_1", np.nan)) else "n/a",
                    f"RR {plan.get('risk_reward_1', np.nan):.2f}" if pd.notna(plan.get('risk_reward_1', np.nan)) else "RR n/a",
                )
                plan_cols[3].metric(
                    "Target 2",
                    f"Rp {plan.get('target_2', np.nan):,.0f}" if pd.notna(plan.get("target_2", np.nan)) else "n/a",
                    f"RR {plan.get('risk_reward_2', np.nan):.2f}" if pd.notna(plan.get('risk_reward_2', np.nan)) else "RR n/a",
                )
                plan_table = pd.DataFrame(
                    [
                        {"Metric": "Plan Reason", "Value": plan.get("plan_reason", "n/a")},
                        {"Metric": "Entry Zone Low", "Value": f"Rp {plan.get('entry_zone_low', np.nan):,.0f}" if pd.notna(plan.get("entry_zone_low", np.nan)) else "n/a"},
                        {"Metric": "Entry Zone High", "Value": f"Rp {plan.get('entry_zone_high', np.nan):,.0f}" if pd.notna(plan.get("entry_zone_high", np.nan)) else "n/a"},
                        {"Metric": "Entry Trigger", "Value": plan.get("entry_trigger", "n/a")},
                        {"Metric": "Entry Price", "Value": f"Rp {plan.get('entry_price_plan', np.nan):,.0f}" if pd.notna(plan.get("entry_price_plan", np.nan)) else "n/a"},
                        {"Metric": "Stop Loss", "Value": f"Rp {plan.get('stop_loss_plan', np.nan):,.0f}" if pd.notna(plan.get("stop_loss_plan", np.nan)) else "n/a"},
                        {"Metric": "Target 1", "Value": f"Rp {plan.get('target_1', np.nan):,.0f}" if pd.notna(plan.get("target_1", np.nan)) else "n/a"},
                        {"Metric": "Target 2", "Value": f"Rp {plan.get('target_2', np.nan):,.0f}" if pd.notna(plan.get("target_2", np.nan)) else "n/a"},
                        {"Metric": "RR 1", "Value": f"{plan.get('risk_reward_1', np.nan):.2f}" if pd.notna(plan.get("risk_reward_1", np.nan)) else "n/a"},
                        {"Metric": "RR 2", "Value": f"{plan.get('risk_reward_2', np.nan):.2f}" if pd.notna(plan.get("risk_reward_2", np.nan)) else "n/a"},
                        {"Metric": "Fill Prob", "Value": f"{plan.get('setup_fill_probability', stock_res.get('setup_fill_probability', np.nan)):.1f}%" if pd.notna(plan.get('setup_fill_probability', stock_res.get('setup_fill_probability', np.nan))) else "n/a"},
                        {"Metric": "ETA Entry", "Value": plan.get("entry_eta_label", plan_eta.get("entry_eta_label", "n/a"))},
                        {"Metric": "ETA Range", "Value": (
                            f"{plan.get('entry_eta_range_days', plan_eta.get('entry_eta_range_days', (np.nan, np.nan)))[0]:.1f} - "
                            f"{plan.get('entry_eta_range_days', plan_eta.get('entry_eta_range_days', (np.nan, np.nan)))[1]:.1f} hari"
                            if isinstance(plan.get('entry_eta_range_days', plan_eta.get('entry_eta_range_days', (np.nan, np.nan))), tuple)
                            and len(plan.get('entry_eta_range_days', plan_eta.get('entry_eta_range_days', (np.nan, np.nan)))) == 2
                            and pd.notna(plan.get('entry_eta_range_days', plan_eta.get('entry_eta_range_days', (np.nan, np.nan)))[0])
                            and pd.notna(plan.get('entry_eta_range_days', plan_eta.get('entry_eta_range_days', (np.nan, np.nan)))[1])
                            else "n/a"
                        )},
                        {"Metric": "Upside to TP1", "Value": f"{plan.get('upside_to_t1_pct', np.nan):.2f}%" if pd.notna(plan.get('upside_to_t1_pct', np.nan)) else "n/a"},
                        {"Metric": "Upside to TP2", "Value": f"{plan.get('upside_to_t2_pct', np.nan):.2f}%" if pd.notna(plan.get('upside_to_t2_pct', np.nan)) else "n/a"},
                    ]
                )
                st.dataframe(plan_table, width="stretch", hide_index=True)

                st.markdown("---")
                st.subheader("Detail indikator")
                detail_cols = st.columns(3)
                detail_cols[0].write(f"**Score:** `{stock_res['score']:.2f}`")
                detail_cols[0].write(f"**Core score:** `{stock_res['core_score']:.2f}`")
                detail_cols[0].write(f"**Smart money score:** `{stock_res['smart_money_score']:.2f}`")
                detail_cols[0].write(f"**Decision:** `{stock_res['decision']}`")
                detail_cols[0].write(f"**Dominant cycle:** `{stock_res['dominant_period']} bars`")
                detail_cols[0].write(f"**Time to top / bottom:** `{stock_res.get('time_to_top', np.nan)} / {stock_res['time_to_bottom']} bars`")
                detail_cols[0].write(f"**ETA entry:** `{plan.get('entry_eta_label', plan_eta.get('entry_eta_label', 'n/a'))}`")
                detail_cols[1].write(f"**FVG:** `{stock_res['fvg_present']}`")
                detail_cols[1].write(f"**Order Block:** `{stock_res['ob_present']}`")
                detail_cols[1].write(f"**Reversal score:** `{stock_res['reversal_score']}`")
                detail_cols[1].write(f"**Phase:** `{stock_res['phase']}`")
                detail_cols[1].write(f"**Phase age:** `{stock_res.get('phase_age_bars', np.nan)} bars` ({stock_res.get('phase_age_pct', np.nan):.0f}%)")
                detail_cols[1].write(f"**Cycle reliability:** `{stock_res.get('cycle_reliability', np.nan):.0f}%`")
                detail_cols[1].write(f"**Cycle gate:** `{stock_res.get('cycle_gate_reason', 'OK')}`")
                detail_cols[1].write(f"**Macro gate:** `{stock_res.get('macro_gate_reason', 'OK')}`")
                detail_cols[1].write(f"**Macro score:** `{stock_res.get('macro_score', np.nan):.0f}`")
                detail_cols[2].write(f"**Entry:** `{stock_res['entry_price']:.2f}`" if pd.notna(stock_res["entry_price"]) else "**Entry:** n/a")
                detail_cols[2].write(f"**Stoploss:** `{stock_res['stop_price']:.2f}`" if pd.notna(stock_res["stop_price"]) else "**Stoploss:** n/a")
                detail_cols[2].write(f"**OBV trend:** `{stock_res['obv_trend']}`")
                detail_cols[2].write(f"**Phase confidence:** `{stock_res['phase_confidence']:.0f}%`")
                detail_cols[2].write(f"**PEG:** `{fundamental.get('peg_ratio', np.nan):.2f}`" if pd.notna(fundamental.get("peg_ratio", np.nan)) else "**PEG:** n/a")
                detail_cols[2].write(f"**Revenue QoQ:** `{format_growth_percent(fundamental.get('revenue_growth_qoq', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Revenue YoY:** `{format_growth_percent(fundamental.get('revenue_growth_yoy', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Revenue Annual YoY:** `{format_growth_percent(fundamental.get('revenue_growth_annual_yoy', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Revenue Y/Y Acceleration:** `{format_growth_percent(fundamental.get('revenue_yoy_acceleration', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Revenue Seasonal QoQ Divergence:** `{format_growth_percent(fundamental.get('revenue_seasonal_qoq_divergence', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Revenue growth period:** `{fundamental.get('revenue_growth_period', 'n/a')}`")
                detail_cols[2].write(f"**Revenue growth basis:** `{fundamental.get('revenue_growth_basis', 'n/a')}`")
                detail_cols[2].write(f"**Revenue growth source:** `{fundamental.get('revenue_growth_source', 'n/a')}`")
                detail_cols[2].write(f"**Earnings QoQ:** `{format_growth_percent(fundamental.get('earnings_growth_qoq', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Earnings YoY:** `{format_growth_percent(fundamental.get('earnings_growth_yoy', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Earnings Annual YoY:** `{format_growth_percent(fundamental.get('earnings_growth_annual_yoy', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Earnings Y/Y Acceleration:** `{format_growth_percent(fundamental.get('earnings_yoy_acceleration', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Earnings Seasonal QoQ Divergence:** `{format_growth_percent(fundamental.get('earnings_seasonal_qoq_divergence', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Earnings growth period:** `{fundamental.get('earnings_growth_period', 'n/a')}`")
                detail_cols[2].write(f"**Earnings growth basis:** `{fundamental.get('earnings_growth_basis', 'n/a')}`")
                detail_cols[2].write(f"**Earnings growth source:** `{fundamental.get('earnings_growth_source', 'n/a')}`")
                detail_cols[2].write(f"**Expected Revenue Next Q:** `{format_growth_percent(future_context.get('expected_revenue_growth_next_q', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Expected EPS Next Q:** `{format_growth_percent(future_context.get('expected_eps_growth_next_q', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Expected Margin Next Q:** `{format_growth_percent(future_context.get('expected_margin_next_q', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Current fundamental grade:** `{fundamental.get('fundamental_grade', 'n/a')}`")
                detail_cols[2].write(f"**Current fundamental score:** `{fundamental.get('fundamental_score', np.nan):.2f}`" if pd.notna(fundamental.get("fundamental_score", np.nan)) else "**Current fundamental score:** n/a")
                detail_cols[2].write("---")
                detail_cols[2].write(f"**Future fundamental grade:** `{stock_res.get('future_fundamental_grade', 'n/a')}`")
                detail_cols[2].write(f"**Future fundamental score:** `{stock_res.get('future_fundamental_score', np.nan):.2f}`" if pd.notna(stock_res.get("future_fundamental_score", np.nan)) else "**Future fundamental score:** n/a")
                detail_cols[2].write(f"**Future direction:** `{stock_res.get('future_fundamental_direction', 'n/a')}`")
                detail_cols[2].write(f"**Future confidence:** `{stock_res.get('future_fundamental_confidence', np.nan):.0f}%`" if pd.notna(stock_res.get("future_fundamental_confidence", np.nan)) else "**Future confidence:** n/a")
                detail_cols[2].write(f"**Future phase:** `{stock_res.get('future_fundamental_phase', 'Unknown')}`")
                detail_cols[2].write(f"**Future reason:** `{stock_res.get('future_fundamental_reason', 'n/a')}`")
                if pd.notna(fundamental.get("fundamental_score", np.nan)) and pd.notna(stock_res.get("future_fundamental_score", np.nan)):
                    divergence = float(stock_res.get("future_fundamental_score", np.nan)) - float(fundamental.get("fundamental_score", np.nan))
                    detail_cols[2].write(f"**Score delta:** `{divergence:+.2f}` pts")
                detail_cols[2].write(f"**Notes:** `{stock_res['notes']}`")
        else:
            st.info("Masukkan ticker lalu klik **Analyze ticker** untuk membuka deep dive.")

    analysis = st.session_state.get("ifs_analysis", {})
    ifs_context = analysis.get("ifs_context", {})
    stock_res = analysis.get("stock_res", {})
    fundamental = analysis.get("fundamental", {})
    future_context = analysis.get("future_context", {})
    stock_df = analysis.get("stock_df", pd.DataFrame())
    bench_df = analysis.get("bench_df", pd.DataFrame())
    selected_symbol = analysis.get("symbol", normalize_ticker(ticker_input))

    with sub_overview:
        st.subheader("Overview Ranking")
        valid_results = st.session_state.get("global_valid_results", [])
        if valid_results:
            rows = []
            for r in valid_results:
                ifs_score = _safe_float(r.get("ifs_score"), np.nan)
                rows.append(
                    {
                        "Rank": 0,
                        "Ticker": r.get("symbol", "-"),
                        "IFS": round(ifs_score, 2) if pd.notna(ifs_score) else np.nan,
                        "Grade": r.get("ifs_grade", "n/a"),
                        "Forward": round(_safe_float(r.get("ifs_breakdown", {}).get("Forward Fundamental"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "Accum": round(_safe_float(r.get("ifs_breakdown", {}).get("Accumulation"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "RS": round(_safe_float(r.get("ifs_breakdown", {}).get("Relative Strength"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "Qual": round(_safe_float(r.get("ifs_breakdown", {}).get("Quality"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "Catalyst": round(_safe_float(r.get("ifs_breakdown", {}).get("Catalyst"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "Decision": r.get("decision", "-"),
                        "Tradeability": round(_safe_float(r.get("tradeability_score"), np.nan), 1) if pd.notna(r.get("tradeability_score", np.nan)) else np.nan,
                        "TradeGate": "OK" if r.get("tradeability_gate_ok", False) else "BLOCK",
                        "MarketStruct": round(_safe_float(r.get("market_structure_score"), np.nan), 1) if pd.notna(r.get("market_structure_score", np.nan)) else np.nan,
                        "ETA_Days": round(_safe_float(r.get("entry_eta_days", np.nan), np.nan), 1) if pd.notna(r.get("entry_eta_days", np.nan)) else np.nan,
                        "FillProb": round(_safe_float(r.get("setup_fill_probability", np.nan), np.nan), 1) if pd.notna(r.get("setup_fill_probability", np.nan)) else np.nan,
                    }
                )
            ov_df = pd.DataFrame(rows).sort_values(["IFS", "MarketStruct"], ascending=[False, False], na_position="last").reset_index(drop=True)
            ov_df["Rank"] = np.arange(1, len(ov_df) + 1)
            st.dataframe(ov_df.head(20), width="stretch", hide_index=True)
        else:
            st.info("Jalankan global scan terlebih dahulu agar ranking IFS muncul di sini.")

    with sub_factor:
        st.subheader(f"Factor Breakdown — {selected_symbol}")
        if ifs_context:
            factor_df = pd.DataFrame(
                [
                    {"Factor": k, "Score": round(v, 2)}
                    for k, v in ifs_context.get("ifs_breakdown", {}).items()
                ]
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("IFS", f'{ifs_context.get("ifs_score", np.nan):.2f}', ifs_context.get("ifs_grade", "n/a"))
            c2.metric("Forward Direction", ifs_context.get("ifs_detail", {}).get("future_direction", "n/a"))
            c3.metric("Confidence", f'{ifs_context.get("ifs_detail", {}).get("future_confidence", np.nan):.0f}%')
            st.dataframe(factor_df, width="stretch", hide_index=True)
        else:
            st.info("Klik Analyze ticker untuk melihat breakdown faktor IFS.")

    with sub_smart:
        st.subheader("Smart Money")
        if ifs_context is not None and stock_res is not None:
            sm_cols = st.columns(4)
            sm_cols[0].metric("Smart Money Score", f'{ifs_context.get("ifs_detail", {}).get("smart_money_score", np.nan):.2f}')
            sm_cols[1].metric("Tradeability", f'{stock_res.get("tradeability_score", np.nan):.2f}' if pd.notna(stock_res.get("tradeability_score", np.nan)) else "n/a")
            sm_cols[2].metric("CMF20", f'{stock_res.get("cmf20", np.nan):.2f}' if pd.notna(stock_res.get("cmf20", np.nan)) else "n/a")
            sm_cols[3].metric("Unicorn", stock_res.get("unicorn_setup_status", "n/a"))
            smart_table = pd.DataFrame(
                [
                    {"Metric": "OBV Trend", "Value": stock_res.get("obv_trend", "n/a")},
                    {"Metric": "OBV Slope", "Value": f'{stock_res.get("obv_slope10", np.nan):.2f}' if pd.notna(stock_res.get("obv_slope10", np.nan)) else "n/a"},
                    {"Metric": "CMF20", "Value": f'{stock_res.get("cmf20", np.nan):.2f}' if pd.notna(stock_res.get("cmf20", np.nan)) else "n/a"},
                    {"Metric": "MFI14", "Value": f'{stock_res.get("mfi14", np.nan):.2f}' if pd.notna(stock_res.get("mfi14", np.nan)) else "n/a"},
                    {"Metric": "Smart Money Score", "Value": f'{ifs_context.get("ifs_detail", {}).get("smart_money_score", np.nan):.2f}'},
                    {"Metric": "FVG Age (bars)", "Value": f'{stock_res.get("fvg_age_bars", np.nan):.0f}' if pd.notna(stock_res.get("fvg_age_bars", np.nan)) else "n/a"},
                    {"Metric": "FVG Status", "Value": stock_res.get("fvg_status", "n/a")},
                    {"Metric": "Unicorn", "Value": "YES" if stock_res.get("unicorn_setup", False) else "NO"},
                    {"Metric": "Unicorn Valid", "Value": "YES" if stock_res.get("unicorn_setup_valid", False) else "NO"},
                    {"Metric": "Unicorn State", "Value": stock_res.get("unicorn_setup_state", "n/a")},
                    {"Metric": "Unicorn Status", "Value": stock_res.get("unicorn_setup_status", "n/a")},
                    {"Metric": "Sniper", "Value": "YES" if stock_res.get("unicorn_sniper", False) else "NO"},
                    {"Metric": "Sniper Valid", "Value": "YES" if stock_res.get("unicorn_sniper_valid", False) else "NO"},
                    {"Metric": "Sniper State", "Value": stock_res.get("unicorn_sniper_state", "n/a")},
                    {"Metric": "Sniper Status", "Value": stock_res.get("unicorn_sniper_status", "n/a")},
                    {"Metric": "Accumulation Score", "Value": f'{ifs_context.get("ifs_detail", {}).get("accumulation_score", np.nan):.2f}'},
                ]
            )
            st.dataframe(smart_table, width="stretch", hide_index=True)
        else:
            st.info("Belum ada hasil analisis untuk Smart Money.")

    with sub_forward:
        st.subheader("Forward Fundamental")
        if ifs_context and fundamental:
            current_score = _safe_float(fundamental.get("fundamental_score"), np.nan)
            future_score = _safe_float(future_context.get("future_fundamental_score"), np.nan)
            divergence = future_score - current_score if np.isfinite(current_score) and np.isfinite(future_score) else np.nan

            cols = st.columns(4)
            cols[0].metric("Current Facts", f'{current_score:.2f}' if pd.notna(current_score) else "n/a", fundamental.get("fundamental_grade", "n/a"))
            cols[1].metric("Model Forecast", f'{future_score:.2f}' if pd.notna(future_score) else "n/a", future_context.get("future_fundamental_grade", "n/a"))
            cols[2].metric("Score Delta", format_score_delta(divergence), "Forecast - Current")
            cols[3].metric("Model Confidence", f'{future_context.get("future_fundamental_confidence", np.nan):.0f}%' if pd.notna(future_context.get("future_fundamental_confidence", np.nan)) else "n/a")

            c_left, c_right = st.columns(2)

            with c_left:
                st.markdown("**Current Facts**")
                current_table = pd.DataFrame(
                    [
                        {"Metric": "Revenue QoQ", "Value": format_growth_percent(fundamental.get("revenue_growth_qoq", np.nan), decimals=0)},
                        {"Metric": "Revenue YoY", "Value": format_growth_percent(fundamental.get("revenue_growth_yoy", np.nan), decimals=0)},
                        {"Metric": "Revenue Annual YoY", "Value": format_growth_percent(fundamental.get("revenue_growth_annual_yoy", np.nan), decimals=0)},
                        {"Metric": "Revenue Y/Y Acceleration", "Value": format_growth_percent(fundamental.get("revenue_yoy_acceleration", np.nan), decimals=0)},
                        {"Metric": "Revenue Seasonal QoQ Div.", "Value": format_growth_percent(fundamental.get("revenue_seasonal_qoq_divergence", np.nan), decimals=0)},
                        {"Metric": "Earnings QoQ", "Value": format_growth_percent(fundamental.get("earnings_growth_qoq", np.nan), decimals=0)},
                        {"Metric": "Earnings YoY", "Value": format_growth_percent(fundamental.get("earnings_growth_yoy", np.nan), decimals=0)},
                        {"Metric": "Earnings Annual YoY", "Value": format_growth_percent(fundamental.get("earnings_growth_annual_yoy", np.nan), decimals=0)},
                        {"Metric": "Earnings Y/Y Acceleration", "Value": format_growth_percent(fundamental.get("earnings_yoy_acceleration", np.nan), decimals=0)},
                        {"Metric": "Earnings Seasonal QoQ Div.", "Value": format_growth_percent(fundamental.get("earnings_seasonal_qoq_divergence", np.nan), decimals=0)},
                        {"Metric": "Revenue Period", "Value": fundamental.get("revenue_growth_period", "n/a")},
                        {"Metric": "Revenue Basis", "Value": fundamental.get("revenue_growth_basis", "n/a")},
                        {"Metric": "Revenue Source", "Value": fundamental.get("revenue_growth_source", "n/a")},
                        {"Metric": "Earnings Period", "Value": fundamental.get("earnings_growth_period", "n/a")},
                        {"Metric": "Earnings Basis", "Value": fundamental.get("earnings_growth_basis", "n/a")},
                        {"Metric": "Earnings Source", "Value": fundamental.get("earnings_growth_source", "n/a")},
                        {"Metric": "PEG", "Value": f'{fundamental.get("peg_ratio", np.nan):.2f}' if pd.notna(fundamental.get("peg_ratio", np.nan)) else "n/a"},
                        {"Metric": "Current Fundamental Grade", "Value": fundamental.get("fundamental_grade", "n/a")},
                        {"Metric": "Fundamental Data Source", "Value": fundamental.get("fundamental_data_source", "n/a")},
                        {"Metric": "Data Quality Flag", "Value": fundamental.get("data_quality_flag", "n/a")},
                    ]
                )
                st.dataframe(current_table, width="stretch", hide_index=True)

            with c_right:
                st.markdown("**Model Forecast**")
                future_table = pd.DataFrame(
                    [
                        {"Metric": "Future Phase", "Value": future_context.get("future_phase", "Unknown")},
                        {"Metric": "Future Direction", "Value": future_context.get("future_fundamental_direction", "n/a")},
                        {"Metric": "Expected Revenue Next Q", "Value": format_growth_percent(future_context.get("expected_revenue_growth_next_q", np.nan), decimals=0)},
                        {"Metric": "Expected EPS Next Q", "Value": format_growth_percent(future_context.get("expected_eps_growth_next_q", np.nan), decimals=0)},
                        {"Metric": "Expected Margin Next Q", "Value": format_growth_percent(future_context.get("expected_margin_next_q", np.nan), decimals=0)},
                        {"Metric": "Future Reason", "Value": future_context.get("future_moat_reason", "n/a")},
                        {"Metric": "Future Macro Gate", "Value": future_context.get("future_macro_gate_reason", "OK")},
                        {"Metric": "Future Macro Adjusted", "Value": f'{future_context.get("future_macro_adjusted_score", np.nan):.2f}' if pd.notna(future_context.get("future_macro_adjusted_score", np.nan)) else "n/a"},
                    ]
                )
                st.dataframe(future_table, width="stretch", hide_index=True)

            st.markdown("**Explainability**")
            explain_table = pd.DataFrame(
                [
                    {"Component": "Forward Fundamental", "Value": f'{future_context.get("future_fundamental_score", np.nan):.2f}' if pd.notna(future_context.get("future_fundamental_score", np.nan)) else "n/a"},
                    {"Component": "Fundamental Momentum", "Value": f'{future_context.get("future_fundamental_momentum_score", np.nan):.2f}' if pd.notna(future_context.get("future_fundamental_momentum_score", np.nan)) else "n/a"},
                    {"Component": "Seasonal Anomaly", "Value": f'{future_context.get("future_seasonal_anomaly_score", np.nan):.2f}' if pd.notna(future_context.get("future_seasonal_anomaly_score", np.nan)) else "n/a"},
                    {"Component": "Inflection Score", "Value": f'{future_context.get("future_inflection_score", np.nan):.2f}' if pd.notna(future_context.get("future_inflection_score", np.nan)) else "n/a"},
                    {"Component": "Growth Proxy", "Value": f'{future_context.get("future_growth_proxy", np.nan):.2f}' if pd.notna(future_context.get("future_growth_proxy", np.nan)) else "n/a"},
                    {"Component": "Cash Flow Proxy", "Value": f'{future_context.get("future_cash_flow_proxy", np.nan):.2f}' if pd.notna(future_context.get("future_cash_flow_proxy", np.nan)) else "n/a"},
                    {"Component": "Balance Quality", "Value": f'{future_context.get("future_balance_quality", np.nan):.2f}' if pd.notna(future_context.get("future_balance_quality", np.nan)) else "n/a"},
                    {"Component": "Price Proxy", "Value": f'{future_context.get("future_price_proxy", np.nan):.2f}' if pd.notna(future_context.get("future_price_proxy", np.nan)) else "n/a"},
                    {"Component": "Cycle Support", "Value": f'{future_context.get("future_cycle_support", np.nan):.2f}' if pd.notna(future_context.get("future_cycle_support", np.nan)) else "n/a"},
                    {"Component": "Future Reliability", "Value": f'{future_context.get("future_reliability", np.nan):.2f}' if pd.notna(future_context.get("future_reliability", np.nan)) else "n/a"},
                ]
            )
            st.dataframe(explain_table, width="stretch", hide_index=True)
        else:
            st.info("Klik Analyze ticker untuk melihat forward fundamental.")
with tab3:
    st.subheader("Walk-Forward Lab")
    st.caption(
        "Upload CSV/ZIP OHLCV untuk menguji edge, lalu simpan hasil fold-level, summary, dan file export untuk audit ulang."
    )

    wf_files = st.file_uploader(
        "Upload OHLCV CSV / ZIP",
        type=["csv", "zip"],
        accept_multiple_files=True,
        key="wf_uploads",
        help="Bisa satu CSV per ticker, CSV gabungan dengan kolom ticker/symbol, atau ZIP berisi banyak CSV.",
    )

    wf_c1, wf_c2, wf_c3 = st.columns(3)
    wf_train_bars = int(wf_c1.number_input("Train bars", min_value=120, max_value=2000, value=252, step=21, key="wf_train_bars"))
    wf_test_bars = int(wf_c2.number_input("Test bars", min_value=21, max_value=500, value=63, step=21, key="wf_test_bars"))
    wf_step_bars = int(wf_c3.number_input("Step bars", min_value=21, max_value=500, value=63, step=21, key="wf_step_bars"))

    wf_c4, wf_c5, wf_c6 = st.columns(3)
    wf_min_trades = int(wf_c4.number_input("Min trades per fold", min_value=1, max_value=50, value=8, step=1, key="wf_min_trades"))
    wf_benchmark_symbol = wf_c5.text_input("Benchmark", value=GLOBAL_BENCHMARK_SYMBOL, key="wf_benchmark_symbol")
    wf_output_prefix = wf_c6.text_input("Output prefix", value="idx_walkforward", key="wf_output_prefix")

    run_wf_clicked = st.button("Run walk-forward test", type="primary", key="wf_run_walkforward")

    bundle = st.session_state.get("wf_uploaded_bundle", {})
    if wf_files:
        bundle = read_uploaded_ohlcv_bundle(wf_files)
        st.session_state["wf_uploaded_bundle"] = bundle

    if not bundle:
        st.info("Upload file OHLCV untuk memulai walk-forward. Format yang didukung: CSV tunggal, CSV gabungan, atau ZIP berisi banyak CSV.")
    else:
        st.write(f"Dataset terdeteksi: **{len(bundle)}**")
        bundle_labels = list(bundle.keys())
        selected_label = st.selectbox("Pilih dataset untuk diuji", bundle_labels, key="wf_selected_dataset")
        selected_df = bundle.get(selected_label, pd.DataFrame()).copy()

        if selected_df is not None and not selected_df.empty:
            st.caption(f"Kolom tersedia: {', '.join(map(str, selected_df.columns[:12]))}")
            st.dataframe(selected_df.head(10), width="stretch", hide_index=False)

            if run_wf_clicked:
                benchmark_df = load_ticker_data(wf_benchmark_symbol, months) if wf_benchmark_symbol else pd.DataFrame()
                with st.spinner(f"Menjalankan walk-forward untuk {selected_label}..."):
                    folds = walk_forward_test(
                        selected_df,
                        benchmark_df=benchmark_df,
                        benchmark_symbol=wf_benchmark_symbol,
                        train_bars=wf_train_bars,
                        test_bars=wf_test_bars,
                        step_bars=wf_step_bars,
                        min_trades=wf_min_trades,
                    )
                    summary = summarize_walk_forward(folds)
                    summary.update(
                        {
                            "dataset_label": selected_label,
                            "benchmark_symbol": wf_benchmark_symbol,
                            "train_bars": wf_train_bars,
                            "test_bars": wf_test_bars,
                            "step_bars": wf_step_bars,
                            "min_trades": wf_min_trades,
                        }
                    )
                    save_info = save_research_bundle(
                        str(RESEARCH_OUTPUT_DIR),
                        summary=summary,
                        folds=folds,
                        prefix=(wf_output_prefix or "idx_walkforward").strip().replace(" ", "_"),
                    )

                st.session_state["wf_last_summary"] = summary
                st.session_state["wf_last_folds"] = folds
                st.session_state["wf_last_save_info"] = save_info

                metrics = st.columns(5)
                metrics[0].metric("Fold", summary.get("fold_count", 0))
                metrics[1].metric("Avg Expectancy (R)", f"{summary.get('avg_test_expectancy_r', np.nan):.3f}" if pd.notna(summary.get("avg_test_expectancy_r", np.nan)) else "n/a")
                metrics[2].metric("Profit Factor", f"{summary.get('avg_test_profit_factor', np.nan):.2f}" if pd.notna(summary.get("avg_test_profit_factor", np.nan)) else "n/a")
                metrics[3].metric("Winrate", f"{summary.get('avg_test_winrate', np.nan) * 100:.1f}%" if pd.notna(summary.get("avg_test_winrate", np.nan)) else "n/a")
                metrics[4].metric("Max DD (R)", f"{summary.get('avg_test_max_drawdown_r', np.nan):.2f}" if pd.notna(summary.get("avg_test_max_drawdown_r", np.nan)) else "n/a")

                st.success(f"Hasil tersimpan ke: {save_info.get('bundle_dir', 'n/a')}")
                summary_df = pd.DataFrame([summary])
                st.dataframe(summary_df, width="stretch", hide_index=True)

                if not folds.empty:
                    st.markdown("**Fold results**")
                    st.dataframe(folds, width="stretch", hide_index=True)

                col_dl1, col_dl2, col_dl3 = st.columns(3)
                col_dl1.download_button(
                    "Download folds CSV",
                    data=folds.to_csv(index=False).encode("utf-8"),
                    file_name=f"{wf_output_prefix}_folds.csv",
                    mime="text/csv",
                )
                col_dl2.download_button(
                    "Download summary CSV",
                    data=summary_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{wf_output_prefix}_summary.csv",
                    mime="text/csv",
                )
                col_dl3.download_button(
                    "Download summary JSON",
                    data=json.dumps(summary, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                    file_name=f"{wf_output_prefix}_summary.json",
                    mime="application/json",
                )
        else:
            st.warning("Dataset terpilih kosong atau gagal dibaca.")

    st.markdown("---")
    st.subheader("Import Walk-Forward Results")
    wf_result_files = st.file_uploader(
        "Upload summary / folds / trades / ZIP bundle",
        type=["csv", "json", "zip"],
        accept_multiple_files=True,
        key="wf_result_uploads",
        help="Bisa upload hasil yang sudah diekspor dari tab ini untuk audit ulang atau review metrik.",
    )
    if wf_result_files:
        imported = read_uploaded_research_bundle(wf_result_files)
        st.session_state["wf_imported_result_bundle"] = imported

        imported_summary = imported.get("summary") or {}
        imported_folds = imported.get("folds", pd.DataFrame())
        imported_trades = imported.get("trades", pd.DataFrame())
        imported_label = imported.get("label") or "uploaded_bundle"

        st.caption(f"Bundle terdeteksi: {imported_label}")
        if imported_summary:
            st.dataframe(pd.DataFrame([imported_summary]), width="stretch", hide_index=True)
        else:
            st.info("Summary belum ditemukan pada file yang di-upload.")

        if isinstance(imported_folds, pd.DataFrame) and not imported_folds.empty:
            st.markdown("**Imported folds**")
            st.dataframe(imported_folds, width="stretch", hide_index=True)

        if isinstance(imported_trades, pd.DataFrame) and not imported_trades.empty:
            st.markdown("**Imported trades**")
            st.dataframe(imported_trades.head(100), width="stretch", hide_index=True)


with tab4:
    st.subheader("OHLCV Batch Downloader")
    st.caption("Upload universe CSV IDX, download OHLCV 1 tahun per ticker, lalu simpan bundle CSV + ZIP untuk walk-forward.")

    universe_file = st.file_uploader(
        "Upload universe CSV",
        type=["csv"],
        key="ohlcv_universe_upload",
        help="CSV berisi ticker IDX, idealnya 1 kolom seperti Ticker.",
    )

    d_c1, d_c2, d_c3 = st.columns(3)
    dl_period = d_c1.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1, key="ohlcv_period")
    dl_interval = d_c2.selectbox("Interval", ["1d", "1wk"], index=0, key="ohlcv_interval")
    dl_workers = int(d_c3.number_input("Max workers", min_value=1, max_value=12, value=4, step=1, key="ohlcv_workers"))

    d_c4, d_c5 = st.columns(2)
    dl_prefix = d_c4.text_input("Output prefix", value="idx_ohlcv", key="ohlcv_prefix")
    dl_auto_adjust = d_c5.checkbox("Auto-adjust prices", value=False, key="ohlcv_auto_adjust")

    run_dl_clicked = st.button("Download OHLCV batch", type="primary", key="ohlcv_run_download")

    if universe_file is None:
        st.info("Upload universe CSV dulu. File kamu bisa langsung memakai kolom Ticker atau kolom pertama akan dipakai otomatis.")
    else:
        try:
            universe_df = load_universe_df_from_csv(universe_file)
            tickers = extract_universe_tickers(universe_df)
            st.write(f"Universe terdeteksi: **{len(tickers)}** ticker")
            st.dataframe(universe_df.head(10), width="stretch", hide_index=True)

            if run_dl_clicked:
                with st.spinner("Mengunduh OHLCV dari Yahoo Finance..."):
                    batch = download_batch_idx_ohlcv(
                        tickers,
                        period=dl_period,
                        interval=dl_interval,
                        max_workers=dl_workers,
                        auto_adjust=dl_auto_adjust,
                    )
                    saved = save_batch_bundle(
                        batch["frames"],
                        out_dir=str(RESEARCH_OUTPUT_DIR / "ohlcv_downloads"),
                        prefix=(dl_prefix or "idx_ohlcv").strip().replace(" ", "_"),
                        combined_csv=True,
                        make_zip=True,
                        summary=batch["summary"],
                        results=batch["results"],
                    )

                st.session_state["ohlcv_last_batch"] = batch
                st.session_state["ohlcv_last_saved"] = saved

                summary_df = pd.DataFrame([batch["summary"]])
                st.dataframe(summary_df, width="stretch", hide_index=True)

                metrics = st.columns(4)
                metrics[0].metric("Downloaded", int(batch["summary"].get("tickers_downloaded", 0)))
                metrics[1].metric("Failed", int(batch["summary"].get("tickers_failed", 0)))
                metrics[2].metric("Total rows", int(batch["summary"].get("rows_total", 0)))
                metrics[3].metric("Bundle", Path(saved.get("bundle_dir", "")).name if saved.get("bundle_dir") else "n/a")

                if isinstance(batch.get("results"), pd.DataFrame) and not batch["results"].empty:
                    st.markdown("**Download results**")
                    st.dataframe(batch["results"], width="stretch", hide_index=True)

                bundle_dir = Path(saved["bundle_dir"]) if saved.get("bundle_dir") else None
                zip_path = Path(saved["zip_path"]) if saved.get("zip_path") else None
                combined_csv_path = Path(saved["combined_csv_path"]) if saved.get("combined_csv_path") else None
                summary_path = Path(saved["summary_path"]) if saved.get("summary_path") else None
                manifest_path = Path(saved["manifest_path"]) if saved.get("manifest_path") else None
                results_path = Path(saved["results_path"]) if saved.get("results_path") else None

                d1, d2, d3, d4 = st.columns(4)
                if zip_path and zip_path.exists():
                    d1.download_button(
                        "Download ZIP",
                        data=zip_path.read_bytes(),
                        file_name=zip_path.name,
                        mime="application/zip",
                    )
                if combined_csv_path and combined_csv_path.exists():
                    d2.download_button(
                        "Download combined CSV",
                        data=combined_csv_path.read_bytes(),
                        file_name=combined_csv_path.name,
                        mime="text/csv",
                    )
                if summary_path and summary_path.exists():
                    d3.download_button(
                        "Download summary JSON",
                        data=summary_path.read_bytes(),
                        file_name=summary_path.name,
                        mime="application/json",
                    )
                if manifest_path and manifest_path.exists():
                    d4.download_button(
                        "Download manifest JSON",
                        data=manifest_path.read_bytes(),
                        file_name=manifest_path.name,
                        mime="application/json",
                    )

                if results_path and results_path.exists():
                    st.download_button(
                        "Download results CSV",
                        data=results_path.read_bytes(),
                        file_name=results_path.name,
                        mime="text/csv",
                    )

                st.success(f"Bundle tersimpan di: {saved.get('bundle_dir', 'n/a')}")
        except Exception as exc:
            st.error(f"Gagal membaca universe CSV: {type(exc).__name__}: {exc}")


with tab5:
    st.subheader("💼 Live Portfolio / Execution Ledger")
    st.caption(
        "Ledger ini menyimpan posisi, order, fill, dan event ke storage persisten. "
        "Session state hanya dipakai untuk UI; reload tidak boleh menghapus catatan ledger."
    )

    pe.init_store()
    pe.ensure_account(account_id="default", label="Main", starting_capital=float(st.session_state.get("live_starting_capital", 100_000_000.0)))
    summary = pe.get_account_summary(account_id="default")

    st.info(summary["warning"])
    st.caption(f"Backend aktif: `{summary["backend"]}` | State path: `{summary["state_path"]}`")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", f'Rp {summary["equity"]:,.0f}')
    c2.metric("Cash", f'Rp {summary["cash"]:,.0f}')
    c3.metric("P/L", f'Rp {summary["total_pnl"]:,.0f}', f'Realized {summary["realized_pnl"]:,.0f}')
    c4.metric("Open Positions", f'{summary["open_positions"]}')

    with st.expander("⚙️ Account & Risk Settings", expanded=True):
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            live_starting_capital = st.number_input(
                "Starting capital (Rp)",
                min_value=1_000_000.0,
                value=float(st.session_state.get("live_starting_capital", 100_000_000.0)),
                step=1_000_000.0,
            )
        with s2:
            risk_pct = st.number_input(
                "Risk per trade (%)",
                min_value=0.1,
                max_value=5.0,
                value=float(st.session_state.get("live_risk_pct", 1.0)),
                step=0.1,
            )
        with s3:
            max_notional_pct = st.number_input(
                "Max notional / trade (%)",
                min_value=1.0,
                max_value=100.0,
                value=float(st.session_state.get("live_max_notional_pct", 20.0)),
                step=1.0,
            )
        with s4:
            lot_size = st.number_input(
                "Lot size",
                min_value=1,
                max_value=10_000,
                value=int(st.session_state.get("live_lot_size", 100)),
                step=1,
            )

        init_col, reset_col = st.columns(2)
        with init_col:
            if st.button("Initialize / update account", type="primary"):
                pe.ensure_account(account_id="default", label="Main", starting_capital=float(live_starting_capital))
                st.session_state["live_starting_capital"] = float(live_starting_capital)
                st.session_state["live_risk_pct"] = float(risk_pct)
                st.session_state["live_max_notional_pct"] = float(max_notional_pct)
                st.session_state["live_lot_size"] = int(lot_size)
                pe.record_event(
                    "account_settings_updated",
                    {
                        "starting_capital": float(live_starting_capital),
                        "risk_pct": float(risk_pct),
                        "max_notional_pct": float(max_notional_pct),
                        "lot_size": int(lot_size),
                    },
                )
                st.success("Account settings tersimpan.")
                st.rerun()
        with reset_col:
            if st.button("Refresh live ledger snapshot"):
                st.rerun()

    analysis_bundle = st.session_state.get("ifs_analysis", {})
    selected_symbol = str(analysis_bundle.get("symbol") or st.session_state.get("deep_selected_symbol") or "BMRI.JK")
    stock_res = analysis_bundle.get("stock_res", {}) or {}
    entry_plan = _effective_entry_plan(stock_res)
    st.markdown(f"**Selected ticker:** `{selected_symbol}`")
    st.caption("Order plan di bawah memakai projected entry zone dari hasil analisa terakhir Tab 2.")

    plan_entry = entry_plan.get("entry_price_plan", np.nan)
    plan_stop = entry_plan.get("stop_loss_plan", entry_plan.get("stop_price", np.nan))
    plan_tp1 = entry_plan.get("target_1", np.nan)
    plan_tp2 = entry_plan.get("target_2", np.nan)

    if pd.notna(plan_entry) and pd.notna(plan_stop):
        sizing = pe.estimate_position_size(
            cash=summary["cash"],
            entry_price=float(plan_entry),
            stop_price=float(plan_stop),
            risk_pct=float(risk_pct) / 100.0,
            max_notional_pct=float(max_notional_pct) / 100.0,
            lot_size=int(lot_size),
        )
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Suggested Qty", f"{int(sizing['qty']):,}")
        sc2.metric("Risk / Share", f"Rp {float(sizing['risk_per_share']):,.0f}" if np.isfinite(sizing["risk_per_share"]) else "n/a")
        sc3.metric("Risk Budget", f"Rp {float(sizing['risk_budget']):,.0f}")
        sc4.metric("Sizing Status", sizing.get("reason", "n/a"))
    else:
        sizing = {"qty": 0, "reason": "missing_entry_or_stop"}
        st.warning("Entry plan belum lengkap, jadi ukuran posisi tidak bisa dihitung.")

    order_tab, history_tab, backup_tab = st.tabs(["Order Builder", "Ledger", "Backup / Restore"])

    with order_tab:
        if not entry_plan:
            st.info("Analisa ticker dulu di Tab 2 agar order builder mendapat entry plan.")
        else:
            with st.form("live_order_form", clear_on_submit=False):
                o1, o2, o3 = st.columns(3)
                with o1:
                    side = st.selectbox("Side", ["BUY", "SELL"], index=0)
                    order_type = st.selectbox("Order type", ["LIMIT", "MARKET"], index=0)
                with o2:
                    qty_default = int(sizing["qty"]) if int(sizing.get("qty", 0) or 0) > 0 else 100
                    qty = st.number_input("Qty", min_value=1, step=100, value=max(1, qty_default))
                    limit_price = st.number_input(
                        "Limit / planned entry price",
                        min_value=0.0,
                        value=float(plan_entry) if pd.notna(plan_entry) else float(stock_res.get("close", 0.0)),
                        step=10.0,
                    )
                with o3:
                    stop_price = st.number_input(
                        "Stop price",
                        min_value=0.0,
                        value=float(plan_stop) if pd.notna(plan_stop) else 0.0,
                        step=10.0,
                    )
                    signal_hash = st.text_input("Signal hash", value=str(entry_plan.get("signal_hash", "")))

                tp_col1, tp_col2 = st.columns(2)
                with tp_col1:
                    tp1_input = st.number_input(
                        "TP1",
                        min_value=0.0,
                        value=float(plan_tp1) if pd.notna(plan_tp1) else 0.0,
                        step=10.0,
                    )
                with tp_col2:
                    tp2_input = st.number_input(
                        "TP2",
                        min_value=0.0,
                        value=float(plan_tp2) if pd.notna(plan_tp2) else 0.0,
                        step=10.0,
                    )

                notes = st.text_area(
                    "Notes",
                    value=f"{selected_symbol} | {entry_plan.get('plan_reason', 'scanner')} | tradeability={entry_plan.get('tradeability_reason', 'n/a')} | exec={entry_plan.get('execution_status', 'n/a')} | TP1={tp1_input if tp1_input > 0 else 'n/a'} | TP2={tp2_input if tp2_input > 0 else 'n/a'}",
                    height=110,
                )
                submit_order = st.form_submit_button("Save planned order", type="primary")

                if submit_order:
                    try:
                        order_id = pe.create_order(
                            symbol=selected_symbol,
                            side=side,
                            qty=int(qty),
                            order_type=order_type,
                            limit_price=float(limit_price) if order_type == "LIMIT" else None,
                            stop_price=float(stop_price) if stop_price > 0 else None,
                            target_1=float(tp1_input) if tp1_input > 0 else None,
                            target_2=float(tp2_input) if tp2_input > 0 else None,
                            status="PLANNED",
                            source="scanner",
                            signal_hash=signal_hash,
                            notes=notes,
                        )
                        pe.record_event(
                            "planned_order_saved",
                            {
                                "order_id": order_id,
                                "symbol": selected_symbol,
                                "side": side,
                                "qty": int(qty),
                                "order_type": order_type,
                                "limit_price": float(limit_price) if order_type == "LIMIT" else None,
                                "stop_price": float(stop_price) if stop_price > 0 else None,
                                "target_1": float(tp1_input) if tp1_input > 0 else None,
                                "target_2": float(tp2_input) if tp2_input > 0 else None,
                                "execution_status": stock_res.get("execution_status", "n/a"),
                            },
                            symbol=selected_symbol,
                        )
                        st.success(f"Order tersimpan: {order_id}")
                    except Exception as exc:
                        st.error(f"Gagal menyimpan order: {type(exc).__name__}: {exc}")

            st.markdown("#### Quick actions")
            q1, q2 = st.columns(2)
            with q1:
                if st.button("Save scanner signal only"):
                    try:
                        pe.record_event(
                            "scanner_signal",
                            {
                                "symbol": selected_symbol,
                                "decision": stock_res.get("decision", ""),
                                "score": float(stock_res.get("score", np.nan)) if pd.notna(stock_res.get("score", np.nan)) else None,
                                "tradeability": float(stock_res.get("tradeability_score", np.nan)) if pd.notna(stock_res.get("tradeability_score", np.nan)) else None,
                                "entry": float(plan_entry) if pd.notna(plan_entry) else None,
                                "stop": float(plan_stop) if pd.notna(plan_stop) else None,
                                "target_1": float(plan_tp1) if pd.notna(plan_tp1) else None,
                                "target_2": float(plan_tp2) if pd.notna(plan_tp2) else None,
                                "execution_status": stock_res.get("execution_status", "n/a"),
                            },
                            symbol=selected_symbol,
                        )
                        st.success("Signal disimpan ke ledger.")
                    except Exception as exc:
                        st.error(f"Gagal menyimpan signal: {type(exc).__name__}: {exc}")
            with q2:
                if st.button("Simulate paper fill on planned BUY"):
                    try:
                        sim = pe.simulate_limit_execution(
                            side="BUY",
                            order_price=float(plan_entry) if pd.notna(plan_entry) else float(stock_res.get("close", 0.0)),
                            open_price=float(stock_res.get("close", 0.0)),
                            high_price=float(stock_res.get("close", 0.0)),
                            low_price=float(stock_res.get("close", 0.0)),
                        )
                        st.write(sim)
                    except Exception as exc:
                        st.error(f"Simulasi gagal: {type(exc).__name__}: {exc}")

    with history_tab:
        pos_df = pe.list_positions()
        ord_df = pe.list_orders(limit=200)
        fill_df = pe.list_fills(limit=200)
        evt_df = pe.list_events(limit=50)

        st.markdown("#### Positions")
        if pos_df.empty:
            st.info("Belum ada posisi tersimpan.")
        else:
            show_pos = pos_df.copy()
            for col in ["avg_price", "mark_price", "stop_loss", "target_1", "target_2"]:
                if col in show_pos.columns:
                    show_pos[col] = pd.to_numeric(show_pos[col], errors="coerce")
            st.dataframe(show_pos, width="stretch", hide_index=True)

        st.markdown("#### Orders")
        if ord_df.empty:
            st.info("Belum ada order tersimpan.")
        else:
            st.dataframe(ord_df, width="stretch", hide_index=True)

        st.markdown("#### Fills")
        if fill_df.empty:
            st.info("Belum ada fill tersimpan.")
        else:
            st.dataframe(fill_df, width="stretch", hide_index=True)

        st.markdown("#### Events")
        if evt_df.empty:
            st.info("Belum ada event tersimpan.")
        else:
            st.dataframe(evt_df, width="stretch", hide_index=True)

        st.markdown("#### Trade Journal")
        journal_df = pe.list_trade_journal(limit=200)
        if journal_df.empty:
            st.info("Belum ada trade journal tersimpan.")
        else:
            show_journal = journal_df.copy()
            for col in ["score", "ifs_score", "catalyst_score", "tradeability_score", "entry_price", "stop_price", "target_1", "target_2", "risk_reward_1", "risk_reward_2", "value_traded_20d", "spread_proxy_20d", "gap_proxy_20d"]:
                if col in show_journal.columns:
                    show_journal[col] = pd.to_numeric(show_journal[col], errors="coerce")
            st.dataframe(show_journal, width="stretch", hide_index=True)

    with backup_tab:
        backup_payload = pe.export_state()
        backup_json = json.dumps(backup_payload, ensure_ascii=False, indent=2, default=str)
        st.download_button(
            "Download portfolio backup JSON",
            data=backup_json.encode("utf-8"),
            file_name=f"portfolio_backup_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

        st.markdown("#### Restore from backup JSON")
        st.caption("Backup JSON berguna sebagai cadangan manual. Untuk source of truth live, pakai Supabase/Postgres remote.")
        uploaded_backup = st.file_uploader("Upload JSON backup", type=["json"], key="portfolio_backup_upload")
        replace_existing = st.checkbox("Replace existing tables on restore", value=False)
        if uploaded_backup is not None and st.button("Import uploaded backup"):
            try:
                payload = json.loads(uploaded_backup.read().decode("utf-8"))
                counts = pe.import_state(payload, replace=replace_existing)
                st.success(f"Restore selesai: {counts}")
                st.rerun()
            except Exception as exc:
                st.error(f"Gagal restore backup: {type(exc).__name__}: {exc}")

