from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

def normalize_ticker(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if not s or s == "NAN":
        return ""
    if s.startswith("^"):
        return s
    return s if s.endswith(".JK") else f"{s}.JK"

def make_flow_score(flow_mode: str) -> float:
    mapping = {
        "Big Akumulasi": 95.0,
        "Small Akumulasi": 75.0,
        "Netral": 50.0,
        "Small Distribusi": 30.0,
        "Big Distribusi": 10.0,
    }
    return mapping.get(flow_mode, 50.0)

def map_flow_to_score(flow_mode: str) -> float:
    """Backward-compatible alias kept for older call sites."""
    return make_flow_score(flow_mode)

def load_ticker_data(symbol: str, months: int) -> pd.DataFrame:
    # Use UTC to minimize environment-specific differences between localhost and deploy.
    end = pd.Timestamp.utcnow().tz_localize(None)
    start = end - pd.DateOffset(months=months)

    base = str(symbol).strip()
    candidates = []
    if base:
        candidates.append(base)
        if base.endswith(".JK"):
            candidates.append(base[:-3])
        elif not base.startswith("^"):
            candidates.append(f"{base}.JK")

    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for candidate in candidates:
        for attempt in range(3):
            try:
                df = yf.download(
                    candidate,
                    period=f"{months}mo",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                if df is None or df.empty:
                    df = yf.download(
                        candidate,
                        start=start,
                        end=end,
                        auto_adjust=False,
                        progress=False,
                        threads=False,
                    )
            except Exception:
                df = pd.DataFrame()

            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    if any(col in df.columns.get_level_values(0) for col in ["Close", "Open", "High", "Low"]):
                        df.columns = df.columns.get_level_values(0)
                    else:
                        df.columns = df.columns.get_level_values(1)

                needed = {"Open", "High", "Low", "Close", "Volume"}
                if needed.issubset(set(df.columns)):
                    out = df.copy()
                    out = out.loc[:, ~out.columns.duplicated()].copy()
                    out = out.dropna().copy()
                    out = out[~out.index.duplicated(keep="last")].sort_index()
                    for col in needed:
                        out[col] = pd.to_numeric(out[col], errors="coerce")
                    out = out.dropna(subset=list(needed)).copy()
                    if not out.empty:
                        return out

            time.sleep(0.25 * (attempt + 1))

    return pd.DataFrame()

def _ticker_candidates(symbol: str) -> list[str]:
    base = str(symbol).strip().upper()
    if not base or base == "NAN":
        return []

    candidates: list[str] = []

    def add(candidate: str) -> None:
        candidate = str(candidate).strip().upper()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(base)
    if base.startswith("^"):
        return candidates

    if base.endswith(".JK"):
        add(base[:-3])
    else:
        add(f"{base}.JK")

    return candidates

def load_yf_info(symbol: str) -> dict:
    """Cached Yahoo Finance info fetch with candidate symbol retries.

    Some environments respond better to NCKL than NCKL.JK, while others need
    the reverse. This function tries both and also merges fast_info when
    available so we can still recover basic fields even if .info is partial.
    """
    base = str(symbol).strip()
    if not base:
        return {}

    for candidate in _ticker_candidates(base):
        try:
            ticker = yf.Ticker(candidate)
            info = {}
            try:
                info = ticker.get_info() or {}
            except Exception:
                try:
                    info = ticker.info or {}
                except Exception:
                    info = {}

            merged: dict = {}
            if isinstance(info, dict):
                merged.update(info)

            try:
                fast_info = getattr(ticker, "fast_info", None)
                if fast_info is not None:
                    fast_dict = dict(fast_info)
                    for key, value in fast_dict.items():
                        if key not in merged or merged.get(key) in (None, ""):
                            merged[key] = value
            except Exception:
                pass

            if merged:
                merged["_resolved_symbol"] = candidate
                return merged
        except Exception:
            continue

    return {}

def parse_universe_text(text: str) -> list[str]:
    tokens: list[str] = []
    for line in text.splitlines():
        line = line.strip().upper()
        if not line:
            continue
        parts = [p.strip().upper() for p in line.replace(";", ",").split(",")]
        tokens.extend([p for p in parts if p])

    cleaned = []
    for t in tokens:
        norm = normalize_ticker(t)
        if norm:
            cleaned.append(norm)
    return list(dict.fromkeys(cleaned))

def load_universe_from_csv(source) -> list[str]:
    if source is None:
        return []
    try:
        dfu = pd.read_csv(source)
    except Exception:
        return []

    if dfu.empty:
        return []

    ticker_col = next(
        (
            col
            for col in dfu.columns
            if str(col).strip().lower() in {"ticker", "symbol", "kode", "code", "stock", "saham"}
        ),
        dfu.columns[0],
    )

    vals = dfu[ticker_col].astype(str).str.upper().str.strip().tolist()
    out = []
    for v in vals:
        norm = normalize_ticker(v)
        if norm:
            out.append(norm)
    return list(dict.fromkeys(out))

