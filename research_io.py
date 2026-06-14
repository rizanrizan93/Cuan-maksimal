from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from data_engine import normalize_ticker


_DATE_COL_CANDIDATES = ["date", "datetime", "timestamp", "time", "tanggal"]
_TICKER_COL_CANDIDATES = ["ticker", "symbol", "kode", "code", "stock", "saham"]
_OHLCV_MAP = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Adj Close",
    "adj_close": "Adj Close",
    "adjusted close": "Adj Close",
    "volume": "Volume",
    "date": "Date",
    "datetime": "Date",
    "timestamp": "Date",
    "time": "Date",
}


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def _guess_symbol_from_name(name: str) -> str:
    stem = Path(str(name or "")).stem.upper().strip()
    if not stem:
        return ""
    m = re.search(r"([A-Z0-9^\.]{2,10})", stem)
    if not m:
        return normalize_ticker(stem)
    token = m.group(1).replace("_", "").replace("-", "")
    return normalize_ticker(token)


def _standardize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    rename_map = {}
    for col in out.columns:
        key = str(col).strip().lower()
        if key in _OHLCV_MAP:
            rename_map[col] = _OHLCV_MAP[key]
    if rename_map:
        out = out.rename(columns=rename_map)

    # Normalize common alternate names to the expected schema.
    if "Adj Close" not in out.columns and "AdjClose" in out.columns:
        out = out.rename(columns={"AdjClose": "Adj Close"})

    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"]).copy()
        out = out.set_index("Date")
    elif not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index, errors="coerce")
            out = out[~out.index.isna()].copy()
        except Exception:
            return pd.DataFrame()

    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(out.columns):
        # Try lower-case source schemas one more time after rename.
        lower_cols = {str(c).lower().strip(): c for c in out.columns}
        missing = [c for c in needed if c not in out.columns]
        if missing:
            remap = {lower_cols[m.lower()]: m for m in missing if m.lower() in lower_cols}
            if remap:
                out = out.rename(columns=remap)

    if not needed.issubset(out.columns):
        return pd.DataFrame()

    out = out.loc[:, ~out.columns.duplicated()].copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")].copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    if out.empty:
        return pd.DataFrame()

    if "Adj Close" in out.columns:
        out["Adj Close"] = pd.to_numeric(out["Adj Close"], errors="coerce")

    return out


def _split_combined_csv(df: pd.DataFrame, fallback_label: str) -> list[dict]:
    if df is None or df.empty:
        return []

    cols = [str(c).strip().lower() for c in df.columns]
    ticker_col = next((c for c in df.columns if str(c).strip().lower() in _TICKER_COL_CANDIDATES), None)

    if ticker_col is None:
        out = _standardize_ohlcv_frame(df)
        if out.empty:
            return []
        return [
            {
                "label": fallback_label,
                "symbol": _guess_symbol_from_name(fallback_label),
                "dataframe": out,
            }
        ]

    entries = []
    for raw_symbol, group in df.groupby(ticker_col):
        symbol = normalize_ticker(raw_symbol)
        if not symbol:
            symbol = _guess_symbol_from_name(raw_symbol)
        if not symbol:
            symbol = fallback_label
        g = group.drop(columns=[ticker_col]).copy()
        if any(str(c).strip().lower() in _DATE_COL_CANDIDATES for c in g.columns):
            date_col = next(c for c in g.columns if str(c).strip().lower() in _DATE_COL_CANDIDATES)
            g = g.rename(columns={date_col: "Date"})
        out = _standardize_ohlcv_frame(g)
        if out.empty:
            continue
        entries.append({"label": f"{fallback_label}::{symbol}", "symbol": symbol, "dataframe": out})

    return entries


def _read_csv_bytes(raw: bytes, label: str) -> list[dict]:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return []

    return _split_combined_csv(df, fallback_label=label)


def _read_zip_bytes(raw: bytes, label: str) -> list[dict]:
    entries: list[dict] = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            for member in zf.namelist():
                if member.endswith("/"):
                    continue
                if not member.lower().endswith(".csv"):
                    continue
                try:
                    data = zf.read(member)
                except Exception:
                    continue
                inner_label = f"{label}::{Path(member).stem}"
                entries.extend(_read_csv_bytes(data, inner_label))
    except Exception:
        return []
    return entries


def read_uploaded_ohlcv_bundle(uploaded_files: Iterable[object] | None) -> dict[str, pd.DataFrame]:
    """Read uploaded CSV/ZIP files into a label -> dataframe bundle.

    Each dataframe is standardized to OHLCV with a DatetimeIndex.
    CSVs containing a ticker/symbol column are split per ticker.
    """
    bundle: dict[str, pd.DataFrame] = {}
    if not uploaded_files:
        return bundle

    for item in uploaded_files:
        name = getattr(item, "name", "uploaded")
        label = Path(str(name)).stem
        try:
            raw = item.getvalue() if hasattr(item, "getvalue") else item.read()
        except Exception:
            continue
        if not raw:
            continue

        if str(name).lower().endswith(".zip"):
            entries = _read_zip_bytes(raw, label)
        else:
            entries = _read_csv_bytes(raw, label)

        for entry in entries:
            df = entry.get("dataframe")
            symbol = str(entry.get("symbol") or "").strip()
            lbl = str(entry.get("label") or label).strip()
            if df is None or df.empty:
                continue
            key = lbl if lbl not in bundle else f"{lbl}_{len(bundle) + 1}"
            if symbol and symbol.upper() != "NAN":
                df = df.copy()
                df.attrs["symbol"] = symbol
            bundle[key] = df

    return bundle


def save_research_bundle(
    out_dir: str | Path,
    *,
    summary: dict,
    folds: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    prefix: str = "walkforward",
) -> dict:
    """Persist walk-forward outputs to timestamped CSV/JSON files."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = out_path / f"{prefix}_{stamp}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    folds_path = bundle_dir / "folds.csv"
    summary_path = bundle_dir / "summary.json"
    summary_csv_path = bundle_dir / "summary.csv"
    folds.to_csv(folds_path, index=False)

    if trades is not None and not trades.empty:
        trades_path = bundle_dir / "trades.csv"
        trades.to_csv(trades_path, index=False)
    else:
        trades_path = None

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)

    return {
        "bundle_dir": str(bundle_dir),
        "folds_path": str(folds_path),
        "summary_path": str(summary_path),
        "summary_csv_path": str(summary_csv_path),
        "trades_path": str(trades_path) if trades_path else "",
    }


def _read_research_zip_bytes(raw: bytes, label: str) -> dict:
    out = {"summary": None, "folds": pd.DataFrame(), "trades": pd.DataFrame(), "label": label}
    if not raw:
        return out
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        members = zf.namelist()
        for name in members:
            lower = name.lower()
            try:
                payload = zf.read(name)
            except Exception:
                continue
            if lower.endswith("summary.json"):
                try:
                    out["summary"] = json.loads(payload.decode("utf-8"))
                except Exception:
                    pass
            elif lower.endswith("summary.csv"):
                try:
                    df = pd.read_csv(io.BytesIO(payload))
                    if not df.empty:
                        out["summary"] = df.iloc[0].to_dict()
                except Exception:
                    pass
            elif lower.endswith("folds.csv"):
                try:
                    out["folds"] = pd.read_csv(io.BytesIO(payload))
                except Exception:
                    pass
            elif lower.endswith("trades.csv"):
                try:
                    out["trades"] = pd.read_csv(io.BytesIO(payload))
                except Exception:
                    pass
    return out


def read_uploaded_research_bundle(uploaded_files: Iterable[object] | None) -> dict:
    """Read saved research outputs (summary/folds/trades) from CSV/JSON/ZIP uploads.

    Supports:
    - a ZIP bundle created by save_research_bundle()
    - individual summary.csv / summary.json / folds.csv / trades.csv files
    - multiple files uploaded together
    """
    result = {"summary": None, "folds": pd.DataFrame(), "trades": pd.DataFrame(), "label": ""}
    if not uploaded_files:
        return result

    for item in uploaded_files:
        name = getattr(item, "name", "uploaded")
        label = Path(str(name)).stem
        try:
            raw = item.getvalue() if hasattr(item, "getvalue") else item.read()
        except Exception:
            continue
        if not raw:
            continue

        lower_name = str(name).lower()
        if lower_name.endswith(".zip"):
            parsed = _read_research_zip_bytes(raw, label)
            if parsed.get("summary") and result["summary"] is None:
                result["summary"] = parsed["summary"]
            if result["folds"].empty and isinstance(parsed.get("folds"), pd.DataFrame) and not parsed["folds"].empty:
                result["folds"] = parsed["folds"]
            if result["trades"].empty and isinstance(parsed.get("trades"), pd.DataFrame) and not parsed["trades"].empty:
                result["trades"] = parsed["trades"]
            if not result["label"]:
                result["label"] = label
            continue

        try:
            if lower_name.endswith(".json"):
                payload = json.loads(raw.decode("utf-8"))
                if isinstance(payload, dict) and result["summary"] is None:
                    result["summary"] = payload
                if not result["label"]:
                    result["label"] = label
                continue

            df = pd.read_csv(io.BytesIO(raw))
            stem = Path(str(name)).stem.lower()
            if "summary" in stem:
                if result["summary"] is None and not df.empty:
                    result["summary"] = df.iloc[0].to_dict()
            elif "fold" in stem:
                if result["folds"].empty:
                    result["folds"] = df
            elif "trade" in stem:
                if result["trades"].empty:
                    result["trades"] = df
            else:
                if result["folds"].empty and not df.empty:
                    result["folds"] = df
        except Exception:
            continue

        if not result["label"]:
            result["label"] = label

    return result
