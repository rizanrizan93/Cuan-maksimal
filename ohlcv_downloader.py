
from __future__ import annotations

import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

try:
    from data_engine import normalize_ticker
except Exception:  # pragma: no cover
    def normalize_ticker(symbol: str) -> str:
        s = str(symbol or "").strip().upper()
        if not s or s == "NAN":
            return ""
        if s in {"TICKER", "SYMBOL", "STOCK", "SAHAM", "CODE", "KODE", "HEADER"}:
            return ""
        if s.startswith("^"):
            return s
        return s if s.endswith(".JK") else f"{s}.JK"


@dataclass
class DownloadResult:
    ticker: str
    status: str
    rows: int = 0
    first_date: str = ""
    last_date: str = ""
    error: str = ""


def _clean_filename(value: str) -> str:
    value = str(value or "").strip().upper()
    value = value.replace(".JK", "")
    value = re.sub(r"[^A-Z0-9^]+", "_", value)
    return value.strip("_") or "UNKNOWN"


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        top = list(out.columns.get_level_values(0))
        if any(col in top for col in ["Open", "High", "Low", "Close", "Volume"]):
            out.columns = out.columns.get_level_values(0)
        else:
            out.columns = out.columns.get_level_values(-1)

    out.columns = [str(c).strip() for c in out.columns]
    if "Adj Close" not in out.columns and "AdjClose" in out.columns:
        out = out.rename(columns={"AdjClose": "Adj Close"})

    if isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()].copy()
    elif "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"]).copy()
        out = out.set_index("Date")
    else:
        try:
            out.index = pd.to_datetime(out.index, errors="coerce")
            out = out[~out.index.isna()].copy()
        except Exception:
            return pd.DataFrame()

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(set(out.columns)):
        return pd.DataFrame()

    out = out.loc[:, ~out.columns.duplicated()].copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")].copy()

    for col in needed:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "Adj Close" in out.columns:
        out["Adj Close"] = pd.to_numeric(out["Adj Close"], errors="coerce")

    out = out.dropna(subset=needed).copy()
    return out if not out.empty else pd.DataFrame()


def extract_universe_tickers(universe_df: pd.DataFrame) -> list[str]:
    if universe_df is None or universe_df.empty:
        return []

    df = universe_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    candidate_cols = ["Ticker", "ticker", "Symbol", "symbol", "Kode", "code", "Stock", "saham"]
    ticker_col = next((c for c in candidate_cols if c in df.columns), None)
    if ticker_col is None:
        ticker_col = df.columns[0]

    tickers = []
    for raw in df[ticker_col].dropna().tolist():
        sym = normalize_ticker(str(raw).strip())
        if sym and sym not in tickers:
            tickers.append(sym)
    return tickers


def load_universe_from_csv(path_or_buffer) -> pd.DataFrame:
    if path_or_buffer is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path_or_buffer)
    except Exception:
        return pd.DataFrame()


def _download_one(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    auto_adjust: bool = False,
    timeout: int = 20,
) -> tuple[str, pd.DataFrame, str]:
    sym = normalize_ticker(ticker)
    if not sym:
        return ticker, pd.DataFrame(), "invalid_symbol"

    candidates = [sym]
    if sym.endswith(".JK"):
        candidates.append(sym[:-3])
    elif not sym.startswith("^"):
        candidates.append(f"{sym}.JK")

    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    last_err = ""
    for candidate in candidates:
        try:
            df = yf.download(
                candidate,
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
                timeout=timeout,
            )
            df = _standardize_ohlcv(df)
            if not df.empty:
                df = df.copy()
                df["Ticker"] = normalize_ticker(ticker)
                return normalize_ticker(ticker), df, ""
            last_err = "empty_dataframe"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
    return normalize_ticker(ticker), pd.DataFrame(), last_err or "download_failed"


def download_batch_idx_ohlcv(
    tickers: Iterable[str],
    *,
    period: str = "1y",
    interval: str = "1d",
    max_workers: int = 6,
    auto_adjust: bool = False,
    timeout: int = 20,
) -> dict:
    tickers = [normalize_ticker(t) for t in tickers if normalize_ticker(t)]
    tickers = list(dict.fromkeys(tickers))

    frames: dict[str, pd.DataFrame] = {}
    results: list[DownloadResult] = []

    if not tickers:
        return {
            "frames": frames,
            "results": pd.DataFrame(),
            "summary": {
                "tickers_requested": 0,
                "tickers_downloaded": 0,
                "tickers_failed": 0,
                "rows_total": 0,
                "period": period,
                "interval": interval,
                "max_workers": max_workers,
            },
        }

    max_workers = max(1, int(max_workers or 1))
    if max_workers == 1:
        for ticker in tickers:
            sym, df, err = _download_one(ticker, period=period, interval=interval, auto_adjust=auto_adjust, timeout=timeout)
            if not df.empty:
                frames[sym] = df
                results.append(
                    DownloadResult(
                        ticker=sym,
                        status="ok",
                        rows=int(len(df)),
                        first_date=str(df.index.min().date()),
                        last_date=str(df.index.max().date()),
                    )
                )
            else:
                results.append(DownloadResult(ticker=sym, status="failed", error=err))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {
                ex.submit(_download_one, ticker, period, interval, auto_adjust, timeout): ticker
                for ticker in tickers
            }
            for fut in as_completed(fut_map):
                ticker = fut_map[fut]
                try:
                    sym, df, err = fut.result()
                    if not df.empty:
                        frames[sym] = df
                        results.append(
                            DownloadResult(
                                ticker=sym,
                                status="ok",
                                rows=int(len(df)),
                                first_date=str(df.index.min().date()),
                                last_date=str(df.index.max().date()),
                            )
                        )
                    else:
                        results.append(DownloadResult(ticker=sym, status="failed", error=err))
                except Exception as exc:
                    results.append(DownloadResult(ticker=normalize_ticker(ticker), status="failed", error=f"{type(exc).__name__}: {exc}"))

    results_df = pd.DataFrame([asdict(r) for r in results]).sort_values(["status", "ticker"], ascending=[True, True], ignore_index=True)
    rows_total = int(results_df.loc[results_df["status"] == "ok", "rows"].sum()) if not results_df.empty else 0
    summary = {
        "tickers_requested": int(len(tickers)),
        "tickers_downloaded": int((results_df["status"] == "ok").sum()) if not results_df.empty else 0,
        "tickers_failed": int((results_df["status"] != "ok").sum()) if not results_df.empty else 0,
        "rows_total": rows_total,
        "first_date": str(min((df.index.min() for df in frames.values()), default=pd.NaT)),
        "last_date": str(max((df.index.max() for df in frames.values()), default=pd.NaT)),
        "period": period,
        "interval": interval,
        "max_workers": max_workers,
    }
    return {"frames": frames, "results": results_df, "summary": summary}


def save_batch_bundle(
    frames: dict[str, pd.DataFrame],
    *,
    out_dir: str | Path,
    prefix: str = "idx_ohlcv",
    combined_csv: bool = True,
    make_zip: bool = True,
    summary: dict | None = None,
    results: pd.DataFrame | None = None,
) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    bundle_dir = out_path / f"{prefix}_{stamp}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "summary": summary or {},
        "files": [],
    }

    # Per-ticker CSVs
    for ticker, df in sorted(frames.items()):
        if df is None or df.empty:
            continue
        filename = f"{_clean_filename(ticker)}.csv"
        path = bundle_dir / filename
        tmp = df.copy()
        if isinstance(tmp.index, pd.DatetimeIndex):
            tmp = tmp.reset_index()
            if "Date" not in tmp.columns:
                tmp = tmp.rename(columns={tmp.columns[0]: "Date"})
        else:
            tmp = tmp.reset_index()
            if "Date" not in tmp.columns:
                tmp = tmp.rename(columns={tmp.columns[0]: "Date"})
        if "Ticker" not in tmp.columns:
            tmp["Ticker"] = normalize_ticker(ticker)
        tmp.to_csv(path, index=False)
        manifest["files"].append({"ticker": ticker, "path": path.name, "rows": int(len(df))})

    combined_path = ""
    if combined_csv and frames:
        combo = []
        for ticker, df in sorted(frames.items()):
            if df is None or df.empty:
                continue
            tmp = df.copy().reset_index()
            if "Date" not in tmp.columns:
                tmp = tmp.rename(columns={tmp.columns[0]: "Date"})
            tmp["Ticker"] = normalize_ticker(ticker)
            combo.append(tmp)
        if combo:
            combined = pd.concat(combo, ignore_index=True)
            combined_path = str(bundle_dir / f"{prefix}_combined.csv")
            combined.to_csv(combined_path, index=False)

    summary_path = bundle_dir / "download_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary or {}, f, ensure_ascii=False, indent=2, default=str)

    results_path = ""
    if results is not None and not results.empty:
        results_path = str(bundle_dir / "download_results.csv")
        results.to_csv(results_path, index=False)

    manifest_path = bundle_dir / "download_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    zip_path = ""
    if make_zip:
        zip_path = str(bundle_dir.with_suffix(".zip"))
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file in bundle_dir.rglob("*"):
                if file.is_file():
                    zf.write(file, arcname=file.relative_to(bundle_dir))

    return {
        "bundle_dir": str(bundle_dir),
        "zip_path": zip_path,
        "combined_csv_path": combined_path,
        "summary_path": str(summary_path),
        "manifest_path": str(manifest_path),
        "results_path": results_path,
    }


def main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Batch IDX OHLCV downloader")
    parser.add_argument("--universe", required=True, help="Path to universe CSV")
    parser.add_argument("--outdir", default="research_outputs/ohlcv_downloads", help="Output directory")
    parser.add_argument("--period", default="1y", help="Download period, e.g. 1y")
    parser.add_argument("--interval", default="1d", help="Download interval, e.g. 1d")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--prefix", default="idx_ohlcv")
    parser.add_argument("--no-zip", action="store_true")
    args = parser.parse_args()

    universe_df = load_universe_from_csv(args.universe)
    tickers = extract_universe_tickers(universe_df)
    batch = download_batch_idx_ohlcv(
        tickers,
        period=args.period,
        interval=args.interval,
        max_workers=args.max_workers,
    )
    saved = save_batch_bundle(
        batch["frames"],
        out_dir=args.outdir,
        prefix=args.prefix,
        combined_csv=True,
        make_zip=not args.no_zip,
        summary=batch["summary"],
        results=batch["results"],
    )
    print(json.dumps({"summary": batch["summary"], "saved": saved}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
