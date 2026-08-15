from __future__ import annotations

"""Research-only live Future Fundamental discovery for Emir production scans."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Any, Iterable, Mapping
from urllib.parse import quote_plus
import re
import xml.etree.ElementTree as ET

import pandas as pd
import requests

LIVE_FORWARD_EVIDENCE_VERSION = "1.1.0-company-entity"

_RULES: tuple[tuple[str, tuple[str, ...], float, float], ...] = (
    ("PROJECT_OR_CONTRACT", ("KONTRAK", "CONTRACT", "BACKLOG", "ORDER BOOK", "ORDERBOOK", "OFFTAKE", "TENDER", "PESANAN"), 76.0, 82.0),
    ("CAPACITY_OR_EXPANSION", ("EKSPANSI", "EXPANSION", "KAPASITAS", "CAPACITY", "PABRIK", "PLANT", "SMELTER", "COMMISSIONING", "COMMERCIAL OPERATION"), 76.0, 80.0),
    ("GUIDANCE_OR_TARGET", ("GUIDANCE", "TARGET PENDAPATAN", "TARGET REVENUE", "TARGET LABA", "TARGET PROFIT", "TARGET PRODUKSI", "PRODUCTION TARGET"), 72.0, 86.0),
    ("PRODUCT_OR_NEW_MARKET", ("PRODUK BARU", "NEW PRODUCT", "PASAR BARU", "NEW MARKET", "PELUNCURAN", "LAUNCH"), 68.0, 72.0),
    ("STRATEGIC_INVESTOR_OR_MA", ("JOINT VENTURE", " JV ", "AKUISISI", "ACQUISITION", "MERGER", "INVESTOR STRATEGIS", "STRATEGIC INVESTOR"), 82.0, 78.0),
    ("CAPEX", ("CAPEX", "BELANJA MODAL", "CAPITAL EXPENDITURE", "INVESTASI"), 70.0, 76.0),
)
_ADVERSE: tuple[tuple[str, tuple[str, ...], float, float], ...] = (
    ("PROJECT_DELAY_OR_CANCEL", ("KONTRAK DIBATALKAN", "CONTRACT CANCELLED", "CONTRACT TERMINATED", "PROYEK DITUNDA", "PROJECT DELAY", "COD MUNDUR"), 86.0, 88.0),
    ("GUIDANCE_CUT", ("GUIDANCE CUT", "GUIDANCE DITURUNKAN", "TARGET DITURUNKAN"), 82.0, 90.0),
)
_STOPWORDS = {"PT", "TBK", "PERSERO", "INDONESIA", "INDUSTRI", "INDUSTRIES", "THE", "AND", "DAN"}


def _ticker(value: Any) -> str:
    text = str(value or "").upper().strip()
    return text if text.endswith(".JK") else f"{text}.JK" if text else ""


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", str(value or "")))).strip()


def _company_label(value: Any) -> str:
    text = _clean(value).upper()
    text = re.sub(r"\bTBK\b|\bPT\b|\bPERSERO\b", " ", text)
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _company_tokens(value: Any) -> list[str]:
    return [token for token in _company_label(value).split() if len(token) >= 4 and token not in _STOPWORDS]


def _entity_match(title: str, bare: str, company_name: str) -> tuple[bool, str]:
    upper = str(title or "").upper()
    if re.search(rf"(?<![A-Z0-9]){re.escape(bare.upper())}(?![A-Z0-9])", upper):
        return True, "TICKER_EXACT_TITLE"
    tokens = _company_tokens(company_name)
    if not tokens:
        return False, "NO_COMPANY_TOKENS"
    matches = [token for token in tokens if re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", upper)]
    needed = 1 if len(tokens) == 1 else 2
    return len(matches) >= needed, f"COMPANY_TOKEN_MATCH_{len(matches)}OF{len(tokens)}"


def _classify(text: str) -> tuple[str, float, float, int] | None:
    upper = f" {str(text or '').upper()} "
    for category, tokens, materiality, bridge in _ADVERSE:
        if any(token in upper for token in tokens):
            return category, materiality, bridge, -1
    for category, tokens, materiality, bridge in _RULES:
        if any(token in upper for token in tokens):
            return category, materiality, bridge, 1
    return None


def _published(value: Any) -> pd.Timestamp | pd.NaT:
    try:
        stamp = pd.Timestamp(parsedate_to_datetime(str(value)))
    except Exception:
        stamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(stamp):
        return pd.NaT
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _one(ticker: str, company_name: str, lookback_days: int, timeout: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    symbol = _ticker(ticker)
    bare = symbol.removesuffix(".JK")
    checked = pd.Timestamp.now(tz="UTC")
    label = _company_label(company_name)
    query_text = f'"{label}" saham' if label else f'"{bare}" IDX saham'
    url = f"https://news.google.com/rss/search?q={quote_plus(query_text)}&hl=id&gl=ID&ceid=ID:id"
    base_audit = {
        "ticker": symbol,
        "company_name": company_name,
        "checked_at": checked.isoformat(),
        "provider": "GOOGLE_NEWS_RSS_FORWARD",
        "collection_version": LIVE_FORWARD_EVIDENCE_VERSION,
        "source_verified": False,
        "strict_official_quorum": False,
    }
    try:
        response = requests.get(url, timeout=max(1.0, float(timeout)), headers={"User-Agent": "Mozilla/5.0 IDXEmirScanner/1.1"})
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as exc:
        return [], {**base_audit, "state": "FORWARD_CHECK_FAILED_RETRYABLE", "coverage_pct": 0.0, "detail": f"{type(exc).__name__}: {str(exc)[:180]}"}

    cutoff = checked - pd.Timedelta(days=max(7, int(lookback_days)))
    events: list[dict[str, Any]] = []
    publishers: set[str] = set()
    scanned = matched = 0
    for item in root.findall(".//item")[:30]:
        title = _clean(item.findtext("title"))
        published = _published(item.findtext("pubDate"))
        if pd.notna(published) and published < cutoff:
            continue
        scanned += 1
        entity_ok, entity_method = _entity_match(title, bare, company_name)
        if not entity_ok:
            continue
        matched += 1
        classified = _classify(title)
        if classified is None:
            continue
        category, materiality, bridge, direction = classified
        source_node = item.find("source")
        publisher = _clean(source_node.text if source_node is not None else "")
        link = str(item.findtext("link") or "").strip()
        if publisher:
            publishers.add(publisher.upper())
        events.append({
            "ticker": symbol,
            "company_name": company_name,
            "published_at": published.isoformat() if pd.notna(published) else checked.isoformat(),
            "title": title[:500],
            "summary": "Live forward research discovery; verify official disclosure before real-money use.",
            "publisher": publisher,
            "url": link,
            "source_tier": "PUBLIC_RESEARCH",
            "materiality_score": materiality,
            "financial_bridge_score": bridge,
            "top_down_catalyst_score": 50.0,
            "industry_translation_score": 66.0 if direction > 0 else 34.0,
            "issuer_alignment_score": 62.0 if direction > 0 else 32.0,
            "category": category,
            "collection_provider": "LIVE_GOOGLE_NEWS_FORWARD_RESEARCH",
            "source_verified": False,
            "entity_match_verified": True,
            "entity_match_method": entity_method,
            "source_quorum_verified": False,
            "source_quorum_count": max(1, len(publishers)),
            "forward_research_only": True,
        })
    if events:
        return events, {
            **base_audit,
            "state": "MATERIAL_FORWARD_RESEARCH_EVIDENCE_FOUND",
            "coverage_pct": 100.0,
            "items_scanned": scanned,
            "entity_matched_items": matched,
            "material_events": len(events),
            "publisher_count": len(publishers),
        }
    return [], {
        **base_audit,
        "state": "FORWARD_CHECK_COMPLETED_NO_MATERIAL_EVENT",
        "coverage_pct": 100.0,
        "items_scanned": scanned,
        "entity_matched_items": matched,
        "material_events": 0,
        "publisher_count": 0,
    }


def collect_live_forward_evidence(
    tickers: Iterable[Any], *, company_names: Mapping[str, Any] | None = None,
    lookback_days: int = 120, max_workers: int = 12, timeout: float = 4.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = list(dict.fromkeys(_ticker(value) for value in tickers if _ticker(value)))
    if not names:
        return pd.DataFrame(), pd.DataFrame()
    company_names = {_ticker(key): str(value or "") for key, value in dict(company_names or {}).items()}
    events: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    workers = max(1, min(16, int(max_workers), len(names)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_one, ticker, company_names.get(ticker, ""), lookback_days, timeout): ticker
            for ticker in names
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                event_rows, audit = future.result()
                events.extend(event_rows)
                audits.append(audit)
            except Exception as exc:
                audits.append({
                    "ticker": ticker,
                    "company_name": company_names.get(ticker, ""),
                    "checked_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "provider": "GOOGLE_NEWS_RSS_FORWARD",
                    "state": "FORWARD_CHECK_FAILED_RETRYABLE",
                    "coverage_pct": 0.0,
                    "detail": f"{type(exc).__name__}: {str(exc)[:180]}",
                })
    return pd.DataFrame(events), pd.DataFrame(audits)


def install_dashboard_cost_integrity() -> None:
    try:
        import top3_dashboard as dashboard
    except Exception:
        return
    original = getattr(dashboard, "render_top3_dashboard_html", None)
    if not callable(original) or getattr(original, "__cost_placement_v2__", False):
        return

    def fixed(top3: pd.DataFrame, *args: Any, **kwargs: Any) -> str:
        html = original(top3, *args, **kwargs)
        blocks = re.findall(r'<div class="es-cost-basis">.*?</div>', html, flags=re.DOTALL)
        if not blocks:
            return html
        html = re.sub(r'<div class="es-cost-basis">.*?</div>', "", html, flags=re.DOTALL)
        markers = ("</div><p>OHLCV PROXY — BUKAN IDENTITAS BROKER</p>", "</div><p>DIRECT BROKER EVIDENCE</p>")
        cursor = 0
        for block in blocks:
            candidates = [(html.find(marker, cursor), marker) for marker in markers]
            candidates = [(idx, marker) for idx, marker in candidates if idx >= 0]
            if not candidates:
                break
            index, marker = min(candidates, key=lambda pair: pair[0])
            replacement = "</div>" + block + marker[len("</div>"):]
            html = html[:index] + replacement + html[index + len(marker):]
            cursor = index + len(replacement)
        return html

    fixed.__cost_placement_v2__ = True
    fixed.__name__ = getattr(original, "__name__", "render_top3_dashboard_html")
    fixed.__doc__ = getattr(original, "__doc__", None)
    setattr(dashboard, "render_top3_dashboard_html", fixed)


install_dashboard_cost_integrity()

__all__ = ["LIVE_FORWARD_EVIDENCE_VERSION", "collect_live_forward_evidence", "install_dashboard_cost_integrity"]
