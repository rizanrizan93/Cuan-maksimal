from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping
import json
import time

import pandas as pd

from persistence import DatabaseConfig, _json_value, _post_payload_in_chunks, database_status, _request
from free_tier_storage import prune_research_memory_best_effort


_NULLISH_DATE_TOKENS = {"", "<NA>", "NA", "N/A", "NAN", "NAT", "NONE", "NULL"}

_VOLATILE_MEMORY_KEYS = {
    "observed_at", "checked_at", "updated_at", "created_at", "fetched_at", "retrieved_at",
    "last_scan_id", "scan_id", "cache_state", "cache_status", "cache_age_hours",
    "provider_latency_ms", "request_elapsed_ms",
}
_EPHEMERAL_MEMORY_FAMILIES = {
    "BROKER_INVENTORY_OHLCV_PROXY",
    "BID_OFFER_EOD_PROXY",
}

_NARRATIVE_KEEP_PER_TICKER_PER_SCAN = 8

_COMMON_MEMORY_KEYS = {
    "ticker", "evidence_type", "provider", "collection_provider", "source_url", "url",
    "source_verified", "source_tier", "observed_at", "published_at", "event_date",
    "title", "summary", "publisher", "category", "materiality_score", "financial_bridge_score",
    "top_down_catalyst_score", "industry_translation_score", "issuer_alignment_score",
}
_MEMORY_PREFIXES = (
    "fundamental_", "future_", "idx_official_", "idx_integrity_", "ksei_", "ownership_",
    "revenue_", "earnings_", "operating_cash_flow", "free_cash_flow", "cash_", "debt_",
    "der_", "roe_", "roa_", "net_margin", "operating_margin", "current_ratio",
    "interest_bearing_", "total_liabilities", "conversion_", "story_runway", "retail_adoption",
)


def _compact_value(value: Any) -> Any:
    safe = _json_value(value)
    if isinstance(safe, str):
        return safe[:1200]
    if isinstance(safe, list):
        return [_compact_value(item) for item in safe[:20]]
    if isinstance(safe, dict):
        items = list(safe.items())[:30]
        return {str(k): _compact_value(v) for k, v in items}
    return safe


def _compact_memory_payload(record: Mapping[str, Any], family: str) -> dict[str, Any]:
    """Store only evidence needed for later research review, not whole wide scan rows."""
    output: dict[str, Any] = {}
    for key, value in record.items():
        name = str(key)
        lower = name.lower()
        if name in _COMMON_MEMORY_KEYS or any(lower.startswith(prefix) for prefix in _MEMORY_PREFIXES):
            output[name] = _compact_value(value)
        elif lower.endswith(("_score", "_state", "_period", "_pct", "_flag")):
            output[name] = _compact_value(value)
    output.setdefault("ticker", _compact_value(record.get("ticker")))
    output.setdefault("evidence_type", family)
    return output


def _bounded_narrative_events(events: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(events, pd.DataFrame) or events.empty or "ticker" not in events.columns:
        return events
    local = events.copy()
    obs_source = local["published_at"] if "published_at" in local.columns else local["event_date"] if "event_date" in local.columns else pd.Series(pd.NaT, index=local.index)
    mat_source = local["materiality_score"] if "materiality_score" in local.columns else pd.Series(0.0, index=local.index)
    local["_obs"] = pd.to_datetime(obs_source, errors="coerce", utc=True)
    local["_mat"] = pd.to_numeric(mat_source, errors="coerce").fillna(0.0)
    local = local.sort_values(["ticker", "_mat", "_obs"], ascending=[True, False, False], na_position="last")
    local = local.groupby("ticker", sort=False, as_index=False, group_keys=False).head(_NARRATIVE_KEEP_PER_TICKER_PER_SCAN)
    return local.drop(columns=["_obs", "_mat"], errors="ignore")


def _semantic_memory_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    """Remove fetch/checkpoint metadata so identical evidence upserts instead of growing forever."""
    return {
        str(key): value
        for key, value in record.items()
        if str(key).lower() not in _VOLATILE_MEMORY_KEYS
    }


def _normalise_date(value: Any) -> str | None:
    safe = _json_value(value)
    if safe is None or (isinstance(safe, str) and safe.strip().upper() in _NULLISH_DATE_TOKENS):
        return None
    stamp = pd.to_datetime(safe, errors="coerce")
    if pd.isna(stamp):
        return None
    return pd.Timestamp(stamp).date().isoformat()


def _normalise_observed_at(value: Any) -> str | None:
    safe = _json_value(value)
    if safe is None or (isinstance(safe, str) and safe.strip().upper() in _NULLISH_DATE_TOKENS):
        return None
    stamp = pd.to_datetime(safe, errors="coerce", utc=True)
    if pd.isna(stamp):
        return None
    return pd.Timestamp(stamp).isoformat()


def _canon(record: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k): _json_value(v) for k, v in record.items()}


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(raw).hexdigest()


def build_research_memory_rows(scan_id: str, events: pd.DataFrame | None, autonomous_evidence: pd.DataFrame | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(events, pd.DataFrame) and not events.empty:
        events = _bounded_narrative_events(events)
        for _, row in events.reset_index(drop=True).iterrows():
            rec = _canon(row.to_dict())
            ticker = str(rec.get("ticker") or "")
            family = "NARRATIVE_EVENT"
            period = _normalise_observed_at(rec.get("published_at") or rec.get("event_date"))
            compact = _compact_memory_payload(_semantic_memory_payload(rec), family)
            content_hash = _digest(compact)
            rows.append({
                "memory_id": _digest({"ticker": ticker, "family": family, "hash": content_hash}),
                "ticker": ticker, "family": family, "effective_period": None,
                "observed_at": period, "provider": str(rec.get("collection_provider") or rec.get("publisher") or ""),
                "source_url": rec.get("url") or rec.get("source_url"), "source_verified": bool(rec.get("source_verified", False)),
                "official_source": str(rec.get("source_tier") or "").upper() in {"OFFICIAL", "ISSUER", "REGULATOR"},
                "content_sha256": content_hash, "last_scan_id": scan_id, "payload": compact,
            })
    if isinstance(autonomous_evidence, pd.DataFrame) and not autonomous_evidence.empty:
        for _, row in autonomous_evidence.reset_index(drop=True).iterrows():
            rec = _canon(row.to_dict())
            ticker = str(rec.get("ticker") or "")
            family = str(rec.get("evidence_type") or "AUTONOMOUS_EVIDENCE")
            if family in _EPHEMERAL_MEMORY_FAMILIES:
                continue
            period_raw = (rec.get("fundamental_latest_period") if family == "PUBLIC_FUNDAMENTAL_PROXY" else rec.get("idx_official_period_end") if family == "IDX_OFFICIAL_FUNDAMENTAL" else None)
            period = _normalise_date(period_raw)
            observed = _normalise_observed_at(rec.get("observed_at")) or _normalise_observed_at(period)
            compact = _compact_memory_payload(_semantic_memory_payload(rec), family)
            content_hash = _digest(compact)
            rows.append({
                "memory_id": _digest({"ticker": ticker, "family": family, "period": period, "hash": content_hash}),
                "ticker": ticker, "family": family, "effective_period": period,
                "observed_at": observed, "provider": str(rec.get("provider") or rec.get("fundamental_provenance_state") or rec.get("ksei_provider_state") or ""),
                "source_url": rec.get("source_url") or rec.get("idx_official_source_url") or rec.get("idx_integrity_source_url") or rec.get("ksei_source_url"),
                "source_verified": bool(rec.get("source_verified", False) or rec.get("idx_official_source_verified", False)), "official_source": bool((rec.get("source_verified", False) and family.startswith("KSEI_")) or (family == "IDX_OFFICIAL_FUNDAMENTAL" and rec.get("idx_official_source_verified", False))),
                "content_sha256": content_hash, "last_scan_id": scan_id, "payload": compact,
            })
    dedup = {row["memory_id"]: row for row in rows}
    return list(dedup.values())


def persist_verify_research_memory(config: DatabaseConfig, *, scan_id: str, rows: list[dict[str, Any]], chunk_size: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = database_status(config)
    if not config.ready:
        return pd.DataFrame([{**base, "table": "cak_research_memory", "rows_attempted": len(rows), "rows_written": 0, "state": "RESEARCH_MEMORY_DATABASE_DISABLED"}]), pd.DataFrame([{**base, "table": "cak_research_memory", "rows_expected": len(rows), "rows_verified": 0, "state": "RESEARCH_MEMORY_VERIFY_SKIPPED", "detail": "database disabled"}])
    if not rows:
        write = pd.DataFrame([{**base, "table": "cak_research_memory", "rows_attempted": 0, "rows_written": 0, "state": "RESEARCH_MEMORY_WRITTEN_EMPTY"}])
        verify = pd.DataFrame([{**base, "table": "cak_research_memory", "rows_expected": 0, "rows_verified": 0, "state": "RESEARCH_MEMORY_VERIFIED_EMPTY"}])
        return write, verify

    safe_chunk = max(20, min(200, int(chunk_size or 100)))
    written = 0
    failed_chunks: list[str] = []
    for start in range(0, len(rows), safe_chunk):
        chunk = rows[start:start + safe_chunk]
        last_error = ""
        chunk_written = False
        for attempt in range(3):
            try:
                count = _post_payload_in_chunks(config, table="cak_research_memory", conflict="memory_id", payload=chunk, chunk_size=len(chunk), return_rows=False)
                written += int(count)
                chunk_written = int(count) == len(chunk)
                if chunk_written:
                    break
                last_error = f"partial write {count}/{len(chunk)}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                time.sleep(0.25 * (attempt + 1))
        if not chunk_written:
            failed_chunks.append(f"{start}:{start+len(chunk)}={last_error[:240]}")

    write_state = "RESEARCH_MEMORY_WRITTEN" if written == len(rows) else "RESEARCH_MEMORY_WRITE_PARTIAL" if written > 0 else "RESEARCH_MEMORY_WRITE_FAILED"
    write = pd.DataFrame([{**base, "table": "cak_research_memory", "rows_attempted": len(rows), "rows_written": written, "failed_chunks": len(failed_chunks), "state": write_state, "detail": " | ".join(failed_chunks[:8])}])

    expected = {row["memory_id"]: row["content_sha256"] for row in rows}
    verified = 0
    read_errors: list[str] = []
    ids = list(expected)
    for start in range(0, len(ids), 100):
        chunk = ids[start:start+100]
        quoted = ",".join('"'+item.replace('"','')+'"' for item in chunk)
        payload = None
        last_error = ""
        for attempt in range(2):
            try:
                response = _request(config, "GET", "cak_research_memory", params={"select": "memory_id,content_sha256", "memory_id": f"in.({quoted})"}, timeout=20)
                payload = response.json()
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < 1:
                    time.sleep(0.2)
        if not isinstance(payload, list):
            read_errors.append(f"{start}:{start+len(chunk)}={last_error[:240]}")
            continue
        verified += sum(1 for item in payload if expected.get(str(item.get("memory_id"))) == str(item.get("content_sha256") or ""))

    if verified == len(expected):
        verify_state = "RESEARCH_MEMORY_VERIFIED_EXACT"
    elif verified > 0:
        verify_state = "RESEARCH_MEMORY_VERIFIED_PARTIAL"
    elif read_errors:
        verify_state = "RESEARCH_MEMORY_READBACK_FAILED"
    else:
        verify_state = "RESEARCH_MEMORY_HASH_MISMATCH"
    verify = pd.DataFrame([{**base, "table": "cak_research_memory", "rows_expected": len(expected), "rows_verified": verified, "read_error_chunks": len(read_errors), "state": verify_state, "detail": " | ".join(read_errors[:8])}])
    prune_research_memory_best_effort(config)
    return write, verify


def load_latest_research_memory(config: DatabaseConfig, tickers: list[str], family: str, *, limit_per_ticker: int = 1) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {str(t): [] for t in tickers}
    if not config.ready or not tickers:
        return output
    safe = [str(t).replace('"', '') for t in dict.fromkeys(tickers) if str(t)]
    for start in range(0, len(safe), 80):
        chunk = safe[start:start+80]
        quoted = ",".join('"'+item+'"' for item in chunk)
        try:
            response = _request(
                config, "GET", "cak_research_memory",
                params={
                    "select": "ticker,family,effective_period,observed_at,provider,source_verified,official_source,content_sha256,payload",
                    "ticker": f"in.({quoted})", "family": f"eq.{family}",
                    "order": "effective_period.desc.nullslast,observed_at.desc.nullslast,updated_at.desc",
                    "limit": str(max(200, len(chunk) * max(1, int(limit_per_ticker)) * 4)),
                }, timeout=20,
            )
            rows = response.json()
        except Exception:
            rows = []
        if not isinstance(rows, list):
            continue
        for row in rows:
            ticker = str(row.get("ticker") or "")
            if ticker not in output or len(output[ticker]) >= max(1, int(limit_per_ticker)):
                continue
            output[ticker].append(row)
    return output

def load_replayable_narrative_events(
    config: DatabaseConfig,
    tickers: list[str],
    *,
    as_of: Any = None,
    limit_per_ticker: int = 6,
    max_age_days: int = 540,
) -> pd.DataFrame:
    """Replay bounded raw narrative evidence from durable research memory.

    Only raw event payloads are replayed. Derived score snapshots are never fed
    back into scoring, which prevents circular coverage inflation. Current-scan
    events retain precedence because the caller appends this frame last and
    deduplicates by ticker/title/url. Future-dated observations and very old
    memories are excluded; source verification is preserved, never promoted.
    """
    if not config.ready or not tickers:
        return pd.DataFrame()
    memory = load_latest_research_memory(
        config, tickers, "NARRATIVE_EVENT", limit_per_ticker=max(1, int(limit_per_ticker))
    )
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        for item in memory.get(str(ticker), []):
            payload = item.get("payload")
            if not isinstance(payload, Mapping):
                continue
            row = dict(payload)
            observed = pd.to_datetime(
                row.get("published_at") or row.get("event_date") or item.get("observed_at"),
                errors="coerce", utc=True,
            )
            if pd.isna(observed):
                continue
            age_days = (now - observed).total_seconds() / 86400.0
            if age_days < -1.0 or age_days > max(1, int(max_age_days)):
                continue
            row["ticker"] = str(row.get("ticker") or ticker)
            row["published_at"] = observed
            row.setdefault("source_verified", bool(item.get("source_verified", False)))
            if not row.get("source_tier") and bool(item.get("official_source", False)):
                row["source_tier"] = "OFFICIAL"
            row["research_memory_replayed"] = True
            row["research_memory_content_sha256"] = str(item.get("content_sha256") or "")
            row["research_memory_original_provider"] = str(item.get("provider") or "")
            row["collection_provider"] = "PERSISTED_RESEARCH_MEMORY"
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    dedupe = [column for column in ("ticker", "title", "url") if column in frame.columns]
    if dedupe:
        frame = frame.drop_duplicates(dedupe, keep="first")
    return frame.reset_index(drop=True)


__all__ = ["build_research_memory_rows", "persist_verify_research_memory", "load_latest_research_memory", "load_replayable_narrative_events"]
