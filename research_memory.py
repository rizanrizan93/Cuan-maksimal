from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping
import json

import pandas as pd

from persistence import DatabaseConfig, _json_value, _post_payload_in_chunks, database_status, _request


def _canon(record: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k): _json_value(v) for k, v in record.items()}


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(raw).hexdigest()


def build_research_memory_rows(scan_id: str, events: pd.DataFrame | None, autonomous_evidence: pd.DataFrame | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(events, pd.DataFrame) and not events.empty:
        for _, row in events.reset_index(drop=True).iterrows():
            rec = _canon(row.to_dict())
            ticker = str(rec.get("ticker") or "")
            family = "NARRATIVE_EVENT"
            period = rec.get("published_at") or rec.get("event_date")
            content_hash = _digest(rec)
            rows.append({
                "memory_id": _digest({"ticker": ticker, "family": family, "hash": content_hash}),
                "ticker": ticker, "family": family, "effective_period": None,
                "observed_at": period, "provider": str(rec.get("collection_provider") or rec.get("publisher") or ""),
                "source_url": rec.get("url") or rec.get("source_url"), "source_verified": bool(rec.get("source_verified", False)),
                "official_source": str(rec.get("source_tier") or "").upper() in {"OFFICIAL", "ISSUER", "REGULATOR"},
                "content_sha256": content_hash, "last_scan_id": scan_id, "payload": rec,
            })
    if isinstance(autonomous_evidence, pd.DataFrame) and not autonomous_evidence.empty:
        for _, row in autonomous_evidence.reset_index(drop=True).iterrows():
            rec = _canon(row.to_dict())
            ticker = str(rec.get("ticker") or "")
            family = str(rec.get("evidence_type") or "AUTONOMOUS_EVIDENCE")
            period = rec.get("fundamental_latest_period") if family == "PUBLIC_FUNDAMENTAL_PROXY" else None
            observed = rec.get("observed_at") or period
            content_hash = _digest(rec)
            rows.append({
                "memory_id": _digest({"ticker": ticker, "family": family, "period": period, "hash": content_hash}),
                "ticker": ticker, "family": family, "effective_period": period,
                "observed_at": observed, "provider": str(rec.get("provider") or rec.get("fundamental_provenance_state") or rec.get("ksei_provider_state") or ""),
                "source_url": rec.get("source_url") or rec.get("idx_integrity_source_url") or rec.get("ksei_source_url"),
                "source_verified": bool(rec.get("source_verified", False)), "official_source": bool(rec.get("source_verified", False) and family.startswith("KSEI_")),
                "content_sha256": content_hash, "last_scan_id": scan_id, "payload": rec,
            })
    dedup = {row["memory_id"]: row for row in rows}
    return list(dedup.values())


def persist_verify_research_memory(config: DatabaseConfig, *, scan_id: str, rows: list[dict[str, Any]], chunk_size: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = database_status(config)
    if not config.ready:
        return pd.DataFrame([{**base, "table": "cak_research_memory", "rows_attempted": len(rows), "rows_written": 0, "state": "RESEARCH_MEMORY_DATABASE_DISABLED"}]), pd.DataFrame([{**base, "table": "cak_research_memory", "rows_expected": len(rows), "rows_verified": 0, "state": "RESEARCH_MEMORY_VERIFY_SKIPPED"}])
    try:
        written = _post_payload_in_chunks(config, table="cak_research_memory", conflict="memory_id", payload=rows, chunk_size=chunk_size, return_rows=False)
        write = pd.DataFrame([{**base, "table": "cak_research_memory", "rows_attempted": len(rows), "rows_written": written, "state": "RESEARCH_MEMORY_WRITTEN" if written == len(rows) else "RESEARCH_MEMORY_WRITE_PARTIAL"}])
    except Exception as exc:
        return pd.DataFrame([{**base, "table": "cak_research_memory", "rows_attempted": len(rows), "rows_written": 0, "state": "RESEARCH_MEMORY_WRITE_FAILED", "detail": str(exc)}]), pd.DataFrame([{**base, "table": "cak_research_memory", "rows_expected": len(rows), "rows_verified": 0, "state": "RESEARCH_MEMORY_VERIFY_SKIPPED"}])
    if not rows:
        return write, pd.DataFrame([{**base, "table": "cak_research_memory", "rows_expected": 0, "rows_verified": 0, "state": "RESEARCH_MEMORY_VERIFIED_EMPTY"}])
    expected = {row["memory_id"]: row["content_sha256"] for row in rows}
    verified = 0
    try:
        ids = list(expected)
        for start in range(0, len(ids), 100):
            chunk = ids[start:start+100]
            quoted = ",".join('"'+item.replace('"','')+'"' for item in chunk)
            response = _request(config, "GET", "cak_research_memory", params={"select": "memory_id,content_sha256", "memory_id": f"in.({quoted})"}, timeout=20)
            payload = response.json()
            if isinstance(payload, list):
                verified += sum(1 for item in payload if expected.get(str(item.get("memory_id"))) == str(item.get("content_sha256") or ""))
        exact = verified == len(expected)
        verify = pd.DataFrame([{**base, "table": "cak_research_memory", "rows_expected": len(expected), "rows_verified": verified, "state": "RESEARCH_MEMORY_VERIFIED_EXACT" if exact else "RESEARCH_MEMORY_HASH_MISMATCH"}])
    except Exception as exc:
        verify = pd.DataFrame([{**base, "table": "cak_research_memory", "rows_expected": len(expected), "rows_verified": verified, "state": "RESEARCH_MEMORY_READBACK_FAILED", "detail": str(exc)}])
    return write, verify


def load_latest_research_memory(config: DatabaseConfig, tickers: list[str], family: str, *, limit_per_ticker: int = 1) -> dict[str, list[dict[str, Any]]]:
    """Load latest durable memory rows for a family, grouped by ticker.

    Used only as a fallback behind the normal source cache/provider refresh path.
    """
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

__all__ = ["build_research_memory_rows", "persist_verify_research_memory", "load_latest_research_memory"]
