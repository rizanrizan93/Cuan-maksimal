from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping
import json
import time

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
            period = (rec.get("fundamental_latest_period") if family == "PUBLIC_FUNDAMENTAL_PROXY" else rec.get("idx_official_period_end") if family == "IDX_OFFICIAL_FUNDAMENTAL" else None)
            observed = rec.get("observed_at") or period
            content_hash = _digest(rec)
            rows.append({
                "memory_id": _digest({"ticker": ticker, "family": family, "period": period, "hash": content_hash}),
                "ticker": ticker, "family": family, "effective_period": period,
                "observed_at": observed, "provider": str(rec.get("provider") or rec.get("fundamental_provenance_state") or rec.get("ksei_provider_state") or ""),
                "source_url": rec.get("source_url") or rec.get("idx_official_source_url") or rec.get("idx_integrity_source_url") or rec.get("ksei_source_url"),
                "source_verified": bool(rec.get("source_verified", False) or rec.get("idx_official_source_verified", False)), "official_source": bool((rec.get("source_verified", False) and family.startswith("KSEI_")) or (family == "IDX_OFFICIAL_FUNDAMENTAL" and rec.get("idx_official_source_verified", False))),
                "content_sha256": content_hash, "last_scan_id": scan_id, "payload": rec,
            })
    dedup = {row["memory_id"]: row for row in rows}
    return list(dedup.values())


def persist_verify_research_memory(config: DatabaseConfig, *, scan_id: str, rows: list[dict[str, Any]], chunk_size: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persist durable memory with chunk-level retries and explicit partial verification.

    A single transient Supabase failure must not erase the status of chunks already
    written. v1.9.2 previously wrapped the whole multi-chunk write in one try/except,
    so one failed chunk could report 0/N VERIFY_SKIPPED even after earlier chunks had
    committed. This routine records partial writes and still performs hash readback.
    """
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
