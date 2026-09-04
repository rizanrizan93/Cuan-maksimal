from __future__ import annotations

"""Bind EMIR strict forward facts to the Shared Evidence Hub.

EMIR local `cak_forward_evidence` remains a producer/audit source. Once the v36
canonical table is readable, the Shared Hub is authoritative for factual forward
input; EMIR's existing `forward_rows_to_events` interpretation remains unchanged.
"""

from functools import wraps
from typing import Any, Iterable

import pandas as pd

from shared_forward_evidence import (
    canonical_rows_to_emir_rows,
    canonicalize_emir_row,
    merge_equivalent_rows,
    read_canonical_forward_rows,
    upsert_canonical_forward_rows,
)

PATCH_VERSION = "1.0.1-shared-canonical-forward-emir-reconcile"


def _reconcile_with_shared(local_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Union producer provenance/factual payload before any upsert.

    PostgREST merge-upsert replaces supplied JSON fields. Reading the canonical
    row first prevents an EMIR refresh from erasing PASTICUAN producer lineage,
    corroboration URLs, or factual payload fields already present in the hub.
    """
    if not local_rows:
        return [], {"state": "NO_ITEMS", "rows": 0}
    tickers = sorted({str(row.get("ticker") or "").strip() for row in local_rows if str(row.get("ticker") or "").strip()})
    existing_rows, audit = read_canonical_forward_rows(tickers, client_id="EMIR")
    if str(audit.get("state") or "") != "SHARED_CANONICAL_FORWARD":
        # Do not risk blind overwrite when the authoritative row cannot be read.
        return [], {"state": "RECONCILE_READ_UNAVAILABLE", "rows": 0, "error": str(audit.get("error") or "")}
    existing_by_id = {
        str(row.get("canonical_event_id") or ""): dict(row)
        for row in existing_rows
        if str(row.get("canonical_event_id") or "")
    }
    reconciled: list[dict[str, Any]] = []
    for local in local_rows:
        key = str(local.get("canonical_event_id") or "")
        existing = existing_by_id.get(key)
        reconciled.append(merge_equivalent_rows(existing, local) if existing else dict(local))
    return reconciled, {"state": "RECONCILED", "rows": len(reconciled)}


def _sync_local_strict_rows(governed: Any, config: Any) -> dict[str, Any]:
    """Publish factual local master rows; never publish derived EMIR scores."""
    try:
        rows, _ = governed._read_strict_rows(config, "cak_forward_evidence")
        canonical: list[dict[str, Any]] = []
        for row in rows:
            try:
                item = canonicalize_emir_row(row)
            except Exception:
                continue
            if item.get("ticker") and item.get("evidence_date") and item.get("primary_source_url"):
                canonical.append(item)
        reconciled, reconcile_audit = _reconcile_with_shared(canonical)
        if str(reconcile_audit.get("state") or "") not in {"RECONCILED", "NO_ITEMS"}:
            return {"producer_rows": 0, **dict(reconcile_audit)}
        _, audit = upsert_canonical_forward_rows(reconciled, client_id="EMIR")
        return {
            "producer_rows": len(reconciled),
            "reconcile_state": str(reconcile_audit.get("state") or ""),
            **dict(audit),
        }
    except Exception as exc:
        return {
            "state": "PUBLISH_FAIL_SOFT",
            "producer_rows": 0,
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }


def install() -> dict[str, str]:
    import governed_evidence_bridge as governed
    import persistent_direct_evidence as persistent

    original = getattr(governed, "load_governed_evidence", None)
    if not callable(original):
        return {"patch_version": PATCH_VERSION, "state": "LOAD_FUNCTION_MISSING"}
    if getattr(original, "__shared_canonical_forward_v101__", False):
        return {"patch_version": PATCH_VERSION, "state": "ALREADY_INSTALLED"}

    @wraps(original)
    def shared_load(config: Any, tickers: Iterable[str] | None = None, *, as_of: Any = None):
        # Preserve management/capital evidence and existing fail-soft audit path.
        result = original(config, tickers, as_of=as_of)
        result = dict(result or {})

        publish_audit = _sync_local_strict_rows(governed, config)
        canonical_rows, shared_audit = read_canonical_forward_rows(tickers, client_id="EMIR")
        if str(shared_audit.get("state") or "") == "SHARED_CANONICAL_FORWARD":
            emir_rows = canonical_rows_to_emir_rows(canonical_rows)
            result["official_forward_events"] = governed.forward_rows_to_events(emir_rows, as_of=as_of)

        audit_frames = []
        existing = result.get("audit")
        if isinstance(existing, pd.DataFrame) and not existing.empty:
            audit_frames.append(existing)
        audit_frames.append(pd.DataFrame([{
            "provider": "SHARED_CANONICAL_FORWARD",
            "status": str(shared_audit.get("state") or "UNKNOWN"),
            "rows": int(shared_audit.get("rows") or 0),
            "contract_version": str(shared_audit.get("contract_version") or ""),
            "producer_status": str(publish_audit.get("state") or ""),
            "producer_rows": int(publish_audit.get("producer_rows") or publish_audit.get("rows") or 0),
            "producer_reconcile_state": str(publish_audit.get("reconcile_state") or ""),
            "error": str(shared_audit.get("error") or publish_audit.get("error") or ""),
        }]))
        result["audit"] = pd.concat(audit_frames, ignore_index=True, sort=False)
        return result

    shared_load.__shared_canonical_forward_v101__ = True
    governed.load_governed_evidence = shared_load
    # persistent_direct_evidence imported the loader by value. Replace that
    # binding too so its durable evidence bundle consumes the same canonical facts.
    persistent.load_governed_evidence = shared_load
    return {
        "patch_version": PATCH_VERSION,
        "state": "INSTALLED",
        "facts": "SHARED_CANONICAL_FORWARD_AUTHORITATIVE_WHEN_READABLE",
        "producer": "CAK_FORWARD_EVIDENCE_FACTS_ONLY_RECONCILED",
        "scoring": "EMIR_FORWARD_ROWS_TO_EVENTS_UNCHANGED",
    }


__all__ = ["PATCH_VERSION", "install", "_reconcile_with_shared", "_sync_local_strict_rows"]
