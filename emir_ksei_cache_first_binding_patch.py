from __future__ import annotations

"""Apply canonical KSEI completion after EMIR's cache-first boundary.

The scan consumes ``fetch_ksei_cache_first``.  Fresh cached per-security profiles
can therefore bypass lower network-fetch wrappers entirely.  Complete the
returned profile frame from Shared Hub and rebind resumable_scan's import-by-value
reference.  No regulatory free-float or beneficial-owner inference is made.
"""

from functools import wraps
from typing import Any, Iterable

import pandas as pd

import ksei_monthly_field_completion_patch as completion
from shared_ksei_runtime_context_patch import read_shared_profiles

PATCH_VERSION = "1.0.0-postproof-ksei-cache-first-binding"


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _requested_from_call(args: tuple[Any, ...], kwargs: dict[str, Any], profiles: pd.DataFrame) -> list[str]:
    candidate: Any = None
    if len(args) >= 2:
        candidate = args[1]
    if candidate is None:
        candidate = kwargs.get("tickers", kwargs.get("chunk"))
    values: list[Any]
    if isinstance(candidate, (list, tuple, set, pd.Series)):
        values = list(candidate)
    else:
        values = []
    if not values and isinstance(profiles, pd.DataFrame) and not profiles.empty and "ticker" in profiles.columns:
        values = profiles["ticker"].tolist()
    return list(dict.fromkeys(_ticker(value) for value in values if _ticker(value)))


def _complete_cache_first_profiles(profiles: Any, requested: Iterable[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = profiles.copy() if isinstance(profiles, pd.DataFrame) else pd.DataFrame()
    wanted = list(dict.fromkeys(_ticker(value) for value in requested if _ticker(value)))
    shared, audit = read_shared_profiles(wanted)
    completed, changed = completion._supplement_profiles(frame, wanted, shared)
    return completed, {
        "state": str(audit.get("state") or "UNKNOWN"),
        "rows": int(audit.get("rows") or 0),
        "changed": int(changed),
        "detail": str(audit.get("semantics") or audit.get("error") or ""),
    }


def install() -> dict[str, str]:
    import persistent_cache
    import resumable_scan

    original = getattr(persistent_cache, "fetch_ksei_cache_first", None)
    if not callable(original):
        return {"patch_version": PATCH_VERSION, "state": "CACHE_FIRST_FUNCTION_MISSING"}
    if getattr(original, "__canonical_ksei_cache_first_v1__", False):
        resumable_scan.fetch_ksei_cache_first = original
        return {"patch_version": PATCH_VERSION, "state": "ALREADY_INSTALLED"}

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any):
        result = original(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 4:
            return result
        profiles, actions, audit, cache_rows = result
        frame = profiles.copy() if isinstance(profiles, pd.DataFrame) else pd.DataFrame()
        requested = _requested_from_call(args, kwargs, frame)
        completed, shared_audit = _complete_cache_first_profiles(frame, requested)

        audit_frames = [item for item in (audit,) if isinstance(item, pd.DataFrame) and not item.empty]
        audit_frames.append(pd.DataFrame([{
            "provider": "SHARED_CANONICAL_KSEI_CACHE_FIRST",
            "status": shared_audit["state"],
            "items": shared_audit["changed"],
            "rows": shared_audit["rows"],
            "detail": shared_audit["detail"],
        }]))
        audit_out = pd.concat(audit_frames, ignore_index=True, sort=False)
        return completed, actions, audit_out, cache_rows

    wrapped.__canonical_ksei_cache_first_v1__ = True
    persistent_cache.fetch_ksei_cache_first = wrapped
    # resumable_scan imported this function by value, so patch that binding too.
    resumable_scan.fetch_ksei_cache_first = wrapped
    return {
        "patch_version": PATCH_VERSION,
        "state": "INSTALLED",
        "boundary": "FETCH_KSEI_CACHE_FIRST_OUTPUT",
        "source": "SHARED_CANONICAL_KSEI",
        "regulatory_free_float": "NOT_INFERRED",
        "beneficial_ownership": "NOT_INFERRED",
        "authorization": "UNCHANGED",
    }


__all__ = ["PATCH_VERSION", "install", "_complete_cache_first_profiles", "_requested_from_call"]
