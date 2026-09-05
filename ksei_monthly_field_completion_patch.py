from __future__ import annotations

"""Field-level completion for verified KSEI profiles.

The per-security KSEI endpoint can return a verified/OK profile with useful
identity/share-count information while leaving the scripless/local/foreign
composition empty.  The existing monthly fallback treated any verified ticker
as complete, so those rows blocked the official monthly archive from filling
missing composition fields.

This patch keeps every meaningful per-security value and uses the official KSEI
monthly holding-composition archive only to complete missing composition facts.
It never infers regulatory free float or beneficial ownership.
"""

from functools import wraps
from typing import Any, Iterable

import numpy as np
import pandas as pd

import ksei_monthly_holding_composition_patch as monthly


PATCH_VERSION = "1.0.0-ksei-monthly-field-completion"
COMPOSITION_STATE = "OFFICIAL_KSEI_SCRIPLESS_COMPOSITION_NOT_REGULATORY_FREE_FLOAT"


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _positive(value: Any) -> bool:
    number = _finite(value)
    return bool(np.isfinite(number) and number > 0)


def _composition_pair_complete(local: Any, foreign: Any) -> bool:
    left, right = _finite(local), _finite(foreign)
    if not (np.isfinite(left) and np.isfinite(right) and left >= 0 and right >= 0):
        return False
    return bool(99.0 <= left + right <= 101.0)


def _composition_complete(row: Any) -> bool:
    if row is None:
        return False
    getter = row.get if hasattr(row, "get") else lambda _key, _default=None: _default
    return _positive(getter("scripless_pct")) and _composition_pair_complete(
        getter("local_pct"), getter("foreign_pct")
    )


def _meaningful(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _merge_monthly_into_existing(existing: dict[str, Any], supplement: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Complete only missing/incoherent composition while preserving richer facts."""
    out = dict(existing)
    changed = False

    if not _positive(out.get("total_shares")) and _positive(supplement.get("total_shares")):
        out["total_shares"] = supplement.get("total_shares")
        changed = True

    if not _positive(out.get("scripless_pct")) and _positive(supplement.get("scripless_pct")):
        out["scripless_pct"] = supplement.get("scripless_pct")
        changed = True

    if not _composition_pair_complete(out.get("local_pct"), out.get("foreign_pct")) and _composition_pair_complete(
        supplement.get("local_pct"), supplement.get("foreign_pct")
    ):
        out["local_pct"] = supplement.get("local_pct")
        out["foreign_pct"] = supplement.get("foreign_pct")
        changed = True

    monthly_fields = (
        "ksei_monthly_local_total_shares",
        "ksei_monthly_foreign_total_shares",
        "ksei_monthly_local_individual_shares",
        "ksei_monthly_foreign_individual_shares",
        "ksei_monthly_institutional_shares",
    )
    for field in monthly_fields:
        if not _meaningful(out.get(field)) and _meaningful(supplement.get(field)):
            out[field] = supplement.get(field)
            changed = True

    if changed:
        out["ksei_monthly_holding_composition_state"] = COMPOSITION_STATE
        out["ksei_monthly_source_url"] = supplement.get("ksei_source_url")
        out["ksei_monthly_observed_on"] = supplement.get("ksei_observed_on")
        out["ksei_composition_completion_state"] = "OFFICIAL_MONTHLY_FIELD_SUPPLEMENT"
        out["ksei_source_verified"] = bool(out.get("ksei_source_verified")) or bool(supplement.get("ksei_source_verified"))
    return out, changed


def _supplement_profiles(
    profiles: pd.DataFrame,
    requested: list[str],
    monthly_profiles: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    base = profiles.copy() if isinstance(profiles, pd.DataFrame) else pd.DataFrame()
    supplement = monthly_profiles.copy() if isinstance(monthly_profiles, pd.DataFrame) else pd.DataFrame()

    if not base.empty and "ticker" in base.columns:
        base["ticker"] = base["ticker"].map(_ticker)
        base = base.drop_duplicates("ticker", keep="last").reset_index(drop=True)
    if not supplement.empty and "ticker" in supplement.columns:
        supplement["ticker"] = supplement["ticker"].map(_ticker)
        supplement = supplement.drop_duplicates("ticker", keep="last").reset_index(drop=True)

    base_by_ticker = {
        str(row.get("ticker")): dict(row)
        for row in base.to_dict(orient="records")
        if _ticker(row.get("ticker"))
    }
    monthly_by_ticker = {
        str(row.get("ticker")): dict(row)
        for row in supplement.to_dict(orient="records")
        if _ticker(row.get("ticker"))
    }

    changed = 0
    for ticker in requested:
        add = monthly_by_ticker.get(ticker)
        if add is None:
            continue
        existing = base_by_ticker.get(ticker)
        if existing is None:
            base_by_ticker[ticker] = add
            changed += 1
            continue
        merged, did_change = _merge_monthly_into_existing(existing, add)
        base_by_ticker[ticker] = merged
        changed += int(did_change)

    ordered = []
    seen: set[str] = set()
    for ticker in requested:
        if ticker in base_by_ticker and ticker not in seen:
            ordered.append(base_by_ticker[ticker])
            seen.add(ticker)
    for ticker, row in base_by_ticker.items():
        if ticker not in seen:
            ordered.append(row)
    return pd.DataFrame(ordered), changed


def _wrap_fetch_many(module: Any) -> None:
    original = getattr(module, "fetch_many_ksei_profiles", None)
    if not callable(original) or getattr(original, "__ksei_monthly_field_completion_v1__", False):
        return

    @wraps(original)
    def wrapped(tickers: Iterable[str], max_workers: int = 2):
        requested = list(dict.fromkeys(_ticker(value) for value in tickers if _ticker(value)))
        profiles, actions, audit = original(requested, max_workers=max_workers)
        frame = profiles.copy() if isinstance(profiles, pd.DataFrame) else pd.DataFrame()
        by_ticker: dict[str, dict[str, Any]] = {}
        if not frame.empty and "ticker" in frame.columns:
            frame["ticker"] = frame["ticker"].map(_ticker)
            by_ticker = {
                str(row.get("ticker")): dict(row)
                for row in frame.to_dict(orient="records")
                if _ticker(row.get("ticker"))
            }

        incomplete = [ticker for ticker in requested if not _composition_complete(by_ticker.get(ticker))]
        monthly_frame = monthly_audit = pd.DataFrame()
        if incomplete:
            monthly_frame, monthly_audit = monthly.fetch_monthly_profiles(incomplete)
        completed, changed = _supplement_profiles(frame, requested, monthly_frame)

        audit_frames = [item for item in (audit, monthly_audit) if isinstance(item, pd.DataFrame) and not item.empty]
        if incomplete:
            audit_frames.append(pd.DataFrame([{
                "provider": "KSEI_MONTHLY_FIELD_COMPLETION",
                "status": "SUPPLEMENTED" if changed else "NO_SUPPLEMENT_AVAILABLE",
                "items": int(changed),
                "detail": f"incomplete_requested={len(incomplete)}; semantics=scripless composition not regulatory free float",
            }]))
        audit_out = pd.concat(audit_frames, ignore_index=True, sort=False) if audit_frames else pd.DataFrame()
        return completed, actions, audit_out

    wrapped.__ksei_monthly_field_completion_v1__ = True
    setattr(module, "fetch_many_ksei_profiles", wrapped)


def install() -> dict[str, str]:
    import autonomous_enrichment
    import persistent_cache

    _wrap_fetch_many(autonomous_enrichment)
    _wrap_fetch_many(persistent_cache)
    return {
        "patch_version": PATCH_VERSION,
        "state": "INSTALLED",
        "semantics": "FIELD_COMPLETION_FROM_OFFICIAL_KSEI_MONTHLY_COMPOSITION",
        "regulatory_free_float": "NOT_INFERRED",
    }


__all__ = [
    "PATCH_VERSION",
    "COMPOSITION_STATE",
    "_composition_complete",
    "_merge_monthly_into_existing",
    "_supplement_profiles",
    "install",
]
