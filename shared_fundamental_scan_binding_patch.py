from __future__ import annotations

"""Bind patched persistent-cache callables into already-imported EMIR scan modules."""

import sys


PATCH_VERSION = "1.0.0-phase5.6-binding"


def install() -> None:
    import persistent_cache as cache

    scan = sys.modules.get("resumable_scan")
    if scan is None:
        return
    for name in (
        "fetch_fundamental_cache_first",
        "fetch_idx_official_fundamental_cache_first",
        "load_cached_fundamentals",
        "load_cached_idx_official_fundamentals",
    ):
        setattr(scan, name, getattr(cache, name))


__all__ = ["PATCH_VERSION", "install"]
