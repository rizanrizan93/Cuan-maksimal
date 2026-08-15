"""Synchronize long-lived Streamlit workers to the on-disk release."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping, Sequence


def refresh_release_runtime(
    *,
    reload_order: Sequence[str],
    version_markers: Mapping[str, str],
) -> tuple[str, tuple[str, ...]]:
    importlib.invalidate_caches()
    contract = importlib.import_module("release_contract")
    contract = importlib.reload(contract)
    expected = str(contract.SCANNER_RELEASE_VERSION)
    stale = any(
        module_name in sys.modules
        and str(getattr(sys.modules[module_name], attribute, "")) != expected
        for module_name, attribute in version_markers.items()
    )
    if not stale:
        return expected, ()
    reloaded: list[str] = []
    for module_name in reload_order:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        importlib.reload(module)
        reloaded.append(module_name)
    return expected, tuple(reloaded)


__all__ = ["refresh_release_runtime"]
