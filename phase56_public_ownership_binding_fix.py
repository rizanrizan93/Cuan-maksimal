from __future__ import annotations

"""Bind Phase 5.6 public ownership context across positional profile calls.

The original projection patch correctly kept Yahoo/public concentration separate
from KSEI/free-float evidence, but its profile wrapper only inspected
``kwargs['ownership']``. ``build_emir_profile`` is also called positionally in
production, so valid context could disappear before persistence.

This patch changes transport/binding only. It does not alter ownership_score,
ownership_coverage_pct, IDX integrity, free-float verification, or execution
authorization.
"""

import inspect
from typing import Any, Mapping

PATCH_VERSION = "1.0.0-phase5.6-public-ownership-positional-binding"


def _bind_ownership(profile_callable: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    ownership = kwargs.get("ownership") if isinstance(kwargs.get("ownership"), Mapping) else None
    if ownership is not None:
        return ownership
    try:
        bound = inspect.signature(profile_callable).bind_partial(*args, **dict(kwargs))
        candidate = bound.arguments.get("ownership")
        return candidate if isinstance(candidate, Mapping) else {}
    except Exception:
        return {}


def _attach_context(result: Any, ownership: Mapping[str, Any], context_fields: tuple[str, ...]) -> Any:
    if not isinstance(result, Mapping):
        return result
    output = dict(result)
    for field in context_fields:
        if field in ownership:
            output[field] = ownership[field]
    if any(field in ownership for field in context_fields):
        output["ownership_public_context_state"] = "CONTEXT_ONLY_NOT_KSEI_FREE_FLOAT"
        output["ownership_public_context_score_eligible"] = False
        output["ownership_public_context_execution_eligible"] = False
    return output


def install() -> dict[str, str]:
    import narrative_flow_engine as engine
    import phase56_public_ownership_projection as public_projection
    import resumable_scan as scan

    if getattr(scan, "_phase56_public_ownership_binding_fix", "") == PATCH_VERSION:
        return {"patch_version": PATCH_VERSION, "state": "ALREADY_INSTALLED"}

    canonical_profile = engine.build_emir_profile
    current_scan_profile = scan.build_emir_profile
    context_fields = tuple(public_projection.CONTEXT_FIELDS)

    def scan_profile_with_bound_context(*args: Any, **kwargs: Any):
        ownership = _bind_ownership(canonical_profile, args, kwargs)
        result = current_scan_profile(*args, **kwargs)
        return _attach_context(result, ownership, context_fields)

    def canonical_profile_with_bound_context(*args: Any, **kwargs: Any):
        ownership = _bind_ownership(canonical_profile, args, kwargs)
        result = canonical_profile(*args, **kwargs)
        return _attach_context(result, ownership, context_fields)

    scan.build_emir_profile = scan_profile_with_bound_context
    engine.build_emir_profile = canonical_profile_with_bound_context
    scan._phase56_public_ownership_binding_fix = PATCH_VERSION
    return {
        "patch_version": PATCH_VERSION,
        "binding": "POSITIONAL_AND_KEYWORD",
        "policy": "PUBLIC_CONTEXT_ONLY_NO_SCORE_OR_EXECUTION_AUTHORIZATION",
    }


__all__ = ["PATCH_VERSION", "install", "_bind_ownership", "_attach_context"]
