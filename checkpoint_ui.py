from __future__ import annotations


def checkpoint_execution_state(
    persisted_status: object,
    *,
    auto_continue: bool,
    checkpoint_in_progress: bool = False,
) -> str:
    """Describe browser execution without changing the canonical job status."""
    status = str(persisted_status or "UNKNOWN").strip().upper()
    if status == "RUNNING" and not auto_continue and not checkpoint_in_progress:
        return "RUNNING (persisted) / WAITING_FOR_CONTINUE"
    return status
