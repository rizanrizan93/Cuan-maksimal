from checkpoint_ui import checkpoint_execution_state


def test_running_job_without_auto_continue_is_shown_waiting_at_checkpoint():
    assert checkpoint_execution_state("RUNNING", auto_continue=False) == (
        "RUNNING (persisted) / WAITING_FOR_CONTINUE"
    )


def test_auto_continue_or_active_checkpoint_keeps_canonical_running_label():
    assert checkpoint_execution_state("RUNNING", auto_continue=True) == "RUNNING"
    assert checkpoint_execution_state(
        "RUNNING", auto_continue=False, checkpoint_in_progress=True
    ) == "RUNNING"


def test_non_running_status_is_not_reinterpreted():
    assert checkpoint_execution_state("PAUSED", auto_continue=False) == "PAUSED"
