from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
HARDENING = WORKFLOWS / "production-hardening.yml"
MANUAL_400 = WORKFLOWS / "manual-live-400-smoke.yml"


def _trigger_block(source: str) -> str:
    return source.split("on:\n", 1)[1].split("\npermissions:", 1)[0]


def test_main_hardening_is_offline_only_and_retains_deterministic_checks() -> None:
    source = HARDENING.read_text(encoding="utf-8")
    triggers = _trigger_block(source)
    assert "push:" in triggers and "branches: [main]" in triggers
    assert "pytest -q --disable-socket" in source
    assert "validation_v1_9_14_future_db_acceleration.py" in source
    assert "streamlit run app.py" in source
    for forbidden in (
        "collect_live_forward_evidence",
        "tools/live_scan_400_smoke.py",
        "validation_400_v1_6_3.py",
        "ZAPI_KEY",
        "SHARED_EVIDENCE_SUPABASE",
        "secrets.",
    ):
        assert forbidden not in source


def test_manual_400_is_dispatch_only_and_uses_an_explicit_ref() -> None:
    source = MANUAL_400.read_text(encoding="utf-8")
    triggers = _trigger_block(source)
    assert "workflow_dispatch:" in triggers
    assert "selected_ref:" in triggers and "required: true" in triggers
    for forbidden in (
        "push:",
        "pull_request:",
        "schedule:",
        "workflow_run:",
        "repository_dispatch:",
        "workflow_call:",
    ):
        assert forbidden not in triggers
    assert "ref: ${{ inputs.selected_ref }}" in source
    assert "persist-credentials: false" in source


def test_manual_400_is_bounded_read_only_and_emits_clear_artifacts() -> None:
    source = MANUAL_400.read_text(encoding="utf-8")
    assert "permissions:\n  contents: read" in source
    assert "timeout-minutes: 40" in source
    assert "collect_live_forward_evidence" in source
    assert "python validation_400_v1_6_3.py" in source
    assert "actions/upload-artifact@" in source
    assert "live-forward-audit.csv" in source
    assert "selected-commit-sha.txt" in source
    assert "secrets." not in source
    for forbidden in ("git push", "git commit", "workflow_run", "repository_dispatch"):
        assert forbidden not in source


def test_no_automatic_workflow_or_cross_repository_dispatch_chain() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(WORKFLOWS.glob("*.yml"))
    )
    for forbidden in (
        "workflow_run:",
        "repository_dispatch:",
        "gh workflow run",
        "/actions/workflows/",
        "/dispatches",
    ):
        assert forbidden not in combined


def test_expected_shared_credential_names_are_referenced_without_values() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(WORKFLOWS.glob("*.yml"))
    )
    for name in (
        "ZAPI_KEY",
        "SHARED_EVIDENCE_SUPABASE_URL",
        "SHARED_EVIDENCE_SUPABASE_SERVICE_ROLE_KEY",
    ):
        assert f"secrets.{name}" in combined
