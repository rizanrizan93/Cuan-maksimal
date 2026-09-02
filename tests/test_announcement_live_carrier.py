from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "announcement-live-validation.yml"
SCRIPT = ROOT / "tools" / "phase5_6_announcement_live_validation.py"


def test_announcement_live_consumer_is_manual_cache_only_and_read_only() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "schedule:" not in workflow
    assert "\npush:" not in workflow
    assert "contents: read" in workflow
    assert "contents: write" not in workflow
    assert "persist-credentials: false" in workflow
    assert "ref: ${{ inputs.selected_ref }}" in workflow
    assert "ZAPI_KEY:" not in workflow
    assert "SHARED_EVIDENCE_SUPABASE_SERVICE_ROLE_KEY" in workflow
    assert "--mode consumer" in workflow
    assert "--client-id EMIR" in workflow

    assert 'api_key=None if args.mode == "producer" else ""' in script
    assert "CONSUMER_NETWORK_BUDGET_VIOLATION" in script
    assert "CONSUMER_REQUEST_NOT_AVOIDED" in script
    assert "FORBIDDEN_SHARED_SEMANTICS" in script
