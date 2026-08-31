from pathlib import Path

import pandas as pd

import persistence
import zapi_flow_enrichment as zapi
from public_idx_broker_flow import PUBLIC_CACHE_URL


ROOT = Path(__file__).resolve().parents[1]
IDX_FLOW_REF = "djqvhbeonmicztxfisav"
FORBIDDEN_TABLES = {
    "flow_scan_runs", "flow_scan_results", "flow_vendor_foreign_flows",
    "flow_official_stock_flows", "flow_broker_flows", "flow_daily_prices",
    "flow_signal_outcomes",
}


def _runtime_text() -> str:
    paths = [path for path in ROOT.glob("*.py")]
    paths.extend(ROOT.glob("scripts/*.py"))
    paths.extend(ROOT.glob(".github/workflows/*.yml"))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def test_emir_runtime_has_no_idx_flow_artifact_or_table_dependency():
    text = _runtime_text().lower()
    assert "rizanrizan93/idx-flow-scanner" not in text
    assert "import idx_flow_scanner" not in text
    assert "from idx_flow_scanner" not in text
    assert not (FORBIDDEN_TABLES & set(text.replace('"', " ").replace("'", " ").split()))


def test_emir_cache_urls_are_emir_owned():
    assert "/rizanrizan93/Cuan-maksimal/" in zapi.OWNED_CACHE_URL
    assert "/rizanrizan93/Cuan-maksimal/" in PUBLIC_CACHE_URL


def test_emir_rejects_idx_flow_supabase_project():
    config = persistence.config_from_mapping({
        "CAK_DATABASE_ENABLED": "true",
        "SUPABASE_URL": f"https://{IDX_FLOW_REF}.supabase.co",
        "SUPABASE_SECRET_KEY": "sb_secret_fixture",
    })
    assert not config.ready
    assert config.key == ""
    assert config.key_type == "CROSS_SCANNER_PROJECT_REJECTED"


def test_missing_zapi_fields_do_not_become_zero_evidence():
    payload = {"data": {"date": "2026-08-28", "unit": "shares", "data": [{"code": "ELSA"}]}}
    assert zapi._normalize_foreign_payload(payload, "2026-08-28", {"ELSA"}).empty


def test_zapi_provider_failure_stays_fail_closed(monkeypatch):
    monkeypatch.setattr(zapi, "_load_owned_cache", lambda: (pd.DataFrame(), {"state": "FAIL_SOFT", "provider": "EMIR_OWNED_ZAPI_CACHE"}))
    monkeypatch.setattr(zapi, "_secret", lambda _name: "")
    history, meta = zapi._load_history_for_universe(["ELSA"])
    assert history.empty
    assert meta["state"] == "NO_ZAPI_KEY"
    assert meta["rows"] == 0
