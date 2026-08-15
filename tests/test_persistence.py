from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistence import BRIDGE_VERSION, DATABASE_SCHEMA_VERSION, _headers, config_from_mapping, database_status  # noqa: E402


def test_database_disabled_without_key():
    config = config_from_mapping({"CAK_DATABASE_ENABLED": "true", "SUPABASE_URL": "https://x.supabase.co"})
    assert config.ready is False


def test_secret_key_is_preferred_and_not_bearer():
    config = config_from_mapping({
        "CAK_DATABASE_ENABLED": "true", "SUPABASE_URL": "https://x.supabase.co",
        "SUPABASE_SECRET_KEY": "sb_secret_example", "SUPABASE_SERVICE_ROLE_KEY": "a.b.c",
    })
    headers = _headers(config)
    assert config.key_type == "SECRET"
    assert headers["apikey"] == "sb_secret_example"
    assert "Authorization" not in headers


def test_legacy_service_role_uses_bearer():
    config = config_from_mapping({
        "CAK_DATABASE_ENABLED": "true", "SUPABASE_URL": "https://x.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "aaa.bbb.ccc",
    })
    headers = _headers(config)
    assert config.key_type == "LEGACY_SERVICE_ROLE"
    assert headers["Authorization"].startswith("Bearer ")


def test_publishable_key_rejected():
    config = config_from_mapping({
        "CAK_DATABASE_ENABLED": "true", "SUPABASE_URL": "https://x.supabase.co",
        "SUPABASE_SECRET_KEY": "sb_publishable_bad",
    })
    assert config.ready is False
    assert config.key_type == "PUBLISHABLE_REJECTED"


def test_database_status_contract_v7():
    config = config_from_mapping({
        "CAK_DATABASE_ENABLED": "true", "SUPABASE_URL": "https://x.supabase.co",
        "SUPABASE_SECRET_KEY": "sb_secret_example",
    })
    status = database_status(config)
    assert status["database_mode"] == "SUPABASE_REST"
    assert status["schema_version"] == "emir_autonomous_schema_v9"
    assert status["write_policy"] == "RESUMABLE_CHUNK_CHECKPOINT_PLUS_BEST_EFFORT_RESULTS"
    assert DATABASE_SCHEMA_VERSION.endswith("v9")
    assert BRIDGE_VERSION == "1.9.1"


def test_direct_evidence_payload_contract():
    import pandas as pd
    from persistence import _direct_evidence_payload

    frame = pd.DataFrame([{
        "ticker": "ADMR.JK", "evidence_type": "ORDERBOOK_BID_OFFER",
        "observed_at": pd.Timestamp("2026-08-03T10:00:00+07:00"),
        "source_verified": True, "resistance_price": 1500,
    }])
    payload = _direct_evidence_payload("scan123", frame)
    assert len(payload) == 1
    assert payload[0]["evidence_id"].startswith("scan123:ORDERBOOK_BID_OFFER:ADMR.JK")
    assert payload[0]["source_verified"] is True
    assert payload[0]["payload"]["resistance_price"] == 1500


def test_migration_v4_declares_outcome_memory_table():
    migration = (ROOT / "database" / "migration_v4.sql").read_text()
    verification = (ROOT / "database" / "verify_v4.sql").read_text()
    assert "cak_outcome_memory" in migration
    assert "outcome_verified" in migration
    assert "cak_outcome_memory" in verification


def test_migration_v5_declares_autonomous_evidence_table():
    migration = (ROOT / "database" / "migration_v5.sql").read_text()
    verification = (ROOT / "database" / "verify_v5.sql").read_text()
    assert "cak_autonomous_evidence" in migration
    assert "source_verified" in migration
    assert "cak_autonomous_evidence" in verification


def test_migration_v6_declares_persistent_cache_tables():
    migration = (ROOT / "database" / "migration_v6.sql").read_text()
    verification = (ROOT / "database" / "verify_v6.sql").read_text()
    assert "cak_ohlcv_cache" in migration
    assert "cak_source_cache" in migration
    assert "content_sha256" in migration
    assert "cak_ohlcv_cache" in verification
    assert "cak_source_cache" in verification
