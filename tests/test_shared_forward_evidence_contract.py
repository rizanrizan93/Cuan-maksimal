from pathlib import Path

import emir_shared_forward_runtime_patch as runtime_patch
from shared_forward_evidence import (
    canonicalize_emir_row,
    canonicalize_pasticuan_row,
    canonical_rows_to_emir_rows,
    merge_equivalent_rows,
    profile_rows,
    strict_active,
)

ISSUER = "https://www.temposcangroup.com/en/read/Joint-Venture-Tempo-Scan-Group-and-Sino-Biopharmaceutical-Group"
BPOM = "https://www.pom.go.id/index.php/berita/bpom-dukung-kemajuan-industri-farmasi-nasional-melalui-kolaborasi-tempo-scan-dan-sino-biopharmaceutical"
EXPECTED_TSPC_ID = "d03e07557692f8bff630e47f1cf7e99312e7d3e6f0462f113dd35b23051e1859"


def emir_tspc():
    return {
        "evidence_id": "emir-tspc", "ticker": "TSPC.JK",
        "evidence_type": "JOINT_VENTURE_SPECIALTY_PHARMA", "evidence_date": "2026-07-15",
        "observed_at": "2026-07-24T00:00:00Z",
        "title": "PT Tempo CTTQ Biopharmaceutical Indonesia joint venture",
        "source_url": ISSUER, "source_family": "ISSUER_OFFICIAL_PRESS_RELEASE|BPOM_REGULATOR",
        "source_quorum_count": 2, "source_quorum_verified": True,
        "entity_match_verified": True, "source_verified": True, "evidence_confidence": 0.97,
        "payload": {"secondary_url": BPOM, "regulator_support_confirmed": True},
    }


def pasticuan_tspc():
    return {
        "snapshot_id": "pasticuan-tspc", "ticker": "TSPC.JK",
        "project_name": "PT Tempo CTTQ Biopharmaceutical Indonesia (TCBI) joint venture",
        "project_stage": "SIGNED_JOINT_VENTURE_REGULATORY_SUPPORT_IN_PROGRESS_2026",
        "project_source_families": "ISSUER_OFFICIAL_PRESS_RELEASE|BPOM_REGULATOR",
        "project_source_urls": f"{ISSUER}|{BPOM}", "project_source_quorum_verified": True,
        "source_quorum_count": 2, "entity_match_verified": True, "evidence_date": "2026-07-24",
    }


def test_contract_hash_matches_cross_repo_expected_identity():
    emir = canonicalize_emir_row(emir_tspc())
    pasticuan = canonicalize_pasticuan_row(pasticuan_tspc())
    assert emir["canonical_event_id"] == EXPECTED_TSPC_ID
    assert pasticuan["canonical_event_id"] == EXPECTED_TSPC_ID
    merged = merge_equivalent_rows(emir, pasticuan)
    assert merged["evidence_date"] == "2026-07-15"
    assert BPOM in merged["corroboration_urls"]


def test_emir_adapter_consumes_canonical_facts_not_external_scores():
    canonical = canonicalize_emir_row(emir_tspc())
    row = canonical_rows_to_emir_rows([canonical])[0]
    assert row["evidence_date"] == "2026-07-15"
    assert row["source_quorum_verified"] is True
    assert row["source_verified"] is True
    assert "materiality_score" not in row
    assert "financial_bridge_score" not in row
    assert "recommendation" not in row


def test_shared_profile_is_point_in_time_and_maha_stale():
    tspc = canonicalize_emir_row(emir_tspc())
    assert strict_active(tspc, as_of="2026-09-05T00:00:00Z")
    profile = profile_rows([tspc], ticker="TSPC.JK", as_of="2026-09-05T00:00:00Z")
    assert profile["shared_forward_active_direct_count"] == 1
    assert profile["shared_forward_contract_coverage_pct"] == 100.0

    maha = canonicalize_emir_row({
        "evidence_id": "maha", "ticker": "MAHA.JK", "evidence_type": "BACKLOG_LONG_TERM_CONTRACT",
        "evidence_date": "2024-06-05", "title": "Coal hauling contract extension through 2034",
        "source_url": "https://mha.co.id/PR/pdf/240604_MAHA-BYAN%20Contract%20v02.pdf",
        "source_family": "ISSUER_PRESS_RELEASE_PDF|ISSUER_PRESS_RELEASE_INDEX",
        "source_quorum_count": 2, "source_quorum_verified": True,
        "entity_match_verified": True, "source_verified": True,
        "payload": {"secondary_url": "https://mha.co.id/press-release"},
    })
    assert not strict_active(maha, as_of="2026-09-05T00:00:00Z")


def test_reconcile_preserves_pasticuan_provenance_and_factual_payload(monkeypatch):
    local = canonicalize_emir_row(emir_tspc())
    existing = dict(local)
    existing["producer_clients"] = ["PASTICUAN"]
    existing["producer_records"] = {"PASTICUAN": "OFFICIAL_FORWARD|TSPC|TCBI_JV_2026"}
    existing["payload"] = {
        "secondary_url": BPOM,
        "regulator_support_confirmed": True,
        "corroboration_publication_date": "2026-07-24",
    }

    def fake_read(tickers, *, client_id):
        assert tickers == ["TSPC.JK"]
        assert client_id == "EMIR"
        return [existing], {"state": "SHARED_CANONICAL_FORWARD", "rows": 1}

    monkeypatch.setattr(runtime_patch, "read_canonical_forward_rows", fake_read)
    rows, audit = runtime_patch._reconcile_with_shared([local])
    assert audit["state"] == "RECONCILED"
    assert len(rows) == 1
    row = rows[0]
    assert set(row["producer_clients"]) == {"EMIR", "PASTICUAN"}
    assert row["producer_records"]["PASTICUAN"] == "OFFICIAL_FORWARD|TSPC|TCBI_JV_2026"
    assert row["producer_records"]["EMIR"] == "emir-tspc"
    assert row["payload"]["corroboration_publication_date"] == "2026-07-24"


def test_local_emir_sync_publishes_only_reconciled_canonical_factual_shape(monkeypatch):
    class Governed:
        @staticmethod
        def _read_strict_rows(config, table):
            assert table == "cak_forward_evidence"
            return [emir_tspc()], []

    existing = canonicalize_pasticuan_row(pasticuan_tspc())
    captured = {}

    def fake_read(tickers, *, client_id):
        return [existing], {"state": "SHARED_CANONICAL_FORWARD", "rows": 1}

    def fake_upsert(rows, *, client_id):
        captured["rows"] = rows
        captured["client_id"] = client_id
        return rows, {"state": "UPSERTED", "rows": len(rows)}

    monkeypatch.setattr(runtime_patch, "read_canonical_forward_rows", fake_read)
    monkeypatch.setattr(runtime_patch, "upsert_canonical_forward_rows", fake_upsert)
    audit = runtime_patch._sync_local_strict_rows(Governed, object())
    assert audit["state"] == "UPSERTED"
    assert audit["reconcile_state"] == "RECONCILED"
    assert captured["client_id"] == "EMIR"
    row = captured["rows"][0]
    assert row["canonical_event_id"] == EXPECTED_TSPC_ID
    assert set(row["producer_clients"]) == {"EMIR", "PASTICUAN"}
    assert not any(key in row for key in ("materiality_score", "financial_bridge_score", "recommendation", "authorization"))


def test_emir_publish_fails_closed_when_shared_row_cannot_be_read(monkeypatch):
    local = canonicalize_emir_row(emir_tspc())

    def fake_read(tickers, *, client_id):
        return [], {"state": "READ_FAIL_SOFT", "error": "temporary"}

    monkeypatch.setattr(runtime_patch, "read_canonical_forward_rows", fake_read)
    rows, audit = runtime_patch._reconcile_with_shared([local])
    assert rows == []
    assert audit["state"] == "RECONCILE_READ_UNAVAILABLE"


def test_runtime_binding_installs_before_phase56_reuse_wrapper():
    text = Path("runtime_release.py").read_text()
    assert text.index('"emir_shared_forward_runtime_patch"') < text.index('"phase56_coverage_runtime_integrity_patch"')
