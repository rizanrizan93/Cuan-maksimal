from __future__ import annotations

import numpy as np
import pandas as pd

import persistent_direct_evidence as persistent
from future_fundamental import calculate_future_fundamental
from governed_evidence_bridge import (
    forward_rows_to_events,
    management_rows_to_events,
    persistent_forward_payload_is_strict,
)


def _forward(**updates):
    row = {
        "ticker": "AAA.JK",
        "evidence_type": "CAPEX_AND_EXPANSION_PLAN",
        "evidence_date": "2026-07-10",
        "observed_at": "2026-08-15T00:00:00Z",
        "title": "Capacity expansion capex plan",
        "value_numeric": 500_000_000_000,
        "unit": "IDR",
        "horizon": "2026_PLUS",
        "source_url": "https://issuer.example/capex.pdf",
        "source_family": "ISSUER_PRESENTATION|IDX_DISCLOSURE",
        "source_quorum_count": 2,
        "source_quorum_verified": True,
        "entity_match_verified": True,
        "source_verified": True,
        "evidence_confidence": 0.95,
        "payload": {"secondary_url": "https://issuer.example/disclosure", "milestone_state": "IN_PROGRESS"},
    }
    row.update(updates)
    return row


def test_strict_forward_raw_row_becomes_future_fundamental_event():
    events = forward_rows_to_events([_forward()], as_of="2026-08-15T00:00:00Z")
    assert len(events) == 1
    event = events.iloc[0]
    assert event["source_tier"] == "ISSUER"
    assert bool(event["source_verified"]) is True
    assert bool(event["source_quorum_verified"]) is True
    assert event["event_role"] == "FORWARD_FUNDAMENTAL"

    future = calculate_future_fundamental(
        ticker="AAA.JK",
        events=events,
        narrative={},
        fundamental={},
        ownership={},
        sector={},
        as_of="2026-08-15T00:00:00Z",
    )
    assert int(future["future_verified_forward_event_count"]) >= 1
    assert int(future["future_official_forward_event_count"]) >= 1
    assert np.isfinite(float(future["future_direct_forward_visibility_score"]))


def test_forward_quorum_and_entity_match_are_fail_closed():
    assert forward_rows_to_events([
        _forward(source_quorum_verified=False, source_quorum_count=1),
        _forward(entity_match_verified=False),
        _forward(source_url="http://issuer.example/capex.pdf"),
    ], as_of="2026-08-15T00:00:00Z").empty


def test_management_roster_is_administrative_but_directional_actions_are_eligible():
    common = {
        "ticker": "AAA.JK",
        "evidence_date": "2026-07-01",
        "observed_at": "2026-08-15T00:00:00Z",
        "source_url": "https://issuer.example/action.pdf",
        "source_family": "ISSUER_GOVERNANCE|IDX_DISCLOSURE",
        "source_quorum_count": 2,
        "source_quorum_verified": True,
        "entity_match_verified": True,
        "source_verified": True,
        "evidence_confidence": 0.95,
    }
    rows = [
        {**common, "evidence_type": "BOARD_ROLE", "person_or_holder": "Director A", "role_or_action": "PRESIDENT_DIRECTOR"},
        {**common, "evidence_type": "RIGHTS_ISSUE", "person_or_holder": None, "role_or_action": "HMETD"},
        {**common, "evidence_type": "CAPEX_DECISION", "person_or_holder": None, "role_or_action": "CAPEX_APPROVAL"},
    ]
    events = management_rows_to_events(rows, as_of="2026-08-15T00:00:00Z").set_index("category")
    assert bool(events.loc["BOARD_ROLE", "narrative_eligible"]) is False
    assert events.loc["BOARD_ROLE", "event_role"] == "GOVERNANCE_ADMINISTRATIVE"
    assert bool(events.loc["DILUTION_EQUITY_RAISE", "narrative_eligible"]) is True
    assert bool(events.loc["CAPEX_DECISION", "narrative_eligible"]) is True
    assert events.loc["CAPEX_DECISION", "event_role"] == "FORWARD_FUNDAMENTAL"


def test_legacy_persistent_forward_requires_explicit_quorum_entity_and_https():
    weak = {
        "source_verified": True,
        "source_url": "https://issuer.example/forward",
        "source_tier": "ISSUER",
    }
    assert persistent_forward_payload_is_strict(weak) is False
    strict = {
        **weak,
        "source_quorum_verified": True,
        "source_quorum_count": 2,
        "entity_match_verified": True,
    }
    assert persistent_forward_payload_is_strict(strict) is True


class _ReadyConfig:
    ready = True


def test_loader_blocks_weak_persistent_forward_and_prefers_governed_raw(monkeypatch):
    weak_persistent = [{
        "evidence_key": "WEAK_TSPC",
        "source_scan_id": "old",
        "ticker": "TSPC.JK",
        "evidence_type": "OFFICIAL_FORWARD_EVENT",
        "observed_at": "2026-07-15T00:00:00Z",
        "source_verified": True,
        "source_url": "https://issuer.example/tspc-jv",
        "payload": {
            "title": "TSPC joint venture",
            "url": "https://issuer.example/tspc-jv",
            "source_tier": "ISSUER",
        },
        "freshness_policy_days": 540,
        "revoked": False,
        "last_seen_at": "2026-08-15T00:00:00Z",
    }]

    governed_forward = forward_rows_to_events([_forward(ticker="OMED.JK")], as_of="2026-08-15T00:00:00Z")
    management = management_rows_to_events([
        {
            "ticker": "OMED.JK", "evidence_type": "BOARD_ROLE", "evidence_date": "2026-07-01",
            "observed_at": "2026-08-15T00:00:00Z", "person_or_holder": "Director A",
            "role_or_action": "PRESIDENT_DIRECTOR", "source_url": "https://issuer.example/board",
            "source_family": "ISSUER_GOVERNANCE|IDX_DISCLOSURE", "source_quorum_count": 2,
            "source_quorum_verified": True, "entity_match_verified": True, "source_verified": True,
        },
        {
            "ticker": "OMED.JK", "evidence_type": "CAPEX_DECISION", "evidence_date": "2026-07-01",
            "observed_at": "2026-08-15T00:00:00Z", "person_or_holder": None,
            "role_or_action": "CAPEX_APPROVAL", "source_url": "https://issuer.example/capex-decision",
            "source_family": "ISSUER_GOVERNANCE|IDX_DISCLOSURE", "source_quorum_count": 2,
            "source_quorum_verified": True, "entity_match_verified": True, "source_verified": True,
        },
    ], as_of="2026-08-15T00:00:00Z")

    monkeypatch.setattr(persistent, "_read_rows", lambda *args, **kwargs: weak_persistent)
    monkeypatch.setattr(persistent, "load_governed_evidence", lambda *args, **kwargs: {
        "official_forward_events": governed_forward,
        "management_capital_events": management,
        "audit": pd.DataFrame([{"provider": "GOVERNED_EVIDENCE_BRIDGE", "status": "DATABASE_CURRENT"}]),
    })

    result = persistent.load_verified_direct_evidence(
        _ReadyConfig(), ["OMED.JK", "TSPC.JK"], as_of="2026-08-15T00:00:00Z"
    )
    scoring = result["official_forward_events"]
    assert "TSPC.JK" not in set(scoring.get("ticker", pd.Series(dtype=str)).astype(str))
    assert "OMED.JK" in set(scoring.get("ticker", pd.Series(dtype=str)).astype(str))
    assert (scoring.get("category", pd.Series(dtype=str)).astype(str) == "CAPEX_DECISION").any()
    assert not (scoring.get("category", pd.Series(dtype=str)).astype(str) == "BOARD_ROLE").any()
    assert (result["management_capital_events"].get("category", pd.Series(dtype=str)).astype(str) == "BOARD_ROLE").any()
    assert (result["audit"].get("status", pd.Series(dtype=str)).astype(str) == "PERSISTED_FORWARD_BLOCKED_MISSING_STRICT_LINEAGE").any()
