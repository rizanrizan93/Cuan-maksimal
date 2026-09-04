from phase56_public_ownership_binding_fix import _attach_context, _bind_ownership


def _profile(ticker, market, sector, ownership, narrative=None):
    return {"ticker": ticker, "ownership_score": 55.0}


def test_positional_ownership_argument_is_bound():
    ownership = {
        "ownership_public_context_coverage_pct": 100.0,
        "ownership_public_context_provenance_state": "PUBLIC_PROVIDER_YAHOO_CONCENTRATION_NOT_IDX_KSEI",
    }
    bound = _bind_ownership(_profile, ("ADMR.JK", {}, {}, ownership), {})
    assert bound is ownership


def test_keyword_ownership_argument_is_bound():
    ownership = {"ownership_public_context_coverage_pct": 100.0}
    bound = _bind_ownership(_profile, ("ADMR.JK", {}, {}), {"ownership": ownership})
    assert bound is ownership


def test_public_context_never_relabels_strict_ownership_or_execution():
    result = {
        "ticker": "ADMR.JK",
        "ownership_score": 55.0,
        "ownership_coverage_pct": 33.8,
        "execution_authorized": False,
    }
    ownership = {
        "ownership_public_context_coverage_pct": 100.0,
        "ownership_public_context_provenance_state": "PUBLIC_PROVIDER_YAHOO_CONCENTRATION_NOT_IDX_KSEI",
    }
    attached = _attach_context(
        result,
        ownership,
        (
            "ownership_public_context_coverage_pct",
            "ownership_public_context_provenance_state",
        ),
    )
    assert attached["ownership_coverage_pct"] == 33.8
    assert attached["ownership_score"] == 55.0
    assert attached["execution_authorized"] is False
    assert attached["ownership_public_context_coverage_pct"] == 100.0
    assert attached["ownership_public_context_score_eligible"] is False
    assert attached["ownership_public_context_execution_eligible"] is False
