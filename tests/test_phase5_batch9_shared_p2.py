import numpy as np
import pandas as pd
import pytest

import data_providers
from data_providers import FetchResult, assess_benchmark_freshness, fetch_many_ohlcv
from idx_trading_calendar import (
    CalendarCoverageError, CalendarState, calendar_state, is_idx_session, latest_expected_completed_session,
    n_idx_sessions_ago, previous_idx_session, trading_session_age,
)
from provider_semantics import (
    EvidenceProvenance, ProviderResult, ProviderStatus, aggregate_provenance,
    canonicalize_provider_audit, normalize_provider_result, normalize_provenance,
)


def test_typed_provider_result_matrix_is_fail_closed_and_zero_safe():
    cases = [
        ({"status": "SUCCESS", "value": 7}, ProviderStatus.SUCCESS),
        (0, ProviderStatus.SUCCESS),
        ({"status": "PARTIAL", "value": [1]}, ProviderStatus.PARTIAL),
        ({"status": "MISSING"}, ProviderStatus.MISSING),
        ({"status": "STALE", "value": 3}, ProviderStatus.STALE),
        ({"status": "INVALID", "value": "bad"}, ProviderStatus.INVALID),
        ({"status": "ERROR", "value": 0, "error": "provider timeout"}, ProviderStatus.PROVIDER_ERROR),
        ({"status": "mystery", "value": 1}, ProviderStatus.INVALID),
        ({"status": "NOT_APPLICABLE"}, ProviderStatus.NOT_APPLICABLE),
        ("unrecognized legacy payload", ProviderStatus.INVALID),
        (None, ProviderStatus.MISSING), (float("nan"), ProviderStatus.MISSING),
        (pd.NA, ProviderStatus.MISSING), (True, ProviderStatus.INVALID),
        ({"status": "SUCCESS", "value": 1, "freshness": "STALE"}, ProviderStatus.STALE),
    ]
    results = [normalize_provider_result(value, provider="FIXTURE") for value, _ in cases]
    assert [result.status for result in results] == [expected for _, expected in cases]
    assert results[1].value == 0 and results[6].value == 0
    assert results[6].status is ProviderStatus.PROVIDER_ERROR
    assert results[6].error_category.value == "TIMEOUT"
    assert all(results[index].value is None for index in (3, 10, 11, 12))

    for malformed in (
        {"status": "SUCCESS"},
        {"status": "SUCCESS", "value": None},
        {"status": "SUCCESS", "value": np.nan},
        {"status": "SUCCESS", "value": pd.NA},
    ):
        assert normalize_provider_result(malformed, provider="FIXTURE").status is ProviderStatus.INVALID
    zero = normalize_provider_result({"status": "SUCCESS", "value": 0}, provider="FIXTURE")
    assert zero.status is ProviderStatus.SUCCESS and zero.value == 0
    boolean = normalize_provider_result({"status": "SUCCESS", "value": False}, provider="FIXTURE")
    assert boolean.status is ProviderStatus.SUCCESS and boolean.value is False
    canonical = normalize_provider_result({"status": "SUCCESS", "value": 3}, provider="FIXTURE")
    assert normalize_provider_result(canonical) is canonical
    assert normalize_provider_result(canonical, provider="FIXTURE") is canonical
    conflict = normalize_provider_result(canonical, provider="OTHER")
    assert conflict.status is ProviderStatus.INVALID and conflict.provider == "FIXTURE"
    provider_error = normalize_provider_result(
        {"status": "ERROR", "value": 0, "error": "timeout"}, provider="FIXTURE"
    )
    assert normalize_provider_result(provider_error) is provider_error
    for provider_status in ProviderStatus:
        canonical_status = ProviderResult(status=provider_status, provider="FIXTURE", value=0)
        assert normalize_provider_result(canonical_status) is canonical_status


def test_calendar_and_timezone_contract():
    assert is_idx_session("2026-08-07") and not is_idx_session("2026-08-08")
    assert calendar_state("2026-08-17") is CalendarState.CLOSED
    assert calendar_state("2025-08-18") is CalendarState.CLOSED
    assert n_idx_sessions_ago("2026-08-10", 1).date().isoformat() == "2026-08-07"
    assert trading_session_age("2026-08-07", "2026-08-10") == 1
    assert trading_session_age("2026-03-17", "2026-03-25") == 1
    assert trading_session_age("2026-08-08", "2026-08-10") is None
    assert calendar_state("2027-01-04") is CalendarState.UNKNOWN
    assert calendar_state("2027-01-02") is CalendarState.UNKNOWN
    assert calendar_state("2099-01-01") is CalendarState.UNKNOWN
    assert trading_session_age("2025-12-30", "2026-01-02") == 1
    assert trading_session_age("2026-12-30", "2027-01-01") is None
    with pytest.raises(CalendarCoverageError):
        latest_expected_completed_session("2027-01-01")
    with pytest.raises(CalendarCoverageError):
        previous_idx_session("2027-01-02")
    with pytest.raises(CalendarCoverageError):
        n_idx_sessions_ago("2027-01-01", 1)
    assert latest_expected_completed_session("2026-08-10T08:30:00Z").date().isoformat() == "2026-08-07"
    assert latest_expected_completed_session("2026-08-10T09:30:00Z").date().isoformat() == "2026-08-10"


def test_provenance_and_provider_audit_adapter_preserve_boundaries():
    assert normalize_provenance("IDX_OFFICIAL_XBRL") is EvidenceProvenance.DIRECT_OR_OFFICIAL
    assert normalize_provenance("VERIFIED_VENDOR_API") is EvidenceProvenance.VERIFIED
    assert normalize_provenance("GOOGLE_NEWS_PUBLIC_RESEARCH") is EvidenceProvenance.PUBLIC_RESEARCH
    assert normalize_provenance("MODEL_INFERRED") is EvidenceProvenance.INFERRED
    assert normalize_provenance("OHLCV_PROXY") is EvidenceProvenance.PROXY
    assert normalize_provenance("YFINANCE_PROXY_NOT_OFFICIAL_FILING") is EvidenceProvenance.PROXY
    assert normalize_provenance("unknown legacy authority") is EvidenceProvenance.MISSING
    assert aggregate_provenance("DIRECT_OR_OFFICIAL", "PROXY") is EvidenceProvenance.PROXY
    assert aggregate_provenance("VERIFIED", None) is EvidenceProvenance.MISSING
    audit = canonicalize_provider_audit(pd.DataFrame([
        {"provider": "ZERO_PROVIDER", "status": "OK", "bars": 0},
        {"provider": "FAIL_PROVIDER", "status": "ERROR", "error": "timeout"},
    ]))
    assert audit["provider"].tolist() == ["ZERO_PROVIDER", "FAIL_PROVIDER"]
    assert audit["provider_result_status"].tolist() == ["SUCCESS", "PROVIDER_ERROR"]


def test_emir_benchmark_lag_uses_idx_sessions_not_calendar_days():
    benchmark = pd.DataFrame({"Close": [7000.0]}, index=pd.DatetimeIndex(["2026-08-07"]))
    references = {str(i): pd.DataFrame(index=pd.DatetimeIndex(["2026-08-10"])) for i in range(20)}
    result = assess_benchmark_freshness(benchmark, references, min_universe_count=20)
    assert result["benchmark_business_lag_days"] == 1
    assert result["benchmark_freshness_state"] == "STALE_RELATIVE_TO_UNIVERSE"


def test_emir_future_ohlcv_is_invalid_not_current(monkeypatch):
    frame = pd.DataFrame(
        {"Close": [1.0] * 220},
        index=pd.DatetimeIndex(["2026-08-10"] * 220),
    )

    def fixture(*_args, **_kwargs):
        return FetchResult("AAA.JK", frame, "FIXTURE", "OK")

    monkeypatch.setattr(data_providers, "fetch_ohlcv", fixture)
    _, audit = fetch_many_ohlcv(
        ["AAA.JK"], max_workers=1, now="2026-08-07", serial_recovery_limit=0,
    )
    row = audit.iloc[0]
    assert row["data_age_sessions"] < 0
    assert row["completed_session_state"] == "INVALID_FUTURE_OBSERVATION"
    assert row["quality_state"] == "INVALID_FUTURE_OBSERVATION"
    assert row["provider_result_status"] == ProviderStatus.INVALID.value
