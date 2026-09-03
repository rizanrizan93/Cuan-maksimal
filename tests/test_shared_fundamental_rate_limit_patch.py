from __future__ import annotations

from types import SimpleNamespace

from shared_fundamental_rate_limit_patch import _rate_limit_detail, _reset_for_tests, install
from shared_fundamental_runtime import SharedFundamentalRuntime


class FakeResponse:
    def __init__(self, status_code: int, payload: dict, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self) -> dict:
        return dict(self._payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP_{self.status_code}")


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls = 0

    def get(self, *_args, **_kwargs) -> FakeResponse:
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("cross-provider fallback should stop after shared ZAPI 429")
        return self.response


def runtime(response: FakeResponse) -> SharedFundamentalRuntime:
    return SharedFundamentalRuntime("TEST", config=SimpleNamespace(ready=True), backend=object(), api_key="zpi_test", session=FakeSession(response))


def test_month_limit_stops_yahoo_fallback() -> None:
    _reset_for_tests(); install()
    item = runtime(FakeResponse(429, {"window":"month"}, {"X-RateLimit-Remaining-Month":"0", "X-RateLimit-Remaining-Minute":"100"}))
    rows, meta = item.refresh_structured("BBCA")
    assert rows == []
    assert meta["state"] == "ZAPI_RATE_LIMIT_MONTH"
    assert len(meta["attempts"]) == 1
    assert item.session.calls == 1


def test_minute_limit_stops_yahoo_fallback() -> None:
    _reset_for_tests(); install()
    item = runtime(FakeResponse(429, {"window":"minute"}, {"X-RateLimit-Remaining-Month":"321", "X-RateLimit-Remaining-Minute":"0"}))
    rows, meta = item.refresh_structured("AALI")
    assert rows == []
    assert meta["state"] == "ZAPI_RATE_LIMIT_MINUTE"
    assert item.session.calls == 1


def test_headers_can_classify_month_window() -> None:
    detail = _rate_limit_detail(FakeResponse(429, {}, {"X-RateLimit-Remaining-Month":"0", "X-RateLimit-Remaining-Minute":"50"}))
    assert detail["window"] == "month"
