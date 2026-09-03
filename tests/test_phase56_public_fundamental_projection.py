from __future__ import annotations

from phase56_public_fundamental_projection import _rows_to_bundle, fetch_public_bundle


class _Response:
    status_code = 200

    def json(self):
        return [{
            "ticker": "BBCA",
            "proxy_period_end": "2026-06-30",
            "proxy_observed_at": "2026-08-14T04:34:31+00:00",
            "official_period_end": "2026-06-30",
            "official_observed_at": "2026-08-14T04:34:31+00:00",
            "proxy_metrics": {"roe_pct": 22.0, "net_margin_pct": 41.0},
            "official_metrics": {"revenue": 100.0, "net_income": 40.0},
            "source_families": ["IDX_OFFICIAL_XBRL", "YAHOO"],
            "official_coverage_pct": 80.0,
        }]


class _Session:
    def get(self, *args, **kwargs):
        return _Response()


def test_rows_to_bundle_preserves_fact_separation():
    bundle = _rows_to_bundle(_Response().json())
    assert bundle["BBCA"]["proxy_metrics"]["roe_pct"] == 22.0
    assert bundle["BBCA"]["official_metrics"]["revenue"] == 100.0
    assert bundle["BBCA"]["official_period_end"] == "2026-06-30"
    assert "IDX_OFFICIAL_XBRL" in bundle["BBCA"]["source_families"]


def test_fetch_public_bundle_uses_read_only_projection_shape():
    bundle, meta = fetch_public_bundle(["BBCA.JK"], session=_Session())
    assert meta["state"] == "PUBLIC_PROJECTION_LOADED"
    assert meta["tickers"] == 1
    assert bundle["BBCA"]["proxy_period_end"] == "2026-06-30"
    assert bundle["BBCA"]["official_coverage_pct"] == 80.0
