import json
import subprocess
import sys
import textwrap
import types

import runtime_release
from release_contract import SCANNER_RELEASE_VERSION


def test_stale_loaded_modules_are_reloaded_in_dependency_order(monkeypatch):
    engine = types.SimpleNamespace(ENGINE_VERSION="old")
    future = types.SimpleNamespace(SCANNER_VERSION="old")
    dashboard_legacy = types.SimpleNamespace()
    dashboard = types.SimpleNamespace(SCANNER_VERSION="old")
    pipeline = types.SimpleNamespace(PIPELINE_VERSION="old")
    monkeypatch.setitem(runtime_release.sys.modules, "narrative_flow_engine", engine)
    monkeypatch.setitem(runtime_release.sys.modules, "future_fundamental", future)
    monkeypatch.setitem(runtime_release.sys.modules, "top3_dashboard_legacy", dashboard_legacy)
    monkeypatch.setitem(runtime_release.sys.modules, "top3_dashboard", dashboard)
    monkeypatch.setitem(runtime_release.sys.modules, "resumable_scan", pipeline)
    calls: list[str] = []
    real_reload = runtime_release.importlib.reload

    def fake_reload(module):
        if getattr(module, "__name__", "") == "release_contract":
            return real_reload(module)
        name = {
            id(engine): "narrative_flow_engine",
            id(future): "future_fundamental",
            id(dashboard_legacy): "top3_dashboard_legacy",
            id(dashboard): "top3_dashboard",
            id(pipeline): "resumable_scan",
        }[id(module)]
        calls.append(name)
        return module

    monkeypatch.setattr(runtime_release.importlib, "reload", fake_reload)
    expected, reloaded = runtime_release.refresh_release_runtime(
        reload_order=(
            "narrative_flow_engine",
            "future_fundamental",
            "top3_dashboard_legacy",
            "top3_dashboard",
            "resumable_scan",
        ),
        version_markers={
            "narrative_flow_engine": "ENGINE_VERSION",
            "future_fundamental": "SCANNER_VERSION",
            "top3_dashboard": "SCANNER_VERSION",
            "resumable_scan": "PIPELINE_VERSION",
        },
    )

    assert expected == SCANNER_RELEASE_VERSION
    assert calls == [
        "narrative_flow_engine",
        "future_fundamental",
        "top3_dashboard_legacy",
        "top3_dashboard",
        "resumable_scan",
    ]
    assert reloaded == tuple(calls)

def test_optional_patch_failure_is_observable(monkeypatch):
    runtime_release._LAST_PATCH_STATUS.clear()

    def broken_import(name):
        raise RuntimeError("provider patch unavailable")

    monkeypatch.setattr(runtime_release.importlib, "import_module", broken_import)
    with __import__("pytest").warns(RuntimeWarning, match="Optional runtime patch"):
        runtime_release._try_optional_patch("optional_patch", "install")

    status = runtime_release.runtime_patch_status()
    assert status["optional_patch.install"]["state"] == "FAILED"
    assert "RuntimeError" in status["optional_patch.install"]["detail"]


def test_release_installation_renders_complete_raw_research_top3_idempotently():
    script = textwrap.dedent(
        """
        import json

        import pandas as pd

        import runtime_release
        from release_contract import SCANNER_RELEASE_VERSION

        rows = []
        states = (
            "FORWARD_CHECK_COMPLETED_NO_MATERIAL_EVENT",
            "MATERIAL_FORWARD_RESEARCH_EVIDENCE_FOUND",
            "FUTURE_FUNDAMENTAL_EVIDENCE_PENDING",
        )
        for rank, (ticker, score) in enumerate(
            (
                ("BBRI.JK", 62.0),
                ("ADRO.JK", 61.2),
                ("BBCA.JK", 59.7),
                ("ANTM.JK", 57.7),
                ("TLKM.JK", 57.6),
            ),
            1,
        ):
            rows.append({
                "ticker": ticker,
                "company_name": ticker,
                "sector": "TEST",
                "raw_research_rank": rank,
                "raw_research_score": score,
                "dashboard_rank": rank,
                "emir_final_score": score,
                "last_price": 1000,
                "broker_inventory_evidence_type": "OHLCV_PROXY",
                "future_fundamental_score": float("nan"),
                "future_fundamental_coverage_pct": 0.0,
                "forward_collection_state": states[min(rank - 1, 2)],
                "future_fundamental_state": states[min(rank - 1, 2)],
                "estimated_smart_money_cost_low": 900,
                "estimated_smart_money_cost_high": 950,
                "estimated_smart_money_cost": 925,
                "smart_money_cost_state": "PROXY",
                "smart_money_cost_evidence_type": "OHLCV_PROXY",
                "smart_money_cost_confidence_pct": 50,
                "real_money_candidate": False,
                "real_money_entry_candidate": False,
                "real_money_ready": False,
                "real_money_gate_class": "HARD_BLOCK" if rank == 1 else "WAIT_TIMING",
                "real_money_authorization_tier": "HARD_BLOCKED" if rank == 1 else "WAIT_TIMING",
                "real_money_hard_block_count": 1 if rank == 1 else 0,
            })

        source = pd.DataFrame(rows)
        frozen = source.copy(deep=True)
        runtime_release._install_integrity_patch(SCANNER_RELEASE_VERSION)
        runtime_release._install_integrity_patch(SCANNER_RELEASE_VERSION)

        import top3_dashboard

        selected = top3_dashboard.select_top3(source, 3)
        research_selected = top3_dashboard.select_research_top3(source, 3)
        html = top3_dashboard.render_top3_dashboard_html(selected)
        pd.testing.assert_frame_equal(source, frozen)

        print(json.dumps({
            "tickers": selected["ticker"].tolist(),
            "research_tickers": research_selected["ticker"].tolist(),
            "banner_count": html.count("RAW_RESEARCH_TOP3"),
            "lane_banner_count": html.count('class="es-lane-banner"'),
            "card_count": html.count('class="es-card '),
            "cost_count": html.count('class="es-cost-basis"'),
            "future_states": [
                html.count(">CHECKED</b>"),
                html.count(">RESEARCH</b>"),
                html.count(">PENDING</b>"),
            ],
            "positions": [
                html.index("<style>"),
                html.index('<div class="es-wrap">'),
                html.index('class="es-lane-banner"'),
                html.index("TOP 3 <b>EMIR-STYLE SCANNER"),
                html.index("BBRI"),
                html.index("ADRO"),
                html.index("BBCA"),
            ],
            "dashboard_title": "TOP 3 <b>EMIR-STYLE SCANNER" in html,
            "risk_label": "Risk flags:" in html,
            "authorization_label": "Decision support — bukan eksekusi otomatis" in html,
            "execution_count": len(top3_dashboard.select_real_money_top3(source, 3)),
        }))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout.strip())

    assert result["tickers"] == ["BBRI.JK", "ADRO.JK", "BBCA.JK"]
    assert result["research_tickers"] == ["BBRI.JK", "ADRO.JK", "BBCA.JK"]
    assert result["banner_count"] == result["lane_banner_count"] == 1
    assert result["card_count"] == result["cost_count"] == 3
    assert result["future_states"] == [1, 1, 1]
    assert result["positions"] == sorted(result["positions"])
    assert result["dashboard_title"] is True
    assert result["risk_label"] is True
    assert result["authorization_label"] is True
    assert result["execution_count"] == 0
