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
