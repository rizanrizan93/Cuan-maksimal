import types

import runtime_release


def test_stale_loaded_modules_are_reloaded_in_dependency_order(monkeypatch):
    engine = types.SimpleNamespace(ENGINE_VERSION="old")
    pipeline = types.SimpleNamespace(PIPELINE_VERSION="old")
    monkeypatch.setitem(runtime_release.sys.modules, "narrative_flow_engine", engine)
    monkeypatch.setitem(runtime_release.sys.modules, "resumable_scan", pipeline)
    calls: list[str] = []
    real_reload = runtime_release.importlib.reload

    def fake_reload(module):
        if getattr(module, "__name__", "") == "release_contract":
            return real_reload(module)
        name = "narrative_flow_engine" if module is engine else "resumable_scan"
        calls.append(name)
        return module

    monkeypatch.setattr(runtime_release.importlib, "reload", fake_reload)
    expected, reloaded = runtime_release.refresh_release_runtime(
        reload_order=("narrative_flow_engine", "resumable_scan"),
        version_markers={
            "narrative_flow_engine": "ENGINE_VERSION",
            "resumable_scan": "PIPELINE_VERSION",
        },
    )

    assert expected.startswith("1.9.17-")
    assert calls == ["narrative_flow_engine", "resumable_scan"]
    assert reloaded == tuple(calls)
