from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


class ExpectedStop(RuntimeError):
    pass


class Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class StreamlitStub(ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = {}
        self.secrets = {}
        self.sidebar = Context()

    def cache_data(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def expander(self, *args, **kwargs):
        return Context()

    def set_page_config(self, *args, **kwargs): pass
    def title(self, *args, **kwargs): pass
    def caption(self, *args, **kwargs): pass
    def markdown(self, *args, **kwargs): pass
    def dataframe(self, *args, **kwargs): pass
    def header(self, *args, **kwargs): pass
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def success(self, *args, **kwargs): pass
    def write(self, *args, **kwargs): pass
    def button(self, *args, **kwargs): return False
    def file_uploader(self, *args, **kwargs): return None
    def selectbox(self, label, options, **kwargs): return list(options)[0]
    def checkbox(self, *args, **kwargs): return kwargs.get("value", False)
    def slider(self, *args, **kwargs):
        if "value" in kwargs: return kwargs["value"]
        return args[3] if len(args) > 3 else args[1]
    def number_input(self, *args, **kwargs): return kwargs.get("value", 0)
    def stop(self): raise ExpectedStop("expected no-upload stop")


def test_app_reaches_expected_no_upload_stop(monkeypatch):
    stub = StreamlitStub()
    monkeypatch.setitem(sys.modules, "streamlit", stub)
    monkeypatch.syspath_prepend(str(ROOT))
    with pytest.raises(ExpectedStop, match="no-upload"):
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")


def test_real_streamlit_app_starts_without_secrets_file():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "app.py"))
    app.run(timeout=30)
    assert not app.exception
    assert [item.value for item in app.title] == ["IDX Emir Autonomous Scanner"]
