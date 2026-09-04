from __future__ import annotations

import pandas as pd

import top3_dashboard
import top3_html_render_patch as patch


def _indented_html() -> str:
    return """
    <style>
    .es-wrap{display:block}
    </style>
    <div class="es-wrap">
      <section class="es-card rank1"><h2>TEST</h2></section>
    </div>
    """


def test_normalize_top3_html_removes_markdown_code_block_indentation() -> None:
    html = patch.normalize_top3_html(_indented_html())

    assert html.startswith("<style>")
    assert '<div class="es-wrap">' in html
    assert '<section class="es-card rank1">' in html
    assert not any(line.startswith("    <") for line in html.splitlines())


def test_install_normalizes_final_dashboard_renderer(monkeypatch) -> None:
    monkeypatch.setattr(
        top3_dashboard,
        "render_top3_dashboard_html",
        lambda *args, **kwargs: _indented_html(),
    )

    state = patch.install()
    html = top3_dashboard.render_top3_dashboard_html(pd.DataFrame())

    assert state["state"] == "INSTALLED"
    assert html.startswith("<style>")
    assert not any(line.startswith("    <") for line in html.splitlines())
    assert getattr(top3_dashboard.render_top3_dashboard_html, "_emir_top3_html_render_patch", False) is True


def test_runtime_release_installs_html_normalizer_last() -> None:
    source = open("runtime_release.py", encoding="utf-8").read()
    html_patch = source.index('_try_optional_patch("top3_html_render_patch", "install")')
    scan_binding = source.index('_try_optional_patch("shared_fundamental_scan_binding_patch", "install")')
    assert html_patch > scan_binding
