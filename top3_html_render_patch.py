from __future__ import annotations

"""Presentation-only fix for Emir Execution Research Top 3 HTML rendering.

Streamlit parses ``st.markdown`` as Markdown before allowing embedded HTML.
The legacy Top 3 renderer intentionally returns indented triple-quoted HTML;
lines with four leading spaces are therefore interpreted as Markdown code
blocks and the tags become visible text.  This patch removes only that common
indentation.  It does not touch candidate selection, scoring, ranking, gates,
or persisted scan data.
"""

from functools import wraps
from textwrap import dedent
from typing import Any


PATCH_VERSION = "1.0.0-streamlit-markdown-html-dedent"


def normalize_top3_html(value: Any) -> str:
    """Return HTML at the left margin so Streamlit does not treat it as code."""
    return dedent(str(value or "")).strip()


def install() -> dict[str, str]:
    import top3_dashboard

    current = top3_dashboard.render_top3_dashboard_html
    if bool(getattr(current, "_emir_top3_html_render_patch", False)):
        return {"patch_version": PATCH_VERSION, "state": "ALREADY_INSTALLED"}

    @wraps(current)
    def wrapped(*args: Any, **kwargs: Any) -> str:
        return normalize_top3_html(current(*args, **kwargs))

    setattr(wrapped, "_emir_top3_html_render_patch", True)
    setattr(wrapped, "_emir_top3_html_render_patch_version", PATCH_VERSION)
    top3_dashboard.render_top3_dashboard_html = wrapped
    return {"patch_version": PATCH_VERSION, "state": "INSTALLED"}


__all__ = ["PATCH_VERSION", "install", "normalize_top3_html"]
