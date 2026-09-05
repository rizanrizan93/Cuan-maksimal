from __future__ import annotations

"""Presentation-only fix for Emir Execution Research Top 3 HTML rendering.

Streamlit parses ``st.markdown`` as Markdown before allowing embedded HTML.
The legacy Top 3 renderer and its runtime wrappers can produce mixed indentation:
a stylesheet may begin at column zero while later HTML lines retain four or more
leading spaces. In that shape ``textwrap.dedent`` alone has no common indentation
to remove, and Markdown renders the indented tags as a visible code block.

This patch normalizes every non-empty output line to the left margin. It does not
touch candidate selection, scoring, ranking, gates, or persisted scan data.
"""

from functools import wraps
from textwrap import dedent
from typing import Any


PATCH_VERSION = "1.0.1-streamlit-markdown-html-left-margin"


def normalize_top3_html(value: Any) -> str:
    """Return HTML with no Markdown code-block indentation on any line."""
    text = dedent(str(value or "")).strip()
    if not text:
        return ""
    return "\n".join(line.lstrip() for line in text.splitlines()).strip()


def install() -> dict[str, str]:
    import top3_dashboard

    current = top3_dashboard.render_top3_dashboard_html
    if str(getattr(current, "_emir_top3_html_render_patch_version", "")) == PATCH_VERSION:
        return {"patch_version": PATCH_VERSION, "state": "ALREADY_INSTALLED"}

    @wraps(current)
    def wrapped(*args: Any, **kwargs: Any) -> str:
        return normalize_top3_html(current(*args, **kwargs))

    setattr(wrapped, "_emir_top3_html_render_patch", True)
    setattr(wrapped, "_emir_top3_html_render_patch_version", PATCH_VERSION)
    top3_dashboard.render_top3_dashboard_html = wrapped
    return {"patch_version": PATCH_VERSION, "state": "INSTALLED"}


__all__ = ["PATCH_VERSION", "install", "normalize_top3_html"]
