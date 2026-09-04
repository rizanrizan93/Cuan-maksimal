from __future__ import annotations

import tomllib
from pathlib import Path

from top3_dashboard import TOP3_UI_VERSION, _INSTITUTIONAL_CSS


def test_institutional_dashboard_skin_is_presentation_only_contract() -> None:
    assert TOP3_UI_VERSION == "1.0.0-institutional-ui"
    assert ".es-card:before" in _INSTITUTIONAL_CSS
    assert ".es-score" in _INSTITUTIONAL_CSS
    assert ".es-rec" in _INSTITUTIONAL_CSS
    assert "@media(max-width:640px)" in _INSTITUTIONAL_CSS
    assert "#071019" in _INSTITUTIONAL_CSS


def test_streamlit_theme_matches_dashboard_shell() -> None:
    config = tomllib.loads(Path(".streamlit/config.toml").read_text(encoding="utf-8"))
    theme = config["theme"]
    sidebar = theme["sidebar"]
    assert theme["base"] == "dark"
    assert theme["backgroundColor"] == "#071019"
    assert theme["primaryColor"] == "#0F9D8A"
    assert theme["showWidgetBorder"] is True
    assert sidebar["backgroundColor"] == "#08131E"
