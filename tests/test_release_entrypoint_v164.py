from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_entrypoint_exists_at_release_root():
    app = ROOT / "app.py"
    assert app.is_file()
    text = app.read_text(encoding="utf-8")
    assert "st.title(\"IDX Emir Autonomous Scanner\")" in text
    assert len(text) > 5_000


def test_deployment_guide_requires_atomic_replacement():
    guide = (ROOT / "DEPLOYMENT_SAFE_UPDATE_V1_7_0.md").read_text(encoding="utf-8")
    assert "Jangan hapus `app.py`" in guide
    assert "satu commit" in guide.lower()
