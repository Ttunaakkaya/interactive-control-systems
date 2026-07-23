"""Streamlit presentation-layer smoke test."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_default_app_renders_without_exceptions():
    app_path = Path(__file__).with_name("app.py")
    app = AppTest.from_file(str(app_path), default_timeout=120).run()

    assert not app.exception
    assert len(app.metric) == 7
    assert len(app.get("plotly_chart")) >= 7
    assert app.selectbox[0].value == "PID (Classical)"
