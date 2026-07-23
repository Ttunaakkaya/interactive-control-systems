"""Streamlit presentation-layer smoke test."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_default_app_renders_without_exceptions():
    app_path = Path(__file__).with_name("app.py")
    app = AppTest.from_file(str(app_path), default_timeout=120).run()

    assert not app.exception
    assert len(app.metric) == 7
    assert len(app.get("plotly_chart")) >= 7
    assert app.selectbox[0].value == "PID (Classical)"

    animation_spec = json.loads(app.get("plotly_chart")[0].proto.spec)
    assert len(animation_spec["frames"]) == 250
    assert animation_spec["frames"][0]["name"] == "0"
    assert animation_spec["frames"][-1]["name"] == "499"
    play_frame = animation_spec["layout"]["updatemenus"][0]["buttons"][0][
        "args"
    ][1]["frame"]
    assert play_frame["duration"] == 40
