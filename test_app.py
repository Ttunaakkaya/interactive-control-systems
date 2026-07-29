"""Streamlit multipage presentation and interaction tests."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_shell_renders_default_live_simulation_without_exceptions():
    app_path = Path(__file__).with_name("app.py")
    app = AppTest.from_file(str(app_path), default_timeout=120).run()

    assert not app.exception
    assert len(app.metric) == 7
    assert len(app.get("plotly_chart")) >= 7
    assert app.selectbox[0].value == "PID (Classical)"


def test_benchmark_dashboard_initial_state_is_lightweight():
    page_path = Path(__file__).with_name("pages") / "benchmark_dashboard.py"
    app = AppTest.from_file(str(page_path), default_timeout=120).run()

    assert not app.exception
    assert app.radio[0].value == "standard"
    assert app.multiselect[0].value == ["pole", "lqr", "lqi"]
    assert app.text_input[0].value == "0, 1, 2"
    assert len(app.metric) == 0
    assert len(app.get("plotly_chart")) == 0


def test_benchmark_dashboard_runs_quick_protocol_and_renders_exports():
    page_path = Path(__file__).with_name("pages") / "benchmark_dashboard.py"
    app = AppTest.from_file(str(page_path), default_timeout=120).run()
    app.radio[0].set_value("quick")
    app.multiselect[0].set_value(["lqr"])
    app.text_input[0].set_value("7")
    app.button[0].click()
    app.run()

    assert not app.exception
    assert app.metric[0].value == "2"
    assert app.metric[1].value == "2/2"
    assert len(app.get("plotly_chart")) == 3
    assert len(app.dataframe) == 3
    assert len(app.get("download_button")) == 2


def test_theory_guide_renders_all_reference_tabs():
    page_path = Path(__file__).with_name("pages") / "theory_guide.py"
    app = AppTest.from_file(str(page_path), default_timeout=120).run()

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Method map",
        "Model & estimation",
        "Optimal & predictive",
        "Learning & safety",
        "Decision guide",
    ]
    assert len(app.dataframe) == 2
