"""Application entry point and native multipage navigation."""

import streamlit as st

from ui_components import apply_global_styles


st.set_page_config(
    page_title="Control Systems Lab",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_global_styles()

navigation = st.navigation(
    {
        "LABORATORY": [
            st.Page(
                "pages/live_simulation.py",
                title="Live Simulation",
                icon="🎛️",
                url_path="live",
                default=True,
            )
        ],
        "EVALUATION": [
            st.Page(
                "pages/benchmark_dashboard.py",
                title="Controller Benchmarks",
                icon="📊",
                url_path="benchmarks",
            )
        ],
        "REFERENCE": [
            st.Page(
                "pages/theory_guide.py",
                title="Theory & Method Guide",
                icon="🧭",
                url_path="theory",
            )
        ],
    },
    position="sidebar",
    expanded=True,
)
navigation.run()
