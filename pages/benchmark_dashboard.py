"""Interactive dashboard for the reproducible benchmark protocol."""

from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import streamlit as st

from benchmark import (
    CONTROLLERS,
    DEFAULT_CONTROLLERS,
    DEFAULT_SEEDS,
    SCENARIOS,
    SUITES,
    benchmark_report_csv,
    run_benchmark,
)
from ui_components import render_page_header, render_section_title


CONTROLLER_COLORS = {
    "pid": "#a78bfa",
    "pole": "#f59e0b",
    "lqr": "#38bdf8",
    "lqi": "#22c55e",
    "mpc": "#f43f5e",
}


def _parse_seeds(value: str) -> tuple[int, ...]:
    tokens = [token for token in re.split(r"[\s,;]+", value.strip()) if token]
    if not tokens:
        raise ValueError("Enter at least one integer seed.")
    try:
        seeds = tuple(sorted({int(token) for token in tokens}))
    except ValueError as exc:
        raise ValueError("Seeds must be integers separated by spaces or commas.") from exc
    if len(seeds) > 20:
        raise ValueError("Use at most 20 seeds in an interactive run.")
    if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        raise ValueError("Every seed must be in the range [0, 2³² − 1].")
    return seeds


@st.cache_data(show_spinner=False)
def _cached_benchmark(
    suite: str,
    controller_ids: tuple[str, ...],
    seeds: tuple[int, ...],
):
    return run_benchmark(
        suite=suite,
        controller_ids=controller_ids,
        seeds=seeds,
    )


def _chart_layout(figure, *, height: int = 430):
    figure.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,.45)",
        font=dict(color="#cbd5e1"),
        legend_title_text="Controller",
    )
    figure.update_xaxes(gridcolor="#25324a")
    figure.update_yaxes(gridcolor="#25324a")
    return figure


render_page_header(
    kicker="Versioned evaluation protocol",
    title="Controller Benchmarks",
    subtitle=(
        "Run repeatable controller comparisons through the exact same closed-loop "
        "runtime as the live simulator. Results measure control quality—not machine speed."
    ),
)

st.sidebar.markdown("### 📊 Benchmark Controls")
st.sidebar.caption(
    "Every scenario/controller/seed combination receives a fresh plant and controller."
)
suite = st.sidebar.radio(
    "Protocol suite",
    options=tuple(SUITES),
    index=1,
    format_func=lambda value: value.title(),
    horizontal=True,
)
controller_ids = tuple(
    st.sidebar.multiselect(
        "Controllers",
        options=tuple(CONTROLLERS),
        default=list(DEFAULT_CONTROLLERS),
        format_func=lambda value: CONTROLLERS[value].label,
    )
)
seed_text = st.sidebar.text_input(
    "Random seeds",
    value=", ".join(str(seed) for seed in DEFAULT_SEEDS),
    help="Seeds are deduplicated and sorted before the protocol is hashed.",
)

seed_error = None
try:
    seeds = _parse_seeds(seed_text)
except ValueError as exc:
    seeds = ()
    seed_error = str(exc)

if seed_error:
    st.sidebar.error(seed_error)
if not controller_ids:
    st.sidebar.warning("Select at least one controller.")
if "mpc" in controller_ids:
    st.sidebar.warning(
        "Nonlinear MPC solves an optimization problem at every control step. "
        "Use the Quick suite first."
    )

scenario_count = len(SUITES[suite])
run_count = scenario_count * len(controller_ids) * len(seeds)
st.sidebar.caption(
    f"Planned workload: {scenario_count} scenarios × "
    f"{len(controller_ids)} controllers × {len(seeds)} seeds = **{run_count} runs**"
)
run_requested = st.sidebar.button(
    "Run benchmark",
    type="primary",
    width="stretch",
    disabled=bool(seed_error or not controller_ids),
)

selection_signature = (suite, controller_ids, seeds)
if run_requested:
    with st.spinner(f"Running {run_count} isolated closed-loop simulations…"):
        try:
            report = _cached_benchmark(suite, controller_ids, seeds)
        except Exception as exc:
            st.error("The benchmark could not be completed.")
            st.exception(exc)
        else:
            st.session_state["benchmark_report"] = report
            st.session_state["benchmark_signature"] = selection_signature

report = st.session_state.get("benchmark_report")
report_signature = st.session_state.get("benchmark_signature")

with st.expander("Protocol catalog", expanded=report is None):
    catalog_rows = []
    for scenario_id in SUITES[suite]:
        scenario = SCENARIOS[scenario_id]
        catalog_rows.append(
            {
                "Scenario": scenario_id,
                "Purpose": scenario.description,
                "Duration (s)": scenario.total_time,
                "Noise σ (deg)": scenario.sensor_noise_std_deg,
                "Disturbance": scenario.disturbance_profile.value,
            }
        )
    st.dataframe(catalog_rows, hide_index=True, width="stretch")

if report is None:
    st.info(
        "Choose a protocol in the sidebar and run it. The Standard suite uses "
        "36 fast linear-controller runs with the default configuration."
    )
    st.stop()

if report_signature != selection_signature:
    st.warning(
        "The controls have changed since this report was generated. Select "
        "Run benchmark to refresh the results."
    )

successful_runs = sum(run.success for run in report.runs)
terminated_runs = sum(run.terminated for run in report.runs)
metric_columns = st.columns(4)
metric_columns[0].metric("Completed runs", len(report.runs))
metric_columns[1].metric(
    "Successful",
    f"{successful_runs}/{len(report.runs)}",
    f"{100 * successful_runs / len(report.runs):.0f}%",
)
metric_columns[2].metric(
    "Track-limit terminations",
    terminated_runs,
    f"{100 * terminated_runs / len(report.runs):.0f}%",
    delta_color="inverse",
)
metric_columns[3].metric("Protocol", report.protocol_sha256[:12])

aggregate_frame = pd.DataFrame(
    [aggregate.to_dict() for aggregate in report.aggregates]
)
aggregate_frame["Controller"] = aggregate_frame["controller_id"].map(
    lambda value: CONTROLLERS[value].label
)
aggregate_frame["Scenario"] = aggregate_frame["scenario_id"].str.replace(
    "_", " "
).str.title()

render_section_title("Performance overview")
chart_left, chart_right = st.columns(2)

success_pivot = aggregate_frame.pivot(
    index="Scenario",
    columns="Controller",
    values="success_rate",
) * 100.0
success_figure = px.imshow(
    success_pivot,
    color_continuous_scale=["#7f1d1d", "#f59e0b", "#15803d"],
    range_color=(0, 100),
    text_auto=".0f",
    aspect="auto",
    labels={"color": "Success %"},
    title="Success rate by scenario",
)
success_figure.update_traces(texttemplate="%{z:.0f}%")
success_figure.update_coloraxes(colorbar_ticksuffix="%")
chart_left.plotly_chart(
    _chart_layout(success_figure),
    width="stretch",
    config={"displaylogo": False},
)

rms_figure = px.bar(
    aggregate_frame,
    x="Scenario",
    y="rms_tracking_error_mean",
    color="controller_id",
    barmode="group",
    color_discrete_map=CONTROLLER_COLORS,
    labels={
        "rms_tracking_error_mean": "RMS tracking error (m)",
        "controller_id": "Controller",
    },
    title="Tracking accuracy",
    hover_data={"Controller": True, "controller_id": False},
)
chart_right.plotly_chart(
    _chart_layout(rms_figure),
    width="stretch",
    config={"displaylogo": False},
)

tradeoff_figure = px.scatter(
    aggregate_frame,
    x="rms_tracking_error_mean",
    y="total_energy_mean",
    color="controller_id",
    symbol="scenario_id",
    color_discrete_map=CONTROLLER_COLORS,
    labels={
        "rms_tracking_error_mean": "RMS tracking error (m)",
        "total_energy_mean": "Control energy (∫u²dt)",
        "controller_id": "Controller",
        "scenario_id": "Scenario",
    },
    title="Accuracy–effort trade-off",
    hover_data={
        "Controller": True,
        "Scenario": True,
        "steady_state_error_mean": ":.4f",
        "max_angle_deg_mean": ":.2f",
    },
)
tradeoff_figure.update_traces(marker=dict(size=12, line=dict(width=1, color="#e2e8f0")))
st.plotly_chart(
    _chart_layout(tradeoff_figure, height=480),
    width="stretch",
    config={"displaylogo": False},
)

render_section_title("Aggregate results")
display_columns = {
    "Scenario": "Scenario",
    "Controller": "Controller",
    "success_rate": "Success",
    "termination_rate": "Terminated",
    "rms_tracking_error_mean": "RMS error (m)",
    "steady_state_error_mean": "SS error (m)",
    "total_energy_mean": "Energy",
    "peak_force_mean": "Peak force (N)",
    "max_angle_deg_mean": "Max |θ| (deg)",
}
st.dataframe(
    aggregate_frame[list(display_columns)].rename(columns=display_columns),
    hide_index=True,
    width="stretch",
    column_config={
        "Success": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "Terminated": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        "RMS error (m)": st.column_config.NumberColumn(format="%.4f"),
        "SS error (m)": st.column_config.NumberColumn(format="%.4f"),
        "Energy": st.column_config.NumberColumn(format="%.2f"),
        "Peak force (N)": st.column_config.NumberColumn(format="%.2f"),
        "Max |θ| (deg)": st.column_config.NumberColumn(format="%.2f"),
    },
)

with st.expander("Raw runs and reproducibility metadata"):
    raw_frame = pd.DataFrame([run.to_dict() for run in report.runs])
    st.dataframe(raw_frame, hide_index=True, width="stretch")
    st.markdown("**Artifact fingerprints**")
    st.markdown(
        f"Implementation: <span class='hash-value'>{report.implementation_sha256}</span><br>"
        f"Protocol: <span class='hash-value'>{report.protocol_sha256}</span><br>"
        f"Results: <span class='hash-value'>{report.results_sha256}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Reports intentionally omit timestamps and wall-clock duration. The same "
        "source, protocol and dependency environment produces stable artifacts."
    )

render_section_title("Export report")
download_json, download_csv = st.columns(2)
download_json.download_button(
    "Download complete JSON",
    data=f"{report.to_json()}\n",
    file_name=f"benchmark_{report.suite}_{report.protocol_sha256[:8]}.json",
    mime="application/json",
    width="stretch",
)
download_csv.download_button(
    "Download raw runs CSV",
    data=benchmark_report_csv(report),
    file_name=f"benchmark_{report.suite}_{report.protocol_sha256[:8]}.csv",
    mime="text/csv",
    width="stretch",
)
