"""Network-independent block diagrams for the control-system interface."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class DiagramSpec:
    body: str
    height: int


BLOCK_DIAGRAMS: dict[str, DiagramSpec] = {
    "method_map": DiagramSpec(
        body=r"""
        pid [label="PID\nerror feedback"]
        state [label="LQR / LQI\nstate feedback"]
        ilqr [label="iLQR\ntrajectory planning"]
        mpc [label="MPC\nconstrained re-planning"]
        gp [label="GP-MPC\nlearned residual"]
        safe [label="MPSC\nsafety certification"]
        pid -> state -> ilqr -> mpc -> gp -> safe
        """,
        height=410,
    ),
    "pid": DiagramSpec(
        body=r"""
        reference [label="Position reference", shape=ellipse]
        error [label="Σ", shape=circle, fixedsize=true, width=0.45]
        controller [label="PID\nKp · Ki · Kd"]
        plant [label="Cart-pole plant"]
        reference -> error [label="+"]
        error -> controller [label="e"]
        controller -> plant [label="u"]
        plant -> error [label="−x", style=dashed, constraint=false]
        """,
        height=290,
    ),
    "lqi": DiagramSpec(
        body=r"""
        reference [label="Position reference", shape=ellipse]
        error [label="Σ", shape=circle, fixedsize=true, width=0.45]
        integrator [label="Error integrator\n∫e dt"]
        feedback [label="Augmented gain\n−Kaug [x̂, xi]"]
        plant [label="Cart-pole plant"]
        reference -> error [label="+"]
        error -> integrator [label="e"]
        integrator -> feedback [label="xi"]
        feedback -> plant [label="u"]
        plant -> feedback [label="x̂", style=dashed, constraint=false]
        plant -> error [label="−x", style=dashed, constraint=false]
        """,
        height=350,
    ),
    "ilqr": DiagramSpec(
        body=r"""
        initial [label="Initial state\nhanging pole", shape=ellipse]
        forward [label="Forward rollout\nnonlinear dynamics"]
        backward [label="Backward pass\nQ expansion → k, K"]
        search [label="Line search\nstep size α"]
        solution [label="Trajectory policy\nx* · u* · Kt"]
        terminal [label="Terminal LQR\nupright hold"]
        initial -> forward -> backward -> search
        search -> forward [label="improve", style=dashed, constraint=false]
        search -> solution [label="converged"]
        solution -> terminal
        """,
        height=400,
    ),
    "mpc": DiagramSpec(
        body=r"""
        state [label="Measured state\nxk", shape=ellipse]
        optimizer [label="Finite-horizon optimizer\ndynamics + constraints"]
        action [label="Apply first action\nu0"]
        plant [label="Cart-pole plant"]
        state -> optimizer
        optimizer -> action [label="optimal plan"]
        action -> plant
        plant -> state [label="xk+1 · re-plan", style=dashed, constraint=false]
        """,
        height=310,
    ),
    "gp_mpc": DiagramSpec(
        body=r"""
        data [label="Rollout data", shape=ellipse]
        gp [label="GP residual model\nmean μ · uncertainty σ"]
        prior [label="Physics prior f̄\nnominal dynamics"]
        corrected [label="Corrected prediction\nf̄ + δf"]
        optimizer [label="MPC optimizer\nconstraint tightening κσ"]
        plant [label="Cart-pole plant"]
        data -> gp
        prior -> corrected
        gp -> corrected [label="μ, σ"]
        corrected -> optimizer
        optimizer -> plant [label="u0"]
        plant -> data [label="x, u, x+", style=dashed, constraint=false]
        """,
        height=390,
    ),
    "reckless": DiagramSpec(
        body=r"""
        policy [label="Balancing policy\ntarget beyond rail"]
        plant [label="Cart-pole plant"]
        boundary [label="Rail-boundary risk", shape=diamond, color="#fb7185"]
        policy -> plant [label="u"]
        plant -> policy [label="x", style=dashed, constraint=false]
        plant -> boundary [label="position"]
        """,
        height=250,
    ),
    "state_feedback": DiagramSpec(
        body=r"""
        reference [label="Position reference", shape=ellipse]
        prefilter [label="Reference prefilter\nNr"]
        sum [label="Σ", shape=circle, fixedsize=true, width=0.45]
        plant [label="Cart-pole plant"]
        observer [label="State observer\nx̂"]
        gain [label="Feedback gain\n−K"]
        reference -> prefilter -> sum [label="+"]
        sum -> plant [label="u"]
        plant -> observer [label="y"]
        observer -> gain
        gain -> sum [label="−Kx̂", style=dashed, constraint=false]
        """,
        height=350,
    ),
    "feedback_linearisation": DiagramSpec(
        body=r"""
        reference [label="Position reference", shape=ellipse]
        linear [label="Linear controller"]
        sum [label="Σ", shape=circle, fixedsize=true, width=0.45]
        plant [label="Nonlinear cart-pole"]
        cancellation [label="Nonlinear cancellation\nΔu(θ, θ̇)"]
        reference -> linear [label="error"]
        linear -> sum [label="virtual input v"]
        sum -> plant [label="u = v + Δu"]
        plant -> cancellation [label="θ, θ̇", style=dashed]
        cancellation -> sum [label="Δu", style=dashed, constraint=false]
        """,
        height=330,
    ),
    "mpsc": DiagramSpec(
        body=r"""
        policy [label="Active policy\nPID · LQR · MPC · RL"]
        filter [label="MPSC safety filter\nminimize ‖u − uprop‖²\nsafe recovery must exist"]
        plant [label="Cart-pole plant"]
        policy -> filter [label="proposed action"]
        filter -> plant [label="certified action"]
        plant -> policy [style=dashed, constraint=false]
        plant -> filter [style=dashed, constraint=false]
        """,
        height=310,
    ),
}


def render_block_diagram(name: str) -> None:
    """Render a responsive Graphviz diagram without external network requests."""

    try:
        specification = BLOCK_DIAGRAMS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown block diagram: {name}") from exc

    dot = rf"""
    digraph ControlSystem {{
        graph [
            rankdir=TB,
            bgcolor="transparent",
            margin=0,
            pad=0.08,
            nodesep=0.28,
            ranksep=0.42,
            splines=polyline
        ]
        node [
            shape=box,
            style="rounded,filled",
            fillcolor="#162033",
            color="#38bdf8",
            penwidth=1.2,
            fontcolor="#f8fafc",
            fontname="Arial",
            fontsize=10,
            margin="0.14,0.09"
        ]
        edge [
            color="#94a3b8",
            penwidth=1.1,
            arrowsize=0.72,
            fontcolor="#cbd5e1",
            fontname="Arial",
            fontsize=9
        ]
        {specification.body}
    }}
    """
    st.graphviz_chart(dot, width="stretch", height=specification.height)
