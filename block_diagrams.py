"""Network-independent block diagrams for the control-system interface."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class DiagramSpec:
    body: str
    height: int
    rankdir: str = "TB"


BLOCK_DIAGRAMS: dict[str, DiagramSpec] = {
    "method_map": DiagramSpec(
        body=r"""
        reference [label="r(t)", shape=plaintext]
        error [label="Σ", shape=circle, fixedsize=true, width=0.42,
               fillcolor="#fde047", fontcolor="#111827"]
        controller [label="Selected controller\nPID · LQR/LQI · iLQR · MPC · GP-MPC"]
        safety [label="MPSC\noptional safety filter"]
        plant [label="Plant / process", fillcolor="#e2e8f0", fontcolor="#111827"]
        output [label="y(t)", shape=plaintext]
        reference -> error [label="+"]
        error -> controller [label="error / state"]
        controller -> safety [label="proposed u"]
        safety -> plant [label="certified u"]
        plant -> output
        output -> error [label="−", constraint=false]
        """,
        height=385,
    ),
    "pid": DiagramSpec(
        body=r"""
        reference [label="r(t)", shape=plaintext]
        error [label="Σ", shape=circle, fixedsize=true, width=0.42,
               fillcolor="#fde047", fontcolor="#111827"]
        proportional [label="P\nKp e(t)", fillcolor="#bbf7d0", fontcolor="#111827"]
        integral [label="I\nKi ∫e(t)dτ", fillcolor="#99f6e4", fontcolor="#111827"]
        derivative [label="D\nKd de(t)/dt", fillcolor="#fdba74", fontcolor="#111827"]
        control [label="Σ", shape=circle, fixedsize=true, width=0.42,
                 fillcolor="#fde047", fontcolor="#111827"]
        plant [label="Plant / process", fillcolor="#e2e8f0", fontcolor="#111827"]
        output [label="y(t)", shape=plaintext]
        { rank=same; proportional; integral; derivative }
        reference -> error [label="+"]
        error -> proportional [label="e(t)"]
        error -> integral
        error -> derivative
        proportional -> control
        integral -> control
        derivative -> control
        control -> plant [label="u(t)"]
        plant -> output
        output -> error [label="−", constraint=false]
        """,
        height=410,
    ),
    "lqi": DiagramSpec(
        body=r"""
        reference [label="r(t)", shape=plaintext]
        error [label="Σ", shape=circle, fixedsize=true, width=0.42,
               fillcolor="#fde047", fontcolor="#111827"]
        integrator [label="Integrator\nξ = ∫e(t)dt"]
        integral_gain [label="Integral gain\n−Ki ξ"]
        state_gain [label="State feedback\n−K x̂"]
        control [label="Σ", shape=circle, fixedsize=true, width=0.42,
                 fillcolor="#fde047", fontcolor="#111827"]
        plant [label="Plant / process", fillcolor="#e2e8f0", fontcolor="#111827"]
        output [label="y(t)", shape=plaintext]
        reference -> error [label="+"]
        error -> integrator [label="e"]
        integrator -> integral_gain
        integral_gain -> control
        state_gain -> control
        control -> plant [label="u(t)"]
        plant -> output
        output -> error [label="−", constraint=false]
        plant -> state_gain [label="x̂", constraint=false]
        """,
        height=370,
    ),
    "ilqr": DiagramSpec(
        body=r"""
        command [label="Target + initial state", shape=plaintext]
        trajectory [label="Nominal iLQR trajectory\nx*(t), u*(t), Kt"]
        deviation [label="Σ", shape=circle, fixedsize=true, width=0.42,
                   fillcolor="#fde047", fontcolor="#111827"]
        gain [label="Time-varying feedback\nKt δx"]
        control [label="Σ", shape=circle, fixedsize=true, width=0.42,
                 fillcolor="#fde047", fontcolor="#111827"]
        plant [label="Nonlinear plant", fillcolor="#e2e8f0", fontcolor="#111827"]
        output [label="x(t)", shape=plaintext]
        command -> trajectory
        trajectory -> deviation [label="−x*(t)"]
        deviation -> gain [label="δx"]
        gain -> control
        trajectory -> control [label="u*(t)", constraint=false]
        control -> plant [label="u(t)"]
        plant -> output
        output -> deviation [label="+x(t)", constraint=false]
        """,
        height=405,
    ),
    "mpc": DiagramSpec(
        body=r"""
        reference [label="r(k)", shape=plaintext]
        optimizer [label="MPC controller\nfinite-horizon optimizer"]
        model [label="Prediction model\n+ constraints", fillcolor="#c4b5fd", fontcolor="#111827"]
        plant [label="Plant / process", fillcolor="#e2e8f0", fontcolor="#111827"]
        output [label="y(k)", shape=plaintext]
        measurement [label="State estimate\nx̂(k)"]
        reference -> optimizer
        model -> optimizer [constraint=false]
        optimizer -> plant [label="apply u₀"]
        plant -> output
        plant -> measurement [label="y(k)"]
        measurement -> optimizer [label="re-plan", constraint=false]
        """,
        height=320,
    ),
    "gp_mpc": DiagramSpec(
        body=r"""
        reference [label="r(k)", shape=plaintext]
        optimizer [label="GP-MPC controller\noptimizer + κσ tightening"]
        prediction [label="Corrected prediction model\nf̄ + μGP"]
        prior [label="Physics prior f̄"]
        gp [label="GP residual\nμGP, σGP", fillcolor="#99f6e4", fontcolor="#111827"]
        plant [label="Plant / process", fillcolor="#e2e8f0", fontcolor="#111827"]
        output [label="y(k)", shape=plaintext]
        reference -> optimizer
        prediction -> optimizer [label="model, uncertainty", constraint=false]
        prior -> prediction
        gp -> prediction
        optimizer -> plant [label="u₀"]
        plant -> output
        output -> optimizer [label="x̂(k)", constraint=false]
        plant -> gp [label="rollout data", style=dashed, constraint=false]
        """,
        height=395,
    ),
    "reckless": DiagramSpec(
        body=r"""
        reference [label="runsafe\nbeyond rail", shape=plaintext]
        error [label="Σ", shape=circle, fixedsize=true, width=0.42,
               fillcolor="#fde047", fontcolor="#111827"]
        policy [label="Unsafe balancing policy"]
        plant [label="Plant / process", fillcolor="#e2e8f0", fontcolor="#111827"]
        output [label="position y(t)", shape=plaintext]
        boundary [label="Rail limit", shape=diamond, color="#fb7185"]
        reference -> error [label="+"]
        error -> policy -> plant
        plant -> output
        output -> error [label="−", constraint=false]
        output -> boundary [label="violation"]
        """,
        height=300,
    ),
    "state_feedback": DiagramSpec(
        body=r"""
        reference [label="r(t)", shape=plaintext]
        prefilter [label="Reference prefilter\nNr"]
        sum [label="Σ", shape=circle, fixedsize=true, width=0.42,
             fillcolor="#fde047", fontcolor="#111827"]
        plant [label="Plant / process", fillcolor="#e2e8f0", fontcolor="#111827"]
        output [label="y(t)", shape=plaintext]
        observer [label="State observer\nx̂"]
        gain [label="Feedback gain\n−K"]
        reference -> prefilter -> sum [label="+"]
        sum -> plant [label="u"]
        plant -> output
        output -> observer [label="y"]
        observer -> gain
        gain -> sum [label="−Kx̂", constraint=false]
        """,
        height=350,
    ),
    "feedback_linearisation": DiagramSpec(
        body=r"""
        reference [label="r(t)", shape=plaintext]
        error [label="Σ", shape=circle, fixedsize=true, width=0.42,
               fillcolor="#fde047", fontcolor="#111827"]
        linear [label="Linear controller"]
        sum [label="Σ", shape=circle, fixedsize=true, width=0.42,
             fillcolor="#fde047", fontcolor="#111827"]
        plant [label="Nonlinear plant", fillcolor="#e2e8f0", fontcolor="#111827"]
        output [label="y(t)", shape=plaintext]
        cancellation [label="Nonlinear cancellation\nΔu(θ, θ̇)"]
        reference -> error [label="+"]
        error -> linear [label="e(t)"]
        linear -> sum [label="virtual input v"]
        sum -> plant [label="u = v + Δu"]
        plant -> output
        output -> error [label="−", constraint=false]
        plant -> cancellation [label="θ, θ̇"]
        cancellation -> sum [label="Δu", constraint=false]
        """,
        height=375,
    ),
    "mpsc": DiagramSpec(
        body=r"""
        reference [label="r(t)", shape=plaintext]
        policy [label="Nominal controller / policy"]
        filter [label="MPSC safety filter\nnearest certifiably safe action"]
        plant [label="Plant / process", fillcolor="#e2e8f0", fontcolor="#111827"]
        output [label="y(t)", shape=plaintext]
        reference -> policy
        policy -> filter [label="uprop"]
        filter -> plant [label="ucert"]
        plant -> output
        output -> policy [label="state feedback", constraint=false]
        output -> filter [label="safety state", constraint=false]
        """,
        height=320,
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
            rankdir={specification.rankdir},
            bgcolor="transparent",
            margin=0,
            pad=0.08,
            nodesep=0.24,
            ranksep=0.36,
            splines=polyline
        ]
        node [
            shape=box,
            style="filled",
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
