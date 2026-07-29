"""Structured theory reference for the control-method ladder."""

import pandas as pd
import streamlit as st

from ui_components import render_page_header, render_section_title


render_page_header(
    kicker="Concepts, assumptions and engineering trade-offs",
    title="Theory & Method Guide",
    subtitle=(
        "A compact reference for the models and algorithms used in this project. "
        "Each method is presented with the limitation that motivates the next one."
    ),
)

st.sidebar.markdown("### 📚 Guide")
st.sidebar.caption(
    "This page explains the design intent. Use Live Simulation for exploration "
    "and Controller Benchmarks for controlled comparisons."
)
st.sidebar.markdown(
    """
**Notation**

- `x` — cart position
- `θ` — pole angle; 0 is upright
- `u` — horizontal cart force
- `F` — discrete nonlinear dynamics
- `f̄` — nominal prior model
"""
)

overview_tab, foundations_tab, optimal_tab, learning_tab, decision_tab = st.tabs(
    [
        "Method map",
        "Model & estimation",
        "Optimal & predictive",
        "Learning & safety",
        "Decision guide",
    ]
)

with overview_tab:
    render_section_title("One plant, one escalating sequence of methods")
    first_row = st.columns(4)
    cards = [
        (
            "1 · Classical",
            "PID",
            "Acts on position error only. Useful as a structural baseline, "
            "but it cannot explicitly regulate all four states.",
        ),
        (
            "2 · State feedback",
            "Pole Placement · LQR · LQI",
            "Uses the full state. LQR prices performance and effort; LQI "
            "adds integral disturbance rejection.",
        ),
        (
            "3 · Nonlinear planning",
            "iLQR · MPC",
            "Plans through nonlinear dynamics. MPC re-plans online and "
            "includes actuator and rail constraints.",
        ),
        (
            "4 · Learning & assurance",
            "GP-MPC · MPSC",
            "Learns model residuals and exposes uncertainty; a safety filter "
            "certifies proposed actions at runtime.",
        ),
    ]
    for column, (kicker, title, body) in zip(first_row, cards):
        column.markdown(
            (
                "<div class='method-card'>"
                f"<div class='page-kicker'>{kicker}</div>"
                f"<h4>{title}</h4><p>{body}</p></div>"
            ),
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.info(
        "The project is intentionally cumulative: later methods do not make the "
        "earlier ones obsolete. They solve different problems and carry different "
        "computational and modelling assumptions."
    )

    render_section_title("Course-to-code map")
    st.dataframe(
        [
            {
                "Topic": "State feedback and observers",
                "Course": "Ch. 2",
                "Implementation": (
                    "StateSpaceController, LQRController, LQIController"
                ),
            },
            {
                "Topic": "Trajectory optimisation",
                "Course": "Ch. 4",
                "Implementation": "solve_ilqr, iLQRController",
            },
            {
                "Topic": "Constrained optimal control",
                "Course": "Ch. 5",
                "Implementation": "MPCController",
            },
            {
                "Topic": "Robust constraint tightening",
                "Course": "Ch. 5.5",
                "Implementation": "GPMPCController robust mode",
            },
            {
                "Topic": "Learning-based control",
                "Course": "Ch. 6",
                "Implementation": "ResidualGP, GPMPCController",
            },
            {
                "Topic": "Dynamic programming",
                "Course": "Ch. 1.2 / 7.3.1",
                "Implementation": "value_iteration.py",
            },
            {
                "Topic": "Runtime safety certification",
                "Course": "Safe-RL literature",
                "Implementation": "MPSCFilter",
            },
        ],
        hide_index=True,
        width="stretch",
    )

with foundations_tab:
    render_section_title("Nonlinear cart-pole model")
    model_left, model_right = st.columns([1.35, 1])
    with model_left:
        st.markdown(
            r"""
The state and input are

$$mathbf{x}=[x,\dot{x},\theta,\dot{\theta}]^\top,\qquad u=F_{cart},$$

with $\theta=0$ representing the unstable upright equilibrium. The denominator
$m_c+m_p\sin^2\theta$ keeps the nonlinear equations well-defined. The project
defines these dynamics once in CasADi and derives:

- continuous dynamics $\dot{\mathbf{x}}=f(\mathbf{x},u)$,
- exact Jacobians $A=\partial f/\partial x$, $B=\partial f/\partial u$,
- a discrete RK4 map $\mathbf{x}_{k+1}=F(\mathbf{x}_k,u_k)$.

The fast NumPy simulator is regression-tested against that symbolic source, so
controllers and the simulated plant cannot silently use different equations.
"""
        )
    with model_right:
        st.markdown(
            """
<div class="info-box">
<strong>Operating assumptions</strong><br><br>
• Point-mass pendulum<br>
• Rigid link and rigid cart<br>
• Horizontal rail<br>
• Force-controlled actuator<br>
• Optional viscous and Coulomb rail friction<br>
• 20 ms control interval; 2 ms physics sub-step
</div>
""",
            unsafe_allow_html=True,
        )
        st.warning(
            "Linear controllers are designed around upright. Beyond roughly "
            "|θ| = 20°, small-angle accuracy degrades and the nonlinear planner "
            "or feedback linearisation becomes materially important."
        )

    render_section_title("From output feedback to state estimation")
    estimator_left, estimator_right = st.columns(2)
    with estimator_left:
        st.markdown(
            r"""
#### Luenberger observer

$$\dot{\hat{x}}=A\hat{x}+Bu+L(y-C\hat{x})$$

Observer poles are placed faster than controller poles. This gives explicit
convergence speed, but aggressive gains amplify measurement noise.

**Use when:** the model is trusted and measurement noise is modest.
"""
        )
    with estimator_right:
        st.markdown(
            r"""
#### Steady-state Kalman filter

The gain is obtained from the discrete algebraic Riccati equation. $Q_v$ encodes
process uncertainty and $R_w$ measurement uncertainty.

**Use when:** noise statistics matter and velocity states must be reconstructed
without raw finite-difference amplification.
"""
        )

    render_section_title("Reference shaping and feedback linearisation")
    st.markdown(
        r"""
A position step requests infinite acceleration. The quintic planner replaces it
with a smooth path whose endpoint velocity and acceleration are both zero:

$$p(\tau)=p_0+\Delta p(10\tau^3-15\tau^4+6\tau^5),\qquad \tau=t/T.$$

A two-degree-of-freedom term $(m_c+m_p)a_{ref}$ handles nominal inertia, leaving
feedback to reject residual error. Feedback linearisation separately cancels
known nonlinear Coriolis and gravity distortion terms. It can enlarge the useful
region of a linear controller, but it is only as good as the state estimate and
parameter model used in the cancellation.
"""
    )

with optimal_tab:
    render_section_title("Linear optimal control")
    linear_left, linear_right = st.columns(2)
    with linear_left:
        st.markdown(
            r"""
#### Pole Placement

Chooses $K$ so the eigenvalues of $A-BK$ match desired closed-loop poles. It is
transparent and direct, but pole locations do not explicitly price actuator
effort. Fast poles can demand unrealistic force and interact badly with saturation.

#### LQR

$$J=\int_0^\infty(x^\top Qx+u^\top Ru)\,dt$$

LQR makes the performance–effort trade-off explicit. Its solution is locally
optimal for the linearised model, but the reference prefilter cannot remove
offset caused by unknown constant disturbances.
"""
        )
    with linear_right:
        st.markdown(
            r"""
#### LQI

Augments the state with integrated position error:

$$x_{aug}=[x,\dot{x},\theta,\dot{\theta},\int e\,dt]^\top.$$

The internal model principle forces persistent offset toward zero. Excessive
integral weight or prolonged saturation can cause wind-up, so LQI must be judged
with realistic actuator limits and observer transients.

**Benchmark signature:** under constant wind, LQR retains a steady-state offset;
LQI spends additional transient effort to remove it.
"""
        )

    render_section_title("Nonlinear optimisation")
    nonlinear_left, nonlinear_right = st.columns(2)
    with nonlinear_left:
        st.markdown(
            r"""
#### iLQR

iLQR alternates a local quadratic backward pass with a nonlinear forward rollout.
The policy along the trajectory is

$$u_t=\bar{u}_t+\alpha k_t+K_t(x_t-\bar{x}_t).$$

It solves global manoeuvres such as swing-up that a fixed upright gain cannot.
The trajectory is computed for a particular start/goal pair; online re-planning
is the natural next step.
"""
        )
    with nonlinear_right:
        st.markdown(
            r"""
#### Nonlinear MPC

At each control step MPC solves a finite-horizon problem subject to nonlinear
dynamics, force bounds and rail constraints, applies the first action, and then
re-plans from the measured state.

Constraints are part of the decision—not an after-the-fact clip. The cost is
computation: this project solves a real CasADi/IPOPT problem every 20 ms of
simulated time, so complex runs are intentionally slower than wall clock.
"""
        )

with learning_tab:
    render_section_title("Learning the model residual")
    st.markdown(
        "GP-MPC keeps a physics prior and learns only its transition error:"
    )
    st.latex(
        r"""
        \begin{aligned}
        x_{k+1} &= \bar f(x_k,u_k) + \delta f(x_k,u_k), \\
        \delta f &\sim \mathcal{GP}(\mu,\sigma^2).
        \end{aligned}
        """
    )
    st.markdown(
        """
The prior is deliberately imperfect—wrong pole mass and no friction. Two Gaussian
processes learn the velocity-state residuals from rollout data. The GP mean corrects
the prediction and its variance exposes where the learned model has little support.
"""
    )

    constraint_left, constraint_right = st.columns(2)
    with constraint_left:
        st.markdown(
            r"""
#### Chance-constrained mode

Uncertainty is propagated along the candidate plan and the usable rail is tightened:

$$|x_{pos,k}|\le L-\kappa\sqrt{\Sigma_k[0,0]}.$$

The margin can shrink as data improves. This is less conservative than a fixed
worst-case bound, provided the GP uncertainty represents the actual uncertainty.
"""
        )
    with constraint_right:
        st.markdown(
            r"""
#### Robust mode

Uses a fixed disturbance bound rather than learned confidence. It pays the same
margin regardless of data, but remains meaningful for bounded effects that the GP
never observed—such as an exogenous wind absent from the training distribution.
"""
        )

    render_section_title("Model Predictive Safety Certification")
    st.markdown(
        r"""
The MPSC filter wraps any policy. Given a proposed action $u_{prop}$, it searches
for the nearest action from which a constraint-satisfying recovery trajectory
still exists:

$$\min (u_0-u_{prop})^2\quad\text{s.t. dynamics, input/state limits and a terminal recovery set.}$$

The filter has no performance objective. A safe proposal passes through nearly
unchanged; an unsafe proposal is modified only when necessary to retain a safe
future. This separates **what the policy wants** from **what the system may safely do**.
"""
    )
    st.warning(
        "Honest limitation: the implementation uses a terminal recovery box as "
        "a practical stand-in for a formally computed robust invariant set. The "
        "robust GP-MPC mode likewise uses simplified first-order propagation."
    )

with decision_tab:
    render_section_title("Capability matrix")
    comparison = pd.DataFrame(
        [
            ["PID", "No", "No", "No", "Very low", "Educational SISO baseline"],
            ["Pole Placement", "Full state", "No", "No", "Very low", "Transparent transient design"],
            ["LQR", "Full state", "No", "No", "Very low", "Nominal local regulation"],
            ["LQI", "Full state + integral", "No", "Constant offsets", "Very low", "Disturbance rejection"],
            [
                "iLQR",
                "Full state",
                "Plan only",
                "Feedback along plan",
                "Offline high",
                "Nonlinear manoeuvres",
            ],
            [
                "MPC",
                "Full state",
                "Yes",
                "Via re-planning",
                "Online high",
                "Constraint-aware operation",
            ],
            [
                "GP-MPC",
                "Full state + data",
                "Yes",
                "Learned residual",
                "Online very high",
                "Model-error compensation",
            ],
            [
                "MPSC",
                "Policy wrapper",
                "Safety constraints",
                "Rejects unsafe actions",
                "Online high",
                "Runtime certification",
            ],
        ],
        columns=["Method", "Feedback", "Constraints", "Uncertainty/disturbance", "Compute", "Best fit"],
    )
    st.dataframe(comparison, hide_index=True, width="stretch")

    render_section_title("Choose by engineering need")
    choice_columns = st.columns(3)
    choice_columns[0].success(
        "**Fast nominal regulator**\n\nStart with LQR. Add a Kalman filter when measurements are noisy."
    )
    choice_columns[1].info(
        "**Hard operating limits**\n\nUse MPC when force and position "
        "constraints must influence the plan itself."
    )
    choice_columns[2].warning(
        "**Unknown model error**\n\nUse GP-MPC only with coverage "
        "diagnostics; retain a robust or safety fallback."
    )

    render_section_title("How to make a defensible comparison")
    st.markdown(
        """
1. Fix the plant, initial state, target, actuator bounds and disturbance profile.
2. Declare every random seed and rebuild stateful controllers for every run.
3. Report tracking, steady-state error, control effort, constraint violations and failures.
4. Keep wall-clock timing separate from control quality unless hardware and solver settings are fixed.
5. Preserve the protocol, implementation and result hashes with exported raw data.

The **Controller Benchmarks** page implements exactly this protocol and uses the
same `run_simulation` function as the live laboratory.
"""
    )
