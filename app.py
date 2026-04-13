import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import control as ct
from plant import CartPolePlant
from controller import (PIDController, StateSpaceController, LQRController,
                        LuenbergerObserver, KalmanFilter, LQIController,
                        TrajectoryPlanner)

# ============================================================================ #
#  Mermaid renderer                                                             #
# ============================================================================ #
def render_mermaid(code: str, height: int = 180):
    components.html(
        f"""<body style="background-color:#0f172a;margin:0;display:flex;
            justify-content:center;align-items:center;height:100%;overflow:hidden;">
        <div class="mermaid">{code}</div>
        <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{startOnLoad:true,theme:'dark',themeVariables:{{
            background:'#0f172a',primaryColor:'#1e293b',
            primaryBorderColor:'#38bdf8',primaryTextColor:'#f8fafc',
            lineColor:'#94a3b8'}}}});
        </script></body>""",
        height=height)


# ============================================================================ #
#  Helpers                                                                      #
# ============================================================================ #
def matrix_to_html(M: np.ndarray, fmt: str = "{:.3f}") -> str:
    """Render a numpy matrix as a dark-styled HTML table."""
    rows = ""
    for row in M:
        cells = "".join(
            f"<td style='padding:3px 8px;text-align:right;color:#38bdf8;"
            f"font-family:monospace;font-size:0.82rem;'>{fmt.format(v)}</td>"
            for v in row
        )
        rows += f"<tr>{cells}</tr>"
    return (
        "<table style='border-collapse:collapse;background:#0f172a;"
        "border:1px solid #1e293b;border-radius:4px;'>"
        f"{rows}</table>"
    )

def pole_label(p: complex) -> str:
    if abs(p.imag) < 1e-6:
        return f"{p.real:.3f}"
    sign = "+" if p.imag >= 0 else "−"
    return f"{p.real:.3f} {sign} {abs(p.imag):.3f}j"

PLOT_THEME = dict(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font          = dict(color="#f8fafc"),
    xaxis         = dict(showgrid=True, gridcolor="#1e293b"),
    yaxis         = dict(showgrid=True, gridcolor="#1e293b"),
    legend        = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc")),
)

# ============================================================================ #
#  Page config                                                                  #
# ============================================================================ #
st.set_page_config(page_title="Advanced Control Systems",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background-color:#020617;}
[data-testid="stSidebar"]         {background-color:#0f172a;}
header[data-testid="stHeader"]    {background:rgba(0,0,0,0);}
button[kind="header"]{background-color:rgba(56,189,248,0.1)!important;
  border:1px solid #38bdf8!important;color:#38bdf8!important;
  border-radius:5px!important;}
.block-container{padding-top:1rem;padding-bottom:1rem;max-width:95%;}
.main-header{font-size:2.0rem;font-weight:700;color:#38bdf8;
             margin-bottom:0px;margin-top:-30px;}
.sub-header {font-size:1.0rem;font-weight:300;color:#94a3b8;margin-bottom:10px;}
.section-title{font-size:1.1rem;font-weight:600;color:#38bdf8;
               margin-top:18px;margin-bottom:6px;}
.info-box{background:#0f172a;border:1px solid #1e293b;border-radius:6px;
          padding:10px 14px;margin-bottom:8px;}
[data-testid="stMetricValue"]{font-size:1.5rem!important;}
[data-testid="stMetricLabel"]{font-size:0.85rem!important;}
[data-testid="stSidebarUserContent"]{padding-top:1rem;}
</style>
<div class="main-header">Inverted Pendulum Digital Twin</div>
<div class="sub-header">
  MuJoCo Nonlinear Plant · Full-State Feedback · Optimal Control ·
  Stochastic Estimation · Feedback Linearisation
</div>
<hr style="border-color:#1e293b;margin-top:5px;margin-bottom:15px;">
""", unsafe_allow_html=True)


# ============================================================================ #
#  SIDEBAR                                                                      #
# ============================================================================ #
st.sidebar.markdown("### 🎛️ Control Panel")
st.sidebar.markdown("---")

# --- Simulation Duration ---
st.sidebar.subheader("⏱️ Simulation Duration")
total_time = st.sidebar.select_slider(
    "Duration (s)", options=[5, 8, 10, 12, 15, 20, 25, 30], value=10)
st.sidebar.markdown("---")

# --- Physical Parameters ---
st.sidebar.subheader("⚙️ Physical Parameters")
m_c         = st.sidebar.slider("Cart Mass (kg)",          0.1,  5.0,  1.0,  0.1)
m_p         = st.sidebar.slider("Pendulum Mass (kg)",      0.01, 2.0,  0.1,  0.01)
l           = st.sidebar.slider("Pendulum Length (m)",     0.1,  2.0,  0.5,  0.1)
track_limit = st.sidebar.slider("Track Half-Length (± m)", 1.0,  5.0,  2.8,  0.1)

env = CartPolePlant(m_c=m_c, m_p=m_p, l=l)

# --- Initial Conditions ---
st.sidebar.subheader("📍 Initial Conditions")
init_p         = st.sidebar.slider("Starting Position (m)", -track_limit, track_limit, 0.0, 0.1)
init_theta_deg = st.sidebar.slider("Starting Angle (deg)",  -180.0, 180.0, 5.7, 0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Global Setpoint")
target_p = st.sidebar.slider("Target Position (m)", -track_limit, track_limit, 0.0, 0.1)

# --- Controller ---
st.sidebar.markdown("---")
controller_type = st.sidebar.selectbox(
    "🧠 Brain Architecture",
    ["PID (Classical)", "Pole Placement (State-Space)",
     "LQR (Optimal Control)", "LQI (Integral Optimal Control)"])

with st.sidebar.expander(f"ℹ️ Theory: {controller_type}"):
    if controller_type == "PID (Classical)":
        render_mermaid("""flowchart LR
            R((Target))-->Sum{+}
            Sum-->|Pos Error|C[PID]
            C-->|Force|P[Plant]
            P-->|Position|Y((Output))
            P--->|-|Sum""")
        st.markdown(r"""
        **SISO — wired to cart POSITION error (metres).**
        $u = K_p e + K_i\int e\,dt + K_d\dot{e}$

        ⚠️ No explicit angle feedback. Cannot stabilise the upright
        equilibrium on its own — demonstrates the SISO limitation.
        """)
    elif controller_type == "LQI (Integral Optimal Control)":
        render_mermaid("""flowchart LR
            R((Target))-->Sum1{+}
            Sum1-->|Error e|Int[∫]
            Int-->|x_i|Ki[K_i]
            Ki-->Sum2{+}
            P[Plant]-->|State x|K[K]
            K-->|-|Sum2
            Sum2-->|u|P""")
        st.markdown(r"""
        **Augmented state:** $x_{aug}=[x,\dot x,\theta,\dot\theta,\int e]^\top$

        Integral term enforces zero steady-state error under constant
        disturbances (Internal Model Principle).
        ⚠️ High $Q_{int}$ + slow observer → integral windup.
        """)
    else:
        render_mermaid("""flowchart LR
            R((Target))-->N[Nr]
            N-->Sum{+}
            Sum-->|u|P[Plant]
            P-->|y|Obs[Observer]
            Obs-->|x̂|K[K]
            K--->|-|Sum""")
        if controller_type == "Pole Placement (State-Space)":
            st.markdown(r"""
            **Pole Placement:** specify $\zeta,\omega_n$ → compute $K$ via
            Ackermann. Nr pre-filter ensures DC tracking. No cost on effort.
            """)
        else:
            st.markdown(r"""
            **LQR:** minimise $J=\int(x'Qx+u'Ru)dt$.
            Globally optimal for the linear model. Nr valid only under
            zero disturbance — use LQI for wind/friction rejection.
            """)

if controller_type == "PID (Classical)":
    kp = st.sidebar.slider("Kp", -200.0, 200.0,  0.0, 1.0)
    ki = st.sidebar.slider("Ki",  -50.0,  50.0,  0.0, 0.1)
    kd = st.sidebar.slider("Kd", -100.0, 100.0,  0.0, 0.1)
    controller = PIDController(kp=kp, ki=ki, kd=kd)
elif controller_type == "Pole Placement (State-Space)":
    zeta = st.sidebar.slider("Zeta (Damping Ratio)",   0.1, 2.0,  0.7, 0.1)
    wn   = st.sidebar.slider("Wn (Natural Frequency)", 1.0, 10.0, 3.5, 0.1)
    controller = StateSpaceController(A=env.A, B=env.B, zeta=zeta, wn=wn)
elif controller_type == "LQR (Optimal Control)":
    st.sidebar.markdown("**Q — State Cost**")
    q_pos    = st.sidebar.slider("Q_pos", 0.1, 500.0, 100.0, 1.0)
    q_ang    = st.sidebar.slider("Q_ang", 0.1, 500.0,  10.0, 1.0)
    st.sidebar.markdown("**R — Actuator Cost**")
    r_weight = st.sidebar.slider("R",     0.01, 50.0,   1.0, 0.01)
    controller = LQRController(A=env.A, B=env.B,
                               q_pos=q_pos, q_ang=q_ang, r_weight=r_weight)
elif controller_type == "LQI (Integral Optimal Control)":
    st.sidebar.markdown("**Q & R**")
    q_pos    = st.sidebar.slider("Q_pos", 0.1, 500.0, 100.0, 1.0)
    q_ang    = st.sidebar.slider("Q_ang", 0.1, 500.0,  10.0, 1.0)
    q_int    = st.sidebar.slider("Q_int", 0.1, 500.0, 150.0, 1.0)
    r_weight = st.sidebar.slider("R",     0.01, 50.0,   1.0, 0.01)
    controller = LQIController(A=env.A, B=env.B,
                               q_pos=q_pos, q_ang=q_ang,
                               q_int=q_int, r_weight=r_weight)

# --- Nonlinear Dynamics ---
st.sidebar.markdown("---")
st.sidebar.subheader("🧬 Nonlinear Dynamics")
with st.sidebar.expander("ℹ️ Theory: Feedback Linearisation"):
    render_mermaid("""flowchart LR
        C[Linear Controller]-->|v|FL[Inverse Dynamics]
        P[Plant State]-->FL
        FL-->|u|Plant""", height=130)
    st.markdown(r"""
    Cancels Coriolis ($ml\dot\theta^2\sin\theta$) and gravity distortion
    $\bigl(m_p g(\theta - \sin\theta\cos\theta)\bigr)$ in real-time, so the
    inner linear controller sees a perfect $\dot x = Ax+Bu$ system.
    Extends the controllable region from ~$20°$ to ~$40°$+.
    """)
use_fl = st.sidebar.checkbox("Enable Feedback Linearization", value=False)

# --- Motion Profiler ---
st.sidebar.markdown("---")
st.sidebar.subheader("🛤️ Motion Profiler")
with st.sidebar.expander("ℹ️ Theory: Quintic Trajectory"):
    st.markdown(r"""
    5th-order polynomial with zero vel/acc at endpoints.
    Feedforward: $F_{ff}=(m_c+m_p)\,a_{ref}$ — proactive inertia compensation.
    """)
use_trajectory = st.sidebar.checkbox("Enable Smooth Trajectory", value=False)
if use_trajectory:
    move_start    = st.sidebar.slider("Start Move At (s)", 0.0, 5.0, 1.0, 0.5)
    move_duration = st.sidebar.slider("Move Duration (s)", 0.5, 5.0, 2.5, 0.1)
else:
    move_start, move_duration = 0.0, 0.1

# --- Estimation ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔭 State Estimation")
estimator_type = st.sidebar.selectbox(
    "Estimator", ["None", "Luenberger (Deterministic)", "Kalman Filter (Stochastic)"])
with st.sidebar.expander("ℹ️ Theory: Observers"):
    st.markdown(r"""
    Both reconstruct $\hat x \in\mathbb{R}^4$ from $y=[x,\theta]$ only.

    **Luenberger:** $\dot{\hat x}=A\hat x+Bu+L(y-C\hat x)$, deterministic.
    High gains → noise amplification.

    **Kalman (discrete ZOH):** DARE-optimal gain. Balances process noise $Q_v$
    vs sensor noise $R_w$. Mathematically consistent discrete formulation.
    """)
if estimator_type == "Luenberger (Deterministic)":
    obs_speed = st.sidebar.slider("Observer Speed ×", 1.0, 10.0, 4.0, 0.5)
elif estimator_type == "Kalman Filter (Stochastic)":
    qv_mult = st.sidebar.slider("Process Noise Qv", 0.001, 10.0, 0.1, format="%.3f")
    rw_mult = st.sidebar.slider("Sensor Noise Rw",  0.001, 10.0, 1.0, format="%.3f")

# --- Real-World Constraints ---
st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Real-World Constraints")
with st.sidebar.expander("ℹ️ Theory: Physical Limitations"):
    st.markdown(r"""
    **Saturation:** motor voltage ceiling → clip $u$.
    **Friction:** set inside MuJoCo joints (correct physics layer — not input distortion).
    **Noise:** encoder noise injected into $y$ before estimation.
    **Disturbances:** impulse or continuous wind bypasses saturation.
    """)

use_saturation = st.sidebar.checkbox("Actuator Saturation")
max_force = st.sidebar.slider("Max Force (N)", 1.0, 100.0, 15.0, 1.0) if use_saturation else float('inf')

use_friction = st.sidebar.checkbox("Rail Friction")
if use_friction:
    viscous_friction = st.sidebar.slider("Viscous $b_v$ (N·s/m)", 0.0, 20.0, 2.0, 0.1)
    coulomb_friction = st.sidebar.slider("Coulomb $F_c$ (N)",      0.0, 10.0, 0.5, 0.1)
else:
    viscous_friction = coulomb_friction = 0.0
env.set_friction(cart_frictionloss=coulomb_friction, cart_damping=viscous_friction)

use_noise = st.sidebar.checkbox("Sensor Noise")
noise_std = st.sidebar.slider("Angle Noise σ (deg)", 0.01, 5.0, 0.5, 0.01) if use_noise else 0.0

st.sidebar.markdown("**External Disturbance**")
dist_type = st.sidebar.radio("Profile", ["None","Impulse Gust (0.1s)","Continuous Wind"])
if dist_type != "None":
    dist_mag  = st.sidebar.slider("Force (N)",      -100.0, 100.0, 10.0, 1.0)
    dist_time = st.sidebar.slider("Start Time (s)",    1.0,   20.0,  2.0, 0.5)
else:
    dist_mag = dist_time = 0.0

# ============================================================================ #
#  System Analysis — computed before simulation                                 #
# ============================================================================ #
dt               = 0.02
steps            = int(total_time / dt)
TARGET_TOLERANCE = 0.05
LINEARISATION_LIMIT_DEG = 15.0   # sin θ ≈ θ valid below this angle

A, B, C_sys = env.A, env.B, env.C

# --- Controllability & Observability ---
C_obs_mat = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])   # measurement matrix
n = A.shape[0]

ctrl_matrix = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
obsv_matrix = np.vstack([C_obs_mat @ np.linalg.matrix_power(A, i) for i in range(n)])
ctrl_rank   = np.linalg.matrix_rank(ctrl_matrix)
obsv_rank   = np.linalg.matrix_rank(obsv_matrix)
is_ctrl     = ctrl_rank == n
is_obsv     = obsv_rank == n

# --- Closed-loop poles ---
if hasattr(controller, 'K'):
    K_ctrl = controller.K
    if K_ctrl.ndim == 2:
        K_ctrl = K_ctrl[0]
    cl_poles = np.linalg.eigvals(A - B.flatten()[:, None] * K_ctrl[None, :])
else:
    cl_poles = None

# ============================================================================ #
#  Build Estimator                                                              #
# ============================================================================ #
init_theta_rad = np.radians(init_theta_deg)
current_state  = np.array([init_p, 0.0, init_theta_rad, 0.0])
use_estimator  = estimator_type != "None"

if use_estimator:
    if estimator_type == "Luenberger (Deterministic)":
        if hasattr(controller, 'poles'):
            obs_poles = np.array(controller.poles) * obs_speed
        else:
            obs_poles = np.array([-10., -11., -12., -13.])
        estimator = LuenbergerObserver(A, B, C_obs_mat, obs_poles)
    else:
        Q_v = np.eye(n) * qv_mult
        R_w = np.eye(2) * rw_mult
        estimator = KalmanFilter(A, B, C_obs_mat, Q_v, R_w, dt=dt)
    estimator.reset(current_state + np.array([0.2, 0., 0., 0.]))

planner = TrajectoryPlanner(p_start=init_p, p_end=target_p, duration=move_duration)

# ============================================================================ #
#  Simulation Loop                                                              #
# ============================================================================ #
history     = []
obs_history = []
ref_history = []
u_history   = []
total_energy       = 0.0
sat_steps          = 0
tracking_error_sq  = 0.0   # for RMS tracking error
terminated  = False
prev_y      = np.array([init_p, init_theta_rad])

for step_idx in range(steps):
    t = step_idx * dt
    history.append(current_state.copy())
    p = current_state[0]

    # 1. Trajectory
    if use_trajectory:
        p_ref, _, a_ref = planner.get_state(t - move_start)
        u_ff = (m_c + m_p) * a_ref
    else:
        p_ref, u_ff = target_p, 0.0
    ref_history.append(p_ref)

    # 2. Measurement
    y = C_obs_mat @ current_state
    if use_noise:
        y[1] += np.radians(np.random.normal(0.0, noise_std))

    # 3. Estimation
    if use_estimator:
        u_prev = u_history[-1] if u_history else 0.0
        x_hat  = estimator.update(u_prev, y, dt)
        obs_history.append(x_hat.copy())
        feedback_state = x_hat.copy()
    else:
        feedback_state    = np.zeros(4)
        feedback_state[0] = y[0]
        feedback_state[2] = y[1]
        feedback_state[1] = (y[0] - prev_y[0]) / dt
        feedback_state[3] = (y[1] - prev_y[1]) / dt
        obs_history.append(feedback_state.copy())
    prev_y = y.copy()

    # Bounds check
    if abs(p) > track_limit:
        terminated = True
        break

    # 4. Control
    if controller_type == "PID (Classical)":
        u_fb = controller.compute(p_ref, feedback_state[0], dt)
    elif controller_type == "LQI (Integral Optimal Control)":
        u_fb = controller.compute(p_ref, feedback_state, dt)
    else:
        u_fb = controller.compute(p_ref, feedback_state)
    u = u_fb + u_ff

    # 4.5 Feedback Linearisation
    if use_fl:
        th, om = feedback_state[2], feedback_state[3]
        u += -m_p * l * om**2 * np.sin(th)
        u -= m_p * 9.81 * (th - np.sin(th) * np.cos(th))

    # 5. Saturation
    if use_saturation:
        u_clipped = np.clip(u, -max_force, max_force)
        if abs(u_clipped) < abs(u):
            sat_steps += 1
        u = u_clipped

    total_energy      += u**2 * dt
    tracking_error_sq += (feedback_state[0] - p_ref)**2 * dt
    u_history.append(float(u))

    # 5.5 Wind
    u_eff = u
    if dist_type == "Impulse Gust (0.1s)" and dist_time <= t <= dist_time + 0.1:
        u_eff += dist_mag
    elif dist_type == "Continuous Wind" and t >= dist_time:
        u_eff += dist_mag

    current_state = env.step(current_state, u_eff, dt)

# ============================================================================ #
#  Derived Metrics                                                              #
# ============================================================================ #
final_p         = history[-1][0]
final_theta_deg = np.degrees(history[-1][2])
pos_stable      = abs(final_p - target_p) <= TARGET_TOLERANCE and not terminated
angle_stable    = abs(final_theta_deg) <= 0.5
rms_tracking    = np.sqrt(tracking_error_sq / max(len(history) * dt, 1e-9))
peak_force      = max(abs(u) for u in u_history) if u_history else 0.0
sat_pct         = 100.0 * sat_steps / max(len(u_history), 1)
steady_state_err = abs(final_p - target_p)

time_arr  = np.arange(len(history)) * dt
p_arr     = [s[0] for s in history]
theta_arr = [np.degrees(s[2]) for s in history]
omega_arr = [np.degrees(s[3]) for s in history]

# ============================================================================ #
#  ── SECTION B: Metrics Row ────────────────────────────────────────────────  #
# ============================================================================ #
st.markdown("<hr style='border-color:#1e293b;margin:10px 0;'>",
            unsafe_allow_html=True)

mc1, mc2, mc3, mc4, mc5, mc6, mc7 = st.columns(7)
with mc1:
    status = ("STABLE 🟢" if pos_stable
              else ("OUT OF BOUNDS 🔴" if terminated else "OFF TARGET 🔴"))
    st.metric("Position", status)
with mc2:
    st.metric("Final Pos", f"{final_p:.3f} m")
with mc3:
    st.metric("Angle", "UPRIGHT 🟢" if angle_stable else "FALLEN 🔴")
with mc4:
    st.metric("RMS Track Err", f"{rms_tracking:.4f} m")
with mc5:
    st.metric("Control Energy", f"{total_energy:.1f} J")
with mc6:
    st.metric("Peak Force", f"{peak_force:.1f} N")
with mc7:
    st.metric("Saturation", f"{sat_pct:.1f}%" if use_saturation else "OFF ⚪")

# ============================================================================ #
#  ── SECTION C: Animation ──────────────────────────────────────────────────  #
# ============================================================================ #
st.markdown("<hr style='border-color:#1e293b;margin:10px 0;'>",
            unsafe_allow_html=True)

cart_width, cart_height = 0.4, 0.15
wheel_radius = 0.05
track_y      = -0.5
wheel_y_bot  = track_y
wheel_y_top  = track_y + wheel_radius * 2
cart_y_bot   = wheel_y_top
cart_y_top_v = cart_y_bot + cart_height
pivot_y      = cart_y_top_v
visual_l     = env.l * 1.5


def get_shapes(cart_p: float) -> list:
    p = cart_p
    return [
        dict(type="line", x0=-track_limit-.2, y0=track_y-.01,
             x1=track_limit+.2, y1=track_y-.01,
             line=dict(color="#334155", width=10), layer="below"),
        dict(type="line", x0=-track_limit-.2, y0=track_y,
             x1=track_limit+.2, y1=track_y,
             line=dict(color="#64748b", width=2), layer="below"),
        # Linearisation boundary lines on track
        dict(type="line", x0=-track_limit-.2,
             y0=track_y-.12, x1=track_limit+.2, y1=track_y-.12,
             line=dict(color="rgba(245,158,11,0.0)", width=0), layer="below"),
        # Tolerance band
        dict(type="rect",
             x0=target_p-TARGET_TOLERANCE, y0=track_y-.08,
             x1=target_p+TARGET_TOLERANCE, y1=track_y+.08,
             fillcolor="rgba(34,197,94,0.2)",
             line_width=1, line_color="rgba(34,197,94,0.5)", layer="below"),
        dict(type="line", x0=target_p, y0=track_y-.08,
             x1=target_p, y1=track_y+.08,
             line=dict(color="#f43f5e", width=2, dash="dot"), layer="below"),
        # Wheels
        dict(type="circle",
             x0=p-.15-wheel_radius, y0=wheel_y_bot,
             x1=p-.15+wheel_radius, y1=wheel_y_top,
             fillcolor="#020617", line_color="#000", layer="below"),
        dict(type="circle",
             x0=p-.15-wheel_radius*.4, y0=wheel_y_bot+wheel_radius*.6,
             x1=p-.15+wheel_radius*.4, y1=wheel_y_top-wheel_radius*.6,
             fillcolor="#cbd5e1", line_width=0, layer="below"),
        dict(type="circle",
             x0=p+.15-wheel_radius, y0=wheel_y_bot,
             x1=p+.15+wheel_radius, y1=wheel_y_top,
             fillcolor="#020617", line_color="#000", layer="below"),
        dict(type="circle",
             x0=p+.15-wheel_radius*.4, y0=wheel_y_bot+wheel_radius*.6,
             x1=p+.15+wheel_radius*.4, y1=wheel_y_top-wheel_radius*.6,
             fillcolor="#cbd5e1", line_width=0, layer="below"),
        # Cart body
        dict(type="rect",
             x0=p-cart_width/2, y0=cart_y_bot,
             x1=p+cart_width/2, y1=cart_y_top_v,
             fillcolor="#a16207", line_color="#000", line_width=2, layer="below"),
        dict(type="rect",
             x0=p-cart_width/2+.03, y0=cart_y_bot+.03,
             x1=p+cart_width/2-.03, y1=cart_y_top_v-.03,
             fillcolor="#eab308", line_width=0, layer="below"),
    ]


fig_anim = go.Figure()
p0, th0 = history[0][0], history[0][2]

if use_estimator:
    p0h, th0h = obs_history[0][0], obs_history[0][2]
    fig_anim.add_trace(go.Scatter(
        x=[p0h, p0h + visual_l*np.sin(th0h)],
        y=[pivot_y, pivot_y + visual_l*np.cos(th0h)],
        mode="lines", line=dict(color="#38bdf8", width=4, dash="dot"),
        name="Estimate", hoverinfo="skip"))

fig_anim.add_trace(go.Scatter(
    x=[p0, p0 + visual_l*np.sin(th0)],
    y=[pivot_y, pivot_y + visual_l*np.cos(th0)],
    mode="lines+markers",
    line=dict(color="#f43f5e", width=8),
    marker=dict(color=["#f8fafc","#f8fafc"], size=[16,16],
                line=dict(color="#000", width=2)),
    name="True Plant", hoverinfo="skip"))

fig_anim.update_layout(shapes=get_shapes(p0))

frames = []
for i, state in enumerate(history):
    p_i, th_i = state[0], state[2]
    fd = []
    if use_estimator:
        ph, thh = obs_history[i][0], obs_history[i][2]
        fd.append(go.Scatter(
            x=[ph, ph + visual_l*np.sin(thh)],
            y=[pivot_y, pivot_y + visual_l*np.cos(thh)]))
    fd.append(go.Scatter(
        x=[p_i, p_i + visual_l*np.sin(th_i)],
        y=[pivot_y, pivot_y + visual_l*np.cos(th_i)]))
    frames.append(go.Frame(data=fd, layout=go.Layout(shapes=get_shapes(p_i)),
                           name=str(i)))

fig_anim.frames = frames
fig_anim.update_layout(
    xaxis=dict(range=[-track_limit-.2, track_limit+.2], autorange=False, visible=False),
    yaxis=dict(range=[-1.0, 2.5], autorange=False, scaleanchor="x",
               scaleratio=1, visible=False),
    width=1200, height=480, showlegend=True,
    legend=dict(yanchor="top", y=.99, xanchor="left", x=.01,
                bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc")),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=10, b=0),
    updatemenus=[dict(
        type="buttons", showactive=False,
        y=1.08, x=0.5, xanchor="center", yanchor="bottom", direction="left",
        buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, {"frame": {"duration": int(dt*1000), "redraw": True},
                              "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate", "transition": {"duration": 0}}]),
            dict(label="⏮ Reset", method="animate",
                 args=[["0"], {"frame": {"duration": 0, "redraw": True},
                               "mode": "immediate", "transition": {"duration": 0}}]),
        ])])
st.plotly_chart(fig_anim, use_container_width=True)

# ============================================================================ #
#  ── SECTION D: Diagnostic Telemetry ───────────────────────────────────────  #
# ============================================================================ #
st.markdown("<hr style='border-color:#1e293b;margin:4px 0 10px;'>",
            unsafe_allow_html=True)
st.markdown("<div class='section-title'>📊 Diagnostic Telemetry</div>",
            unsafe_allow_html=True)

# --- Estimator convergence ---
if use_estimator:
    err_p     = [history[i][0] - obs_history[i][0] for i in range(len(history))]
    err_theta = [np.degrees(history[i][2] - obs_history[i][2])
                 for i in range(len(history))]
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(x=time_arr, y=err_p,
        mode='lines', line=dict(color="#38bdf8", width=2), name="Pos Error (m)"))
    fig_conv.add_trace(go.Scatter(x=time_arr, y=err_theta,
        mode='lines', line=dict(color="#f43f5e", width=2, dash="dash"),
        name="Angle Error (deg)", yaxis="y2"))
    fig_conv.update_layout(
        height=220, title="Observer Convergence: True − Estimated",
        xaxis_title="Time (s)",
        yaxis =dict(title="Pos Error (m)",   showgrid=True,
                    gridcolor="#1e293b", color="#38bdf8"),
        yaxis2=dict(title="Angle Error (°)", overlaying="y", side="right",
                    color="#f43f5e", showgrid=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        xaxis=dict(showgrid=True, gridcolor="#1e293b"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc")),
        margin=dict(l=0, r=60, t=35, b=0))
    st.plotly_chart(fig_conv, use_container_width=True)

# --- Row 1: Position | Angle ---
r1c1, r1c2 = st.columns(2)

fig_pos = go.Figure()
if use_trajectory:
    fig_pos.add_trace(go.Scatter(x=time_arr, y=ref_history,
        mode='lines', line=dict(color="#94a3b8", width=2, dash="dash"),
        name="Planned Path"))
fig_pos.add_trace(go.Scatter(x=time_arr, y=p_arr,
    mode='lines', line=dict(color="#38bdf8", width=3), name="Position"))
fig_pos.add_hline(y=target_p, line_dash="dash", line_color="#f43f5e",
                  annotation_text="target", annotation_font_color="#f43f5e")
# Steady-state error annotation
if not terminated:
    fig_pos.add_annotation(
        x=time_arr[-1], y=final_p,
        text=f"SSE={steady_state_err:.3f}m",
        showarrow=True, arrowhead=2, font=dict(color="#f59e0b", size=11),
        arrowcolor="#f59e0b", ax=-40, ay=-30)
fig_pos.update_layout(height=300, title="Cart Position Tracking",
    xaxis_title="Time (s)", yaxis_title="Position (m)",
    yaxis=dict(showgrid=True, gridcolor="#1e293b",
               range=[-track_limit-.2, track_limit+.2]),
    **{k: v for k, v in PLOT_THEME.items() if k != 'yaxis'},
    margin=dict(l=0, r=0, t=30, b=0), showlegend=True)

fig_angle = go.Figure()
fig_angle.add_trace(go.Scatter(x=time_arr, y=theta_arr,
    mode='lines', line=dict(color="#f43f5e", width=3), showlegend=False))
fig_angle.add_hline(y=0, line_dash="dash", line_color="#38bdf8")
# Linearisation validity boundaries
fig_angle.add_hrect(y0= LINEARISATION_LIMIT_DEG, y1=max(max(theta_arr)+5, LINEARISATION_LIMIT_DEG+5),
                    fillcolor="rgba(245,158,11,0.08)", line_width=0)
fig_angle.add_hrect(y0=min(min(theta_arr)-5, -LINEARISATION_LIMIT_DEG-5),
                    y1=-LINEARISATION_LIMIT_DEG,
                    fillcolor="rgba(245,158,11,0.08)", line_width=0)
fig_angle.add_hline(y= LINEARISATION_LIMIT_DEG, line_dash="dot",
                    line_color="#f59e0b", line_width=1,
                    annotation_text=f"±{LINEARISATION_LIMIT_DEG}° lin. limit",
                    annotation_font_color="#f59e0b")
fig_angle.add_hline(y=-LINEARISATION_LIMIT_DEG, line_dash="dot",
                    line_color="#f59e0b", line_width=1)
fig_angle.update_layout(height=300, title="Pendulum Angle (shaded = nonlinear region)",
    xaxis_title="Time (s)", yaxis_title="Angle (°)",
    **PLOT_THEME, margin=dict(l=0, r=0, t=30, b=0))

with r1c1: st.plotly_chart(fig_pos,   use_container_width=True)
with r1c2: st.plotly_chart(fig_angle, use_container_width=True)

# --- Row 2: Control Force | Phase Portrait ---
r2c1, r2c2 = st.columns(2)

fig_u = go.Figure()
fig_u.add_trace(go.Scatter(x=time_arr[:len(u_history)], y=u_history,
    mode='lines', line=dict(color="#a78bfa", width=2), name="u(t)"))
if use_saturation:
    fig_u.add_hline(y= max_force, line_dash="dot", line_color="#f59e0b",
                    annotation_text=f"+{max_force:.0f}N", annotation_font_color="#f59e0b")
    fig_u.add_hline(y=-max_force, line_dash="dot", line_color="#f59e0b",
                    annotation_text=f"−{max_force:.0f}N", annotation_font_color="#f59e0b")
    fig_u.add_annotation(x=time_arr[len(u_history)//2], y=max_force*0.6,
        text=f"Sat: {sat_pct:.1f}% of steps",
        font=dict(color="#f59e0b", size=11), showarrow=False)
fig_u.update_layout(height=300, title="Control Force u(t) — Peak & Saturation",
    xaxis_title="Time (s)", yaxis_title="Force (N)",
    **PLOT_THEME, margin=dict(l=0, r=0, t=30, b=0))

fig_phase = go.Figure()
fig_phase.add_trace(go.Scatter(x=theta_arr, y=omega_arr,
    mode='lines+markers',
    line=dict(color="#34d399", width=2),
    marker=dict(size=3, color="#34d399"), name="Trajectory"))
fig_phase.add_trace(go.Scatter(x=[theta_arr[0]], y=[omega_arr[0]],
    mode='markers',
    marker=dict(size=13, color="#f59e0b", symbol="circle"), name="Start"))
fig_phase.add_trace(go.Scatter(x=[theta_arr[-1]], y=[omega_arr[-1]],
    mode='markers',
    marker=dict(size=13, color="#f43f5e", symbol="x"), name="End"))
# Equilibrium annotation
fig_phase.add_annotation(x=0, y=0, text="equilibrium",
    showarrow=True, arrowhead=2, font=dict(color="#94a3b8", size=10),
    arrowcolor="#475569", ax=40, ay=-30)
fig_phase.update_layout(height=300,
    title="Phase Portrait θ vs θ̇ — spiral inward = stable",
    xaxis_title="θ (°)", yaxis_title="θ̇ (°/s)",
    xaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=True, zerolinecolor="#475569"),
    yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=True, zerolinecolor="#475569"),
    **{k: v for k, v in PLOT_THEME.items() if k not in ('xaxis','yaxis')},
    margin=dict(l=0, r=0, t=30, b=0))

with r2c1: st.plotly_chart(fig_u,     use_container_width=True)
with r2c2: st.plotly_chart(fig_phase, use_container_width=True)

# --- Row 3: Velocity states ---
r3c1, r3c2 = st.columns(2)

cart_vel_arr = [s[1] for s in history]
pole_vel_arr = [np.degrees(s[3]) for s in history]

fig_cv = go.Figure()
fig_cv.add_trace(go.Scatter(x=time_arr, y=cart_vel_arr,
    mode='lines', line=dict(color="#38bdf8", width=2), showlegend=False))
fig_cv.add_hline(y=0, line_dash="dash", line_color="#475569")
fig_cv.update_layout(height=240, title="Cart Velocity ẋ (m/s)",
    xaxis_title="Time (s)", yaxis_title="m/s",
    **PLOT_THEME, margin=dict(l=0, r=0, t=30, b=0))

fig_pv = go.Figure()
fig_pv.add_trace(go.Scatter(x=time_arr, y=pole_vel_arr,
    mode='lines', line=dict(color="#f43f5e", width=2), showlegend=False))
fig_pv.add_hline(y=0, line_dash="dash", line_color="#475569")
fig_pv.update_layout(height=240, title="Pole Angular Velocity θ̇ (°/s)",
    xaxis_title="Time (s)", yaxis_title="°/s",
    **PLOT_THEME, margin=dict(l=0, r=0, t=30, b=0))

with r3c1: st.plotly_chart(fig_cv, use_container_width=True)
with r3c2: st.plotly_chart(fig_pv, use_container_width=True)

# ============================================================================ #
#  ── SECTION E: Performance Summary Table ──────────────────────────────────  #
# ============================================================================ #
st.markdown("<hr style='border-color:#1e293b;margin:10px 0;'>",
            unsafe_allow_html=True)
st.markdown("<div class='section-title'>📋 Performance Summary</div>",
            unsafe_allow_html=True)

summary_data = {
    "Metric": [
        "Controller", "Estimator", "Final Position", "Position Error",
        "Final Angle", "RMS Tracking Error", "Steady-State Error",
        "Control Energy (∫u²dt)", "Peak Actuator Force",
        "Saturation (%)",
        "Controllable", "Observable",
        "Linearisation Valid",
        "Feedback Linearisation", "Trajectory Profiler",
    ],
    "Value": [
        controller_type, estimator_type,
        f"{final_p:.4f} m", f"{abs(final_p - target_p):.4f} m",
        f"{final_theta_deg:.3f}°",
        f"{rms_tracking:.4f} m",
        f"{steady_state_err:.4f} m",
        f"{total_energy:.2f} J",
        f"{peak_force:.2f} N",
        f"{sat_pct:.1f}%" if use_saturation else "N/A (no sat.)",
        f"✅ rank={ctrl_rank}" if is_ctrl else f"❌ rank={ctrl_rank}",
        f"✅ rank={obsv_rank}" if is_obsv else f"❌ rank={obsv_rank}",
        f"✅ θ₀={abs(init_theta_deg):.1f}° < {LINEARISATION_LIMIT_DEG}°"
            if abs(init_theta_deg) <= LINEARISATION_LIMIT_DEG
            else f"⚠️ θ₀={abs(init_theta_deg):.1f}° exceeds limit",
        "ON" if use_fl else "OFF",
        "ON" if use_trajectory else "OFF",
    ],
    "Status": [
        "—", "—",
        "🟢" if pos_stable else "🔴",
        "🟢" if abs(final_p - target_p) < TARGET_TOLERANCE else "🔴",
        "🟢" if angle_stable else "🔴",
        "🟢" if rms_tracking < 0.1 else ("🟡" if rms_tracking < 0.3 else "🔴"),
        "🟢" if steady_state_err < TARGET_TOLERANCE else "🔴",
        "🟢" if total_energy < 500 else "🟡",
        "🟢" if peak_force < max_force * 0.9 else "🔴",
        "🟢" if sat_pct < 5 else ("🟡" if sat_pct < 20 else "🔴"),
        "🟢" if is_ctrl else "🔴",
        "🟢" if is_obsv else "🔴",
        "🟢" if abs(init_theta_deg) <= LINEARISATION_LIMIT_DEG else "🟡",
        "🟢" if use_fl else "⚪",
        "🟢" if use_trajectory else "⚪",
    ]
}

table_html = """
<table style='width:100%;border-collapse:collapse;background:#0f172a;
              border:1px solid #1e293b;border-radius:6px;overflow:hidden;'>
  <thead>
    <tr style='background:#1e293b;'>
      <th style='padding:8px 14px;text-align:left;color:#94a3b8;
                 font-size:0.85rem;'>Metric</th>
      <th style='padding:8px 14px;text-align:left;color:#94a3b8;
                 font-size:0.85rem;'>Value</th>
      <th style='padding:8px 14px;text-align:center;color:#94a3b8;
                 font-size:0.85rem;'>Status</th>
    </tr>
  </thead>
  <tbody>
"""
for i, (metric, value, status) in enumerate(
        zip(summary_data["Metric"], summary_data["Value"], summary_data["Status"])):
    bg = "#0f172a" if i % 2 == 0 else "#111827"
    table_html += (
        f"<tr style='background:{bg};'>"
        f"<td style='padding:6px 14px;color:#94a3b8;font-size:0.83rem;'>{metric}</td>"
        f"<td style='padding:6px 14px;color:#f8fafc;font-family:monospace;"
        f"font-size:0.83rem;'>{value}</td>"
        f"<td style='padding:6px 14px;text-align:center;font-size:1.0rem;'>"
        f"{status}</td></tr>"
    )
table_html += "</tbody></table>"
st.markdown(table_html, unsafe_allow_html=True)