import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from plant import CartPolePlant
from controller import PIDController, StateSpaceController, LQRController, LuenbergerObserver, KalmanFilter, LQIController, TrajectoryPlanner

# --- Custom Component: Mermaid Renderer ---
def render_mermaid(code: str, height: int = 180):
    """Injects Mermaid.js natively into the Streamlit iframe with custom Dark Mode styling."""
    components.html(
        f"""
        <body style="background-color: #0f172a; margin: 0; display: flex; justify-content: center; align-items: center; height: 100%; overflow: hidden;">
            <div class="mermaid">
                {code}
            </div>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ 
                    startOnLoad: true, 
                    theme: 'dark',
                    themeVariables: {{
                        background: '#0f172a',
                        primaryColor: '#1e293b',
                        primaryBorderColor: '#38bdf8',
                        primaryTextColor: '#f8fafc',
                        lineColor: '#94a3b8'
                    }}
                }});
            </script>
        </body>
        """,
        height=height
    )

# --- 1. Page Configuration (FORCED SIDEBAR OPEN) ---
st.set_page_config(
    page_title="Advanced Control Systems", 
    layout="wide",
    initial_sidebar_state="expanded" 
)

st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #020617; }
    [data-testid="stSidebar"] { background-color: #0f172a; }
    
    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0);
        color: #38bdf8;
    }
    header[data-testid="stHeader"] > div:first-child {
        visibility: visible !important;
    }
    
    button[kind="header"] {
        background-color: rgba(56, 189, 248, 0.1) !important;
        border: 1px solid #38bdf8 !important;
        color: #38bdf8 !important;
        border-radius: 5px !important;
    }

    .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 95%; }
    .main-header { font-size: 2.0rem; font-weight: 700; color: #38bdf8; margin-bottom: 0px; margin-top: -30px; }
    .sub-header { font-size: 1.0rem; font-weight: 300; color: #94a3b8; margin-bottom: 10px; }
    
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.9rem !important; }
    
    [data-testid="stSidebarUserContent"] { padding-top: 1rem; }
    </style>
    <div class="main-header">Inverted Pendulum Digital Twin</div>
    <div class="sub-header">Phase 7: Smooth Path Planning & 2-DOF Feedforward Control</div>
    <hr style="border-color: #1e293b; margin-top: 5px; margin-bottom: 15px;">
""", unsafe_allow_html=True)

# --- 2. UI Controls (Sidebar) ---
st.sidebar.markdown("### 🎛️ Control Panel")
st.sidebar.markdown("---")

st.sidebar.subheader("⚙️ Physical Parameters")
m_c = st.sidebar.slider("Cart Mass (kg)", 0.1, 5.0, 1.0, 0.1)
m_p = st.sidebar.slider("Pendulum Mass (kg)", 0.01, 2.0, 0.1, 0.01)
l = st.sidebar.slider("Pendulum Length (m)", 0.1, 2.0, 0.5, 0.1)
track_limit = st.sidebar.slider("Track Half-Length (± m)", 1.0, 5.0, 2.8, 0.1) 

env = CartPolePlant(m_c=m_c, m_p=m_p, l=l) 

st.sidebar.subheader("📍 Initial Conditions")
init_p = st.sidebar.slider("Starting Position (m)", -track_limit, track_limit, 0.0, 0.1)
init_theta_deg = st.sidebar.slider("Starting Angle (deg)", -180.0, 180.0, 5.7, 0.1)

# --- 🧠 BRAIN ARCHITECTURE ---
st.sidebar.markdown("---")
controller_type = st.sidebar.selectbox("🧠 Brain Architecture", ["PID (Classical)", "Pole Placement (State-Space)", "LQR (Optimal Control)", "LQI (Integral Optimal Control)"])

with st.sidebar.expander(f"ℹ️ Theory: {controller_type}"):
    if controller_type == "LQI (Integral Optimal Control)":
        render_mermaid("""
        flowchart LR
            R((Target)) --> Sum1{+}
            Sum1 -->|Error e| Int[∫ Integral]
            Int -->|x_i| K_i[Gain K_i]
            K_i --> Sum2{+}
            P[Plant] -->|State x| K[Gain K]
            K -->|-| Sum2
            Sum2 -->|u| P
        """)
        st.markdown(r"""
        **LQI (Linear Quadratic Integral):** Augments the state matrix with the integral of position error ($\int e dt$). Guarantees zero steady-state error under continuous disturbances by building up control effort the longer the system is off-target.
        """)
    elif controller_type == "PID (Classical)":
        render_mermaid("""
        flowchart LR
            R((Target)) --> Sum{+}
            Sum -->|Error| C[PID]
            C -->|Force| P[Plant]
            P -->|Angle| Y((Output))
            P --->|-| Sum
        """)
        st.markdown(r"""
        **Classical SISO Control**
        PID calculates an error value as the difference between a setpoint and a measured variable.
        """)
    else:
        render_mermaid("""
        flowchart LR
            R((Target)) --> N[Nr Gain]
            N --> Sum{+}
            Sum -->|Force u| P[Plant]
            P -->|Sensors y| Obs[Estimator]
            Obs -->|Est State x̂| K[Gain K]
            K --->|-| Sum
        """)
        if controller_type == "Pole Placement (State-Space)":
            st.markdown(r"""
            **Modern MIMO Control**
            Instead of guessing gains, we specify physical behavior. **$\zeta$ (Zeta)** controls the smoothness (damping), and **$\omega_n$ (Omega_n)** controls the speed. The algorithm calculates the exact feedback matrix $K$ required to force the physics engine to obey those specs.
            """)
        else:
            st.markdown(r"""
            **Optimal MIMO Control (LQR)**
            Instead of specifying exact pole locations, LQR frames control as an optimization problem. It calculates the mathematically optimal feedback matrix ($K$) by balancing tracking performance against actuator limits.
            """)

# Controller Tuning
if controller_type == "PID (Classical)":
    # Default values are now 0.0 to encourage student exploration
    kp = st.sidebar.slider("Kp (Proportional)", -200.0, 200.0, 0.0, 1.0)
    ki = st.sidebar.slider("Ki (Integral)", -50.0, 50.0, 0.0, 0.1)
    kd = st.sidebar.slider("Kd (Derivative)", -100.0, 100.0, 0.0, 0.1)
    controller = PIDController(kp=kp, ki=ki, kd=kd)
    
elif controller_type == "Pole Placement (State-Space)":
    zeta = st.sidebar.slider("Zeta (Damping Ratio)", 0.1, 2.0, 0.7, 0.1)
    wn = st.sidebar.slider("Wn (Natural Frequency)", 1.0, 10.0, 4.0, 0.1)
    controller = StateSpaceController(A=env.A, B=env.B, zeta=zeta, wn=wn)

elif controller_type == "LQR (Optimal Control)":
    st.sidebar.markdown("**Cost Matrix ($Q$) - State Priorities**")
    q_pos = st.sidebar.slider("Position Penalty ($Q_{pos}$)", 0.1, 500.0, 100.0, 1.0)
    q_ang = st.sidebar.slider("Angle Penalty ($Q_{ang}$)", 0.1, 500.0, 10.0, 1.0)
    st.sidebar.markdown("**Cost Matrix ($R$) - Actuator Priority**")
    r_weight = st.sidebar.slider("Motor Penalty ($R$)", 0.01, 50.0, 1.0, 0.01)
    controller = LQRController(A=env.A, B=env.B, q_pos=q_pos, q_ang=q_ang, r_weight=r_weight)

elif controller_type == "LQI (Integral Optimal Control)":
    st.sidebar.markdown("**Cost Matrices ($Q$ & $R$)**")
    q_pos = st.sidebar.slider("Position Penalty ($Q_{pos}$)", 0.1, 500.0, 100.0, 1.0)
    q_ang = st.sidebar.slider("Angle Penalty ($Q_{ang}$)", 0.1, 500.0, 10.0, 1.0)
    q_int = st.sidebar.slider("Integral Penalty ($Q_{int}$)", 0.1, 500.0, 150.0, 1.0)
    r_weight = st.sidebar.slider("Motor Penalty ($R$)", 0.01, 50.0, 1.0, 0.01)
    controller = LQIController(A=env.A, B=env.B, q_pos=q_pos, q_ang=q_ang, q_int=q_int, r_weight=r_weight)

# --- 🎯 PATH PLANNING & SETPOINT ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Path Planning & Setpoint")

with st.sidebar.expander("ℹ️ Theory: Trajectories & Feedforward"):
    st.markdown(r"""
    **Stop Teleporting. Start Driving.**
    A "Step Input" (static target) instantly demands the cart move distance, requiring infinite acceleration. This violently whips the pendulum and saturates the motor.
    
    Enable the **Trajectory Generator** to use a **5th-Order Quintic Polynomial**. 
    1. **Planner:** Smoothly ramps up Position, Velocity, and Acceleration.
    2. **Feedforward (2-DOF):** We proactively push the cart using $F=ma$ based on the planned acceleration, so the feedback controller doesn't have to do all the heavy lifting.
    
    **Trajectory Parameters:**
    * **Move To Position:** The final destination on the track.
    * **Start Move At:** The timestamp to begin moving. Allows the system to perfectly stabilize first.
    * **Move Duration:** The time allowed to complete the journey. A short duration demands massive, violent acceleration spikes. A long duration results in a gentle, energy-efficient S-curve.
    """)

use_trajectory = st.sidebar.checkbox("Enable Trajectory Generator", value=False)

if use_trajectory:
    move_target = st.sidebar.slider("Move To Position (m)", -track_limit, track_limit, 1.5, 0.1)
    move_start = st.sidebar.slider("Start Move At (s)", 0.0, 5.0, 1.0, 0.5)
    move_duration = st.sidebar.slider("Move Duration (s)", 0.5, 5.0, 2.5, 0.1)
    target_p = move_target 
else:
    target_p = st.sidebar.slider("Static Target Position (m)", -track_limit, track_limit, 0.0, 0.1)
    move_target, move_start, move_duration = target_p, 0.0, 0.1

# --- 🔭 STATE ESTIMATION ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔭 State Estimation")
estimator_type = st.sidebar.selectbox("Estimator Architecture", ["None", "Luenberger (Deterministic)", "Kalman Filter (Stochastic)"])

with st.sidebar.expander("ℹ️ Theory: Estimators"):
    st.markdown(r"""
    * **Luenberger:** A deterministic observer that relies on raw error ($y - \hat{y}$). Highly susceptible to sensor noise if tuned too aggressively.
    * **Kalman Filter:** An optimal stochastic estimator. It solves the Riccati equation to balance **Process Mistrust ($Q_v$)** against **Sensor Mistrust ($R_w$)**, creating a smooth estimate even with garbage sensor data.
    """)

if estimator_type == "Luenberger (Deterministic)":
    obs_speed = st.sidebar.slider("Observer Speed Multiplier", 1.0, 10.0, 4.0, 0.5)
elif estimator_type == "Kalman Filter (Stochastic)":
    qv_mult = st.sidebar.slider("Process Mistrust (Qv)", 0.001, 10.0, 0.1, format="%.3f")
    rw_mult = st.sidebar.slider("Sensor Mistrust (Rw)", 0.001, 10.0, 1.0, format="%.3f")

# --- 🌍 Real-World Constraints ---
st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Real-World Constraints")

with st.sidebar.expander("ℹ️ Theory: Physical Limitations"):
    st.markdown(r"""
    **Bridging the Sim-to-Real Gap**
    Pure mathematical controllers assume infinite voltage and perfect telemetry. These toggles introduce the harsh realities of physical engineering.
    
    * **Actuator Saturation:** Physical DC motors have a voltage ceiling. 
    * **Sensor Noise Injection:** Optical encoders are imperfect. High-frequency electrical noise is injected into the measurement vector ($y$).
    * **Disturbances:** Inject impulse forces (gusts) or steady-state continuous forces (wind) to test disturbance rejection and steady-state error.
    """)

use_saturation = st.sidebar.checkbox("Limit Motor Force (Saturation)")
if use_saturation:
    max_force = st.sidebar.slider("Max Force (N)", 1.0, 100.0, 15.0, 1.0)
else:
    max_force = float('inf') 

use_noise = st.sidebar.checkbox("Inject Sensor Noise")
if use_noise:
    noise_std = st.sidebar.slider("Angle Noise Std Dev (deg)", 0.01, 5.0, 0.5, 0.01)
else:
    noise_std = 0.0

st.sidebar.markdown("**Disturbances**")
dist_type = st.sidebar.radio("Wind Profile", ["None", "Impulse Gust (0.1s)", "Continuous Wind (Steady-State)"])
if dist_type != "None":
    dist_mag = st.sidebar.slider("Wind Force (N)", -100.0, 100.0, 10.0, 1.0)
    dist_time = st.sidebar.slider("Start Time (s)", 1.0, 9.0, 2.0, 0.5)
else:
    dist_mag = 0.0
    dist_time = -1.0

# --- 3. Simulation Setup & Observer Initialization ---
dt = 0.02
total_time = 10.0 
steps = int(total_time / dt)

init_theta_rad = np.radians(init_theta_deg)
current_state = np.array([init_p, 0.0, init_theta_rad, 0.0])
history = []
obs_history = []
ref_history = [] 
total_energy = 0.0  

# C Matrix: We ONLY measure position [0] and angle [2]
C = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
use_estimator = estimator_type != "None"

if use_estimator:
    if estimator_type == "Luenberger (Deterministic)":
        if controller_type not in ["PID (Classical)", "LQI (Integral Optimal Control)"]:
            ctrl_poles = controller.poles if hasattr(controller, 'poles') else [-2, -2.1, -2.2, -2.3]
            obs_poles = np.array(ctrl_poles) * obs_speed
        else:
            obs_poles = np.array([-10, -11, -12, -13])
        estimator = LuenbergerObserver(env.A, env.B, C, obs_poles)
        
    elif estimator_type == "Kalman Filter (Stochastic)":
        Q_v = np.eye(4) * qv_mult
        R_w = np.eye(2) * rw_mult
        estimator = KalmanFilter(env.A, env.B, C, Q_v, R_w)

    estimator.reset(current_state + np.array([0.2, 0, 0, 0]))

# Initialize Trajectory Planner
planner = TrajectoryPlanner(p_start=init_p, p_end=move_target, duration=move_duration)

# --- 4. The Closed-Loop Simulation ---
terminated = False
history_u = [0.0]

for step_idx in range(steps):
    t = step_idx * dt
    history.append(current_state)
    p = current_state[0]
    
    # 1. Trajectory Planning & Feedforward
    if use_trajectory:
        p_ref, v_ref, a_ref = planner.get_state(t - move_start)
        # 2-DOF Feedforward Force: F = ma
        u_ff = (m_c + m_p) * a_ref 
    else:
        p_ref, v_ref, u_ff = target_p, 0.0, 0.0
        
    ref_history.append(p_ref)
    
    # 2. Measurement (Sensors)
    y = C @ current_state
    if use_noise:
        y[1] += np.radians(np.random.normal(0, noise_std))
        
    # 3. Estimation 
    if use_estimator:
        u_prev = history_u[-1]
        x_hat = estimator.update(u_prev, y, dt)
        obs_history.append(x_hat.copy())
        feedback_state = x_hat
    else:
        obs_history.append(current_state)
        feedback_state = current_state.copy()
        feedback_state[2] = y[1] # "God Mode" fix
        
    # --- Check bounds AFTER appending to obs_history ---
    if abs(p) > track_limit:
        terminated = True
        break 
        
    # 4. Control (Feedback)
    if controller_type == "PID (Classical)":
        u_fb = controller.compute(setpoint=p_ref, current_value=y[1], dt=dt)
    elif controller_type == "LQI (Integral Optimal Control)":
        u_fb = controller.compute(setpoint=p_ref, current_state=feedback_state, dt=dt) 
    else:
        u_fb = controller.compute(setpoint=p_ref, current_state=feedback_state)
    
    # Combine Feedback (u_fb) and Feedforward (u_ff)
    u = u_fb + u_ff
    
    # 5. Plant Actuation
    if use_saturation:
        u = np.clip(u, -max_force, max_force)
        
    total_energy += (u**2) * dt
    history_u.append(u)
        
    # Apply Wind Disturbance Profile
    u_effective = u
    if dist_type == "Impulse Gust (0.1s)" and (dist_time <= t <= dist_time + 0.1):
        u_effective += dist_mag
    elif dist_type == "Continuous Wind (Steady-State)" and (t >= dist_time):
        u_effective += dist_mag
        
    current_state = env.step(current_state, u=u_effective, dt=dt)

# --- 5. Telemetry UI ---
final_p = history[-1][0]
final_theta_deg = np.degrees(history[-1][2])

pos_stable = abs(final_p - target_p) <= 0.05 and not terminated
angle_stable = abs(final_theta_deg) <= 0.5

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Position Status", "STABLE 🟢" if pos_stable else ("OUT OF BOUNDS 🔴" if terminated else "DRIFT 🔴"))
with col2:
    st.metric("Final Position", f"{final_p:.3f} m")
with col3:
    st.metric("Angle Status", "STABLE (UPRIGHT) 🟢" if angle_stable else "UNSTABLE 🔴")
with col4:
    st.metric("Control Effort", f"{total_energy:.1f} J")
with col5:
    estimator_label = "OFF ⚪"
    if use_estimator: estimator_label = "KALMAN 🔭" if "Kalman" in estimator_type else "LUENBERGER 🔭"
    st.metric("Estimator", estimator_label)

# --- 6. Professional Plotly Rendering (Animation) ---
cart_width = 0.4
cart_height = 0.15
wheel_radius = 0.05
track_y = -0.5 
wheel_y_bottom = track_y
wheel_y_top = wheel_y_bottom + (wheel_radius * 2)
cart_y_bottom = wheel_y_top
cart_y_top = cart_y_bottom + cart_height
pivot_y = cart_y_top

track_outer, track_inner = "#334155", "#64748b"
cart_outer, cart_inner = "#a16207", "#eab308"
wheel_tire, wheel_rim = "#020617", "#cbd5e1"
pendulum_color, joint_color, center_marker_color = "#f43f5e", "#f8fafc", "#f43f5e"

def get_shapes(p):
    return [
        dict(type="line", x0=-track_limit-0.2, y0=track_y-0.01, x1=track_limit+0.2, y1=track_y-0.01, line=dict(color=track_outer, width=10), layer="below"),
        dict(type="line", x0=-track_limit-0.2, y0=track_y, x1=track_limit+0.2, y1=track_y, line=dict(color=track_inner, width=2), layer="below"),
        dict(type="line", x0=target_p, y0=track_y-0.03, x1=target_p, y1=track_y+0.03, line=dict(color=center_marker_color, width=4), layer="below"),
        dict(type="circle", x0=p - 0.15 - wheel_radius, y0=wheel_y_bottom, x1=p - 0.15 + wheel_radius, y1=wheel_y_top, fillcolor=wheel_tire, line_color="#000", layer="below"),
        dict(type="circle", x0=p - 0.15 - wheel_radius*0.4, y0=wheel_y_bottom + wheel_radius*0.6, x1=p - 0.15 + wheel_radius*0.4, y1=wheel_y_top - wheel_radius*0.6, fillcolor=wheel_rim, line_width=0, layer="below"),
        dict(type="circle", x0=p + 0.15 - wheel_radius, y0=wheel_y_bottom, x1=p + 0.15 + wheel_radius, y1=wheel_y_top, fillcolor=wheel_tire, line_color="#000", layer="below"),
        dict(type="circle", x0=p + 0.15 - wheel_radius*0.4, y0=wheel_y_bottom + wheel_radius*0.6, x1=p + 0.15 + wheel_radius*0.4, y1=wheel_y_top - wheel_radius*0.6, fillcolor=wheel_rim, line_width=0, layer="below"),
        dict(type="rect", x0=p - (cart_width / 2.0), y0=cart_y_bottom, x1=p + (cart_width / 2.0), y1=cart_y_top, fillcolor=cart_outer, line_color="#000", line_width=2, layer="below"),
        dict(type="rect", x0=p - (cart_width / 2.0) + 0.03, y0=cart_y_bottom + 0.03, x1=p + (cart_width / 2.0) - 0.03, y1=cart_y_top - 0.03, fillcolor=cart_inner, line_width=0, layer="below")
    ]

fig = go.Figure()
visual_l = env.l * 1.5  

p0 = history[0][0]
theta0 = history[0][2]
tip_x0 = p0 + visual_l * np.sin(theta0)
tip_y0 = pivot_y + visual_l * np.cos(theta0)

# Add Shadow Pendulum Trace (Observer Estimate)
if use_estimator:
    p0_hat = obs_history[0][0]
    theta0_hat = obs_history[0][2]
    tip_x0_hat = p0_hat + visual_l * np.sin(theta0_hat)
    tip_y0_hat = pivot_y + visual_l * np.cos(theta0_hat)
    fig.add_trace(go.Scatter(x=[p0_hat, tip_x0_hat], y=[pivot_y, tip_y0_hat], mode="lines", 
                             line=dict(color="#38bdf8", width=4, dash="dot"), name="Estimate", hoverinfo="skip"))

# Add True Plant Trace
fig.add_trace(go.Scatter(x=[p0, tip_x0], y=[pivot_y, tip_y0], mode="lines+markers", 
                         line=dict(color=pendulum_color, width=8),
                         marker=dict(color=[joint_color, joint_color], size=[16, 16], line=dict(color="#000", width=2)), 
                         name="True Plant", hoverinfo="skip"))

fig.update_layout(shapes=get_shapes(p0))

frames = []
for i, state in enumerate(history):
    p, theta = state[0], state[2]
    tip_x = p + visual_l * np.sin(theta)
    tip_y = pivot_y + visual_l * np.cos(theta)
    
    frame_data = []
    
    if use_estimator:
        p_hat, theta_hat = obs_history[i][0], obs_history[i][2]
        tip_x_hat = p_hat + visual_l * np.sin(theta_hat)
        tip_y_hat = pivot_y + visual_l * np.cos(theta_hat)
        frame_data.append(go.Scatter(x=[p_hat, tip_x_hat], y=[pivot_y, tip_y_hat]))
        
    frame_data.append(go.Scatter(x=[p, tip_x], y=[pivot_y, tip_y]))
    
    frames.append(go.Frame(data=frame_data, layout=go.Layout(shapes=get_shapes(p)), name=str(i)))

fig.frames = frames

fig.update_layout(
    xaxis=dict(range=[-track_limit-0.2, track_limit+0.2], autorange=False, visible=False), 
    yaxis=dict(range=[-1.0, 2.5], autorange=False, scaleanchor="x", scaleratio=1, visible=False),
    width=1200, height=500, showlegend=True,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc")),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0),
    updatemenus=[dict(
        type="buttons", showactive=False,
        y=1.1, x=0.5, xanchor="center", yanchor="bottom", direction="left", 
        buttons=[
            dict(label="▶ Play Simulation", method="animate", args=[None, {"frame": {"duration": int(dt * 1000), "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="⏸ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]),
            dict(label="⏮ Reset", method="animate", args=[["0"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}])
        ]
    )]
)

st.plotly_chart(fig, use_container_width=True)

# --- 7. Data Analytics Dashboard (Static Graphs) ---
st.markdown("<hr style='border-color: #1e293b; margin-top: 0px; margin-bottom: 10px;'>", unsafe_allow_html=True)
st.markdown("<h3 style='color: #94a3b8; font-size: 1.2rem;'>📊 Diagnostic Telemetry</h3>", unsafe_allow_html=True)

time_arr = np.arange(len(history)) * dt

# Convergence Error Plot (If Estimator is ON)
if use_estimator:
    err_p = [history[i][0] - obs_history[i][0] for i in range(len(history))]
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(x=time_arr, y=err_p, mode='lines', line=dict(color="#38bdf8", width=2)))
    fig_err.update_layout(
        height=250,
        title="Estimator Convergence: Position Error (True - Estimated)",
        xaxis_title="Time (s)", yaxis_title="Error (m)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        xaxis=dict(showgrid=True, gridcolor="#1e293b"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_err, use_container_width=True)

# Standard Position & Angle Plots
p_arr = [state[0] for state in history]
theta_arr = [np.degrees(state[2]) for state in history]

g_col1, g_col2 = st.columns(2)

fig_pos = go.Figure()
# Add the dashed path planner line
if use_trajectory:
    fig_pos.add_trace(go.Scatter(x=time_arr, y=ref_history, mode='lines', line=dict(color="#94a3b8", width=2, dash="dash"), name="Planned Path"))
fig_pos.add_trace(go.Scatter(x=time_arr, y=p_arr, mode='lines', line=dict(color="#38bdf8", width=3), name="Actual Position"))
fig_pos.add_hline(y=target_p, line_dash="dash", line_color="#f43f5e")

fig_pos.update_layout(
    height=300,
    title="Cart Position Tracking",
    xaxis_title="Time (s)", yaxis_title="Position (m)",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f8fafc"),
    xaxis=dict(showgrid=True, gridcolor="#1e293b"),
    yaxis=dict(showgrid=True, gridcolor="#1e293b", range=[-track_limit-0.2, track_limit+0.2]),
    margin=dict(l=0, r=0, t=30, b=0),
    showlegend=True
)

fig_angle = go.Figure()
fig_angle.add_trace(go.Scatter(x=time_arr, y=theta_arr, mode='lines', line=dict(color="#f43f5e", width=3), showlegend=False))
fig_angle.add_hline(y=0.0, line_dash="dash", line_color="#38bdf8")
fig_angle.update_layout(
    height=300,
    title="Pendulum Angle over Time",
    xaxis_title="Time (s)", yaxis_title="Angle (Degrees)",
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f8fafc"),
    xaxis=dict(showgrid=True, gridcolor="#1e293b"),
    yaxis=dict(showgrid=True, gridcolor="#1e293b"),
    margin=dict(l=0, r=0, t=30, b=0)
)

with g_col1:
    st.plotly_chart(fig_pos, use_container_width=True)
with g_col2:
    st.plotly_chart(fig_angle, use_container_width=True)