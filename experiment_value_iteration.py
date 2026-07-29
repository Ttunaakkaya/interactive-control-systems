"""
experiment_value_iteration.py — Dynamic Programming rediscovers LQR

Discretize the cart-pole around upright (nonuniform sinh grids, dense near
the origin), run value iteration on the symbolic model F, and compare the
resulting policy against the analytic LQR:
  (1) near the origin the DP value function matches LQR's quadratic x'Px,
  (2) the greedy DP policy stabilizes the plant like LQR does,
  (3) the remaining cost gap is the discretization price and motivates
      analytic solutions or function approximation on larger state spaces.

The LQR quadratic is used only to WARM-START V (a pure speed-up: value
iteration is a gamma-contraction, so the fixed point is unique regardless).

Run from the repo root:  python experiment_value_iteration.py
Outputs: phase_dp_value_slice.html, phase_dp_policy_slice.html
Data   : DATA_PATH (converged V for reuse, e.g. as an RL baseline later)
"""
import os
import numpy as np
import scipy.linalg as sla
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plant import CartPolePlant
from value_iteration import (sinh_axis, build_dp_axes,
                             value_iteration_policy_stable, VIPolicy)

DATA_PATH = "data/dp_value_function.npz"
DT = 0.02

env = CartPolePlant(1.0, 0.1, 0.5)
Q = np.diag([10.0, 1.0, 50.0, 1.0]); R = np.array([[0.1]])

axes = [sinh_axis(1.6, 15), sinh_axis(3.0, 13),
        sinh_axis(0.45, 15), sinh_axis(3.0, 15)]
actions = sinh_axis(15.0, 15)
axes, S, P_a, C_a, gamma = build_dp_axes(env, axes, actions, Q, R, dt=DT)
print(f"DP model: {S.shape[0]} states x {len(actions)} actions")

P = sla.solve_continuous_are(env.A, env.B, Q, R)
K = np.linalg.solve(R, env.B.T @ P)
V0 = np.einsum("ni,ij,nj->n", S, P, S)              # warm start only
V, pol, sweeps, stable = value_iteration_policy_stable(P_a, C_a, gamma, V0=V0)
print(f"VI: {sweeps} sweeps, policy stable for {stable} consecutive sweeps")

os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
np.savez(DATA_PATH, V=V, S=S, actions=actions,
         axes=np.array(axes, dtype=object), gamma=gamma, allow_pickle=True)

# ---- closed-loop comparison ------------------------------------------------
vi = VIPolicy(axes, actions, V, env, Q, R, gamma)
def rollout(fn, x0, steps=400):
    x = x0.copy(); J = 0.0
    for _ in range(steps):
        u = fn(x); J += float(x @ Q @ x + R[0, 0] * u * u) * DT
        x = env.step(x, u, DT)
    return x, J
print(f"{'tilt':>5} | {'DP/VI J':>8} {'LQR J':>7} {'ratio':>6} | DP final state")
for tilt in [8, 12, 16]:
    x0 = np.array([0.3, 0.0, np.radians(tilt), 0.0])
    xv, Jv = rollout(lambda x: vi.compute(0.0, x), x0)
    xl, Jl = rollout(lambda x: float(np.clip(float((-K @ x)[0]), -15, 15)), x0)
    print(f"{tilt:4d}° | {Jv:8.2f} {Jl:7.2f} {Jv/Jl:5.2f}x | "
          f"|p|={abs(xv[0]):.3f} m, |th|={np.degrees(abs(xv[2])):.2f}°")

near = np.all(np.abs(S) < np.array([0.8, 1.5, 0.22, 1.5]), axis=1)
V_lqr = np.einsum("ni,ij,nj->n", S, P, S)
print(f"corr(V_DP, x'Px) near origin: {np.corrcoef(V[near], V_lqr[near])[0,1]:.4f}")

# ---- figures: (pos, theta) slice at vel = th_dot = 0 -----------------------
sizes = [len(a) for a in axes]
Vg = V.reshape(sizes)
iv, iw = sizes[1] // 2, sizes[3] // 2          # center indices (= exactly 0)
V_slice = Vg[:, iv, :, iw]                     # (pos, theta)
Vl_slice = V_lqr.reshape(sizes)[:, iv, :, iw]
fig = make_subplots(rows=1, cols=2, subplot_titles=[
    "DP value function V(pos, θ)", "LQR quadratic x'Px (analytic)"])
for col, Zs in [(1, V_slice), (2, Vl_slice)]:
    fig.add_trace(go.Heatmap(x=np.degrees(axes[2]), y=axes[0], z=Zs,
                             colorscale="Viridis", showscale=(col == 2)), 1, col)
    fig.update_xaxes(title="pole angle (deg)", row=1, col=col)
    fig.update_yaxes(title="cart pos (m)", row=1, col=col)
fig.update_layout(title="Dynamic programming rediscovers the LQR value function "
                        "(slice at zero velocities)", height=460)
fig.write_html("phase_dp_value_slice.html")

pol_g = pol.reshape(sizes)
U_dp = actions[pol_g[:, iv, :, iw]]
PP, TT = np.meshgrid(axes[0], axes[2], indexing="ij")
U_lqr = np.clip(-(K[0, 0] * PP + K[0, 2] * TT), -15, 15)
fig2 = make_subplots(rows=1, cols=2, subplot_titles=[
    "DP greedy policy u*(pos, θ)", "LQR policy u = -Kx (clipped)"])
for col, Zs in [(1, U_dp), (2, U_lqr)]:
    fig2.add_trace(go.Heatmap(x=np.degrees(axes[2]), y=axes[0], z=Zs,
                              colorscale="RdBu", zmid=0, showscale=(col == 2)), 1, col)
    fig2.update_xaxes(title="pole angle (deg)", row=1, col=col)
    fig2.update_yaxes(title="cart pos (m)", row=1, col=col)
fig2.update_layout(title="The DP policy recovers LQR's structure "
                         "(slice at zero velocities)", height=460)
fig2.write_html("phase_dp_policy_slice.html")
print("plots -> phase_dp_value_slice.html, phase_dp_policy_slice.html")
