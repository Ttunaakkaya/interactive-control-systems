"""
experiment_phase3.py — Residual model learning showcase.

Setup:
  TRUE plant    : m_p = 0.10, with rail friction        (reality)
  NOMINAL prior : m_p = 0.15 (+50% wrong), frictionless (what the controller thinks)

Learn delta_f = x+ - f_bar(x,u) with 2 GPs from ~36s of closed-loop data,
then show (1) one-step and (2) 40-step prediction: prior vs GP-corrected.

Run from the repo root:   python experiment_phase3.py
Outputs: phase3_rollout_comparison.html, phase3_residual_parity.html
Data   : saved to DATA_PATH for reuse by the GP-MPC experiment.
"""
import warnings
import numpy as np
import scipy.linalg as sla
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.exceptions import ConvergenceWarning

from plant import CartPolePlant
from learning import collect_rollouts, ResidualGP, RESIDUAL_DIMS

# A railed RBF lengthscale just means "this input barely affects the residual" — benign.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_PATH = "data/phase3_rollouts.npz"
DT = 0.02

# --------------------------------------------------------------------------- #
# 1. True plant vs wrong nominal prior
# --------------------------------------------------------------------------- #
true_env = CartPolePlant(m_c=1.0, m_p=0.10, l=0.5)
true_env.set_friction(cart_frictionloss=0.5, cart_damping=2.0)   # unknown to prior
nominal_env = CartPolePlant(m_c=1.0, m_p=0.15, l=0.5)            # +50% mass, no friction

# LQR synthesized on the WRONG model (as it would be in reality)
Q = np.diag([100., 1., 10., 1.]); R = np.array([[1.0]])
P = sla.solve_continuous_are(nominal_env.A, nominal_env.B, Q, R)
K = np.linalg.solve(R, nominal_env.B.T @ P)

# --------------------------------------------------------------------------- #
# 2. Collect data from the TRUE plant, fit the residual GP
# --------------------------------------------------------------------------- #
Z, Rres, (X, U, Xn) = collect_rollouts(true_env, nominal_env, K, seed=0)
import os; os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
np.savez(DATA_PATH, X=X, U=U, Xn=Xn, Z=Z, R=Rres)
print(f"collected {len(Z)} transitions (~{len(Z)*DT:.0f}s sim time) -> {DATA_PATH}")

n_tr = int(0.8 * len(Z))
gp = ResidualGP(max_points=300).fit(Z[:n_tr], Rres[:n_tr])
print("residual GP fitted (2 GPs on dims [cart_vel, pole_angvel])")

# --------------------------------------------------------------------------- #
# 3. One-step prediction: prior vs corrected (held-out data)
# --------------------------------------------------------------------------- #
Xte, Ute, Xnte = X[n_tr:], U[n_tr:], Xn[n_tr:]
mu_te, _ = gp.predict(Xte, Ute)
Fbar_te = np.array([np.array(nominal_env.F(Xte[i], [Ute[i]])).flatten()
                    for i in range(len(Xte))])
e_prior = Xnte[:, RESIDUAL_DIMS] - Fbar_te[:, RESIDUAL_DIMS]
e_corr  = e_prior - mu_te
rmse_p = np.sqrt((e_prior**2).mean(0)); rmse_c = np.sqrt((e_corr**2).mean(0))
for i, name in enumerate(["cart_vel", "pole_angvel"]):
    print(f"one-step RMSE {name:12s}: prior={rmse_p[i]:.5f}  corrected={rmse_c[i]:.5f}  ({rmse_p[i]/rmse_c[i]:.0f}x)")

# --------------------------------------------------------------------------- #
# 4. 40-step open-loop rollout: truth vs prior vs corrected
# --------------------------------------------------------------------------- #
H = 40
rng = np.random.default_rng(7)
x0 = np.array([0.2, 0.0, np.radians(8), 0.0])
x_true, x_pri, x_cor = x0.copy(), x0.copy(), x0.copy()
T_true, T_pri, T_cor, u_seq = [x0.copy()], [x0.copy()], [x0.copy()], []
for k in range(H):
    u = float(np.clip(float((-K @ x_true)[0]) + rng.normal(0, 1.5), -15, 15))
    u_seq.append(u)
    x_true = true_env.step(x_true, u, DT)
    x_pri  = np.array(nominal_env.F(x_pri, [u])).flatten()
    x_cor  = gp.corrected_step(nominal_env, x_cor, u)
    T_true.append(x_true.copy()); T_pri.append(x_pri.copy()); T_cor.append(x_cor.copy())
T_true, T_pri, T_cor = map(np.array, (T_true, T_pri, T_cor))
print(f"40-step final-state error: prior={np.linalg.norm(T_pri[-1]-T_true[-1]):.4f}  "
      f"corrected={np.linalg.norm(T_cor[-1]-T_true[-1]):.4f}")

t = np.arange(H + 1) * DT
names = ["cart pos (m)", "cart vel (m/s)", "pole angle (deg)", "pole angvel (deg/s)"]
scale = [1, 1, 180/np.pi, 180/np.pi]
fig = make_subplots(rows=2, cols=2, subplot_titles=names)
for i in range(4):
    r, c = i // 2 + 1, i % 2 + 1
    fig.add_trace(go.Scatter(x=t, y=T_true[:, i]*scale[i], name="truth",
                             line=dict(color="black"), showlegend=(i == 0)), r, c)
    fig.add_trace(go.Scatter(x=t, y=T_pri[:, i]*scale[i], name="wrong prior f̄",
                             line=dict(color="crimson", dash="dash"), showlegend=(i == 0)), r, c)
    fig.add_trace(go.Scatter(x=t, y=T_cor[:, i]*scale[i], name="f̄ + GP residual",
                             line=dict(color="seagreen"), showlegend=(i == 0)), r, c)
fig.update_layout(title="40-step open-loop prediction: wrong prior vs learned model",
                  height=600)
fig.write_html("phase3_rollout_comparison.html")

# --------------------------------------------------------------------------- #
# 5. Residual parity scatter (does the GP predict the actual residual?)
# --------------------------------------------------------------------------- #
fig2 = make_subplots(rows=1, cols=2,
                     subplot_titles=["cart_vel residual", "pole_angvel residual"])
for i in range(2):
    a = e_prior[:, i]; b = mu_te[:, i]
    lim = [min(a.min(), b.min()), max(a.max(), b.max())]
    fig2.add_trace(go.Scatter(x=a, y=b, mode="markers",
                              marker=dict(size=4, opacity=0.5), showlegend=False), 1, i+1)
    fig2.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                              line=dict(color="black", dash="dot"), showlegend=False), 1, i+1)
    fig2.update_xaxes(title="actual residual", row=1, col=i+1)
    fig2.update_yaxes(title="GP predicted", row=1, col=i+1)
fig2.update_layout(title="GP residual predictions vs ground truth (held-out)", height=420)
fig2.write_html("phase3_residual_parity.html")
print("plots -> phase3_rollout_comparison.html, phase3_residual_parity.html")
