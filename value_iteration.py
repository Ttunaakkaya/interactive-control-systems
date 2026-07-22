"""
value_iteration.py — Dynamic Programming on the discretized cart-pole
(Book Ch. 1.2: the DP algorithm  ==  Book Ch. 7.3.1: model-based RL)

Discretize the state box around upright, evaluate the ONE symbolic model F on
every (state, action) pair, and run discounted value iteration:

    V_{i+1}(s) = min_a [ c(s,a) + gamma * V_i(F(s,a)) ]

Next states fall between grid points -> multilinear interpolation of V,
implemented as one sparse matrix per action (16 corner weights per row),
so each Bellman sweep is 7 sparse mat-vecs. Same Q, R as the LQR so the
comparison is apples-to-apples.
"""
import numpy as np
import casadi as ca
from scipy import sparse


def make_grid(bounds, n_per_dim):
    axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(bounds, n_per_dim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    S = np.stack([m.ravel() for m in mesh], axis=1)          # (Ns, 4)
    return axes, S


def interp_matrix(axes, Xn):
    """Sparse (Ns_query x Ns_grid) multilinear interpolation operator in 4D."""
    n_dims = len(axes)
    sizes = [len(a) for a in axes]
    strides = np.cumprod([1] + sizes[::-1][:-1])[::-1]        # row-major strides
    idx_lo, w_hi = [], []
    for d, ax in enumerate(axes):
        x = np.clip(Xn[:, d], ax[0], ax[-1])
        i = np.clip(np.searchsorted(ax, x, side="right") - 1, 0, sizes[d] - 2)
        t = (x - ax[i]) / (ax[i + 1] - ax[i])
        idx_lo.append(i); w_hi.append(t)
    Nq = Xn.shape[0]
    rows, cols, vals = [], [], []
    for corner in range(2 ** n_dims):                          # 16 corners
        w = np.ones(Nq); flat = np.zeros(Nq, dtype=np.int64)
        for d in range(n_dims):
            hi = (corner >> d) & 1
            w = w * (w_hi[d] if hi else (1.0 - w_hi[d]))
            flat = flat + (idx_lo[d] + hi) * strides[d]
        rows.append(np.arange(Nq)); cols.append(flat); vals.append(w)
    M = sparse.csr_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(Nq, int(np.prod(sizes))))
    return M


def build_dp(plant, bounds, n_per_dim, actions, Q, R,
             oob_penalty=200.0, gamma=0.999, dt=0.02):
    axes, S = make_grid(bounds, n_per_dim)
    Ns = S.shape[0]
    Fmap = plant.F.map(Ns)
    P_a, C_a = [], []
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    for u in actions:
        Xn = np.array(Fmap(S.T, np.full((1, Ns), u))).T        # (Ns,4)
        oob = np.any((Xn < lo) | (Xn > hi), axis=1)
        P_a.append(interp_matrix(axes, Xn))
        c = (np.einsum("ni,ij,nj->n", S, Q, S) + R[0, 0] * u * u) * dt
        c = c + oob_penalty * oob                              # leaving the box hurts
        C_a.append(c)
    return axes, S, P_a, C_a, gamma


def value_iteration(P_a, C_a, gamma, iters=12000, tol=2e-4):
    Ns = C_a[0].shape[0]
    V = np.zeros(Ns)
    for it in range(iters):
        Qsa = np.stack([C_a[a] + gamma * (P_a[a] @ V) for a in range(len(C_a))], axis=1)
        V_new = Qsa.min(axis=1)
        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < tol:
            break
    pol = Qsa.argmin(axis=1)
    return V, pol, it, delta


class VIPolicy:
    """
    Online greedy policy w.r.t. the converged value function (Book 7.3.1):
        u*(x) = argmin_a [ c(x,a) + gamma * V(F(x,a)) ]
    evaluated at the ACTUAL continuous state with V multilinearly interpolated.
    This removes state-quantization error from the policy: the grid only
    discretizes V, not the decision.
    """
    def __init__(self, axes, actions, V, plant, Q, R, gamma=0.999, dt=0.02):
        self.axes = axes; self.actions = np.asarray(actions, float)
        self.V = V; self.plant = plant; self.Q = Q; self.R = R
        self.gamma = gamma; self.dt = dt
        na = len(self.actions)
        self._Fbatch = plant.F.map(na)

    def _V_at(self, X):
        M = interp_matrix(self.axes, np.atleast_2d(X))
        return M @ self.V

    def compute(self, setpoint, current_state):
        x = np.asarray(current_state, float)
        na = len(self.actions)
        Xn = np.array(self._Fbatch(np.tile(x.reshape(-1, 1), (1, na)),
                                   self.actions.reshape(1, -1))).T   # (na,4)
        c = (float(x @ self.Q @ x) + self.R[0, 0] * self.actions ** 2) * self.dt
        q = c + self.gamma * self._V_at(Xn)
        return float(self.actions[int(np.argmin(q))])


def sinh_axis(lim, n, sharp=2.2):
    """Nonuniform axis, dense near zero (sinh-spaced). Crucial: the value
    function has all its curvature near the origin; uniform grids waste
    resolution at the rim and chatter near the target."""
    t = np.linspace(-1, 1, n)
    return lim * np.sinh(sharp * t) / np.sinh(sharp)


def build_dp_axes(plant, axes, actions, Q, R, oob_penalty=200.0,
                  gamma=0.999, dt=0.02):
    """build_dp variant taking explicit (possibly nonuniform) axes."""
    mesh = np.meshgrid(*axes, indexing="ij")
    S = np.stack([m.ravel() for m in mesh], axis=1)
    lo = np.array([a[0] for a in axes]); hi = np.array([a[-1] for a in axes])
    Ns = S.shape[0]
    Fmap = plant.F.map(Ns)
    P_a, C_a = [], []
    for u in actions:
        Xn = np.array(Fmap(S.T, np.full((1, Ns), u))).T
        oob = np.any((Xn < lo) | (Xn > hi), axis=1)
        P_a.append(interp_matrix(axes, Xn))
        c = (np.einsum("ni,ij,nj->n", S, Q, S) + R[0, 0] * u * u) * dt
        C_a.append(c + oob_penalty * oob)
    return axes, S, P_a, C_a, gamma


def value_iteration_policy_stable(P_a, C_a, gamma, V0=None,
                                  max_sweeps=4000, stable_for=30):
    """Sweep until the greedy policy is unchanged for `stable_for` consecutive
    sweeps. With gamma near 1 the VALUE keeps creeping by a near-constant
    offset long after the POLICY (the argmin) has converged; policy stability
    is the honest stopping criterion for control purposes."""
    Ns = C_a[0].shape[0]
    V = np.zeros(Ns) if V0 is None else V0.copy()
    pol_prev, stable = None, 0
    for it in range(max_sweeps):
        Qsa = np.stack([C_a[a] + gamma * (P_a[a] @ V) for a in range(len(C_a))], axis=1)
        V = Qsa.min(axis=1)
        pol = Qsa.argmin(axis=1)
        stable = stable + 1 if (pol_prev is not None and np.array_equal(pol, pol_prev)) else 0
        pol_prev = pol
        if stable >= stable_for:
            break
    return V, pol, it + 1, stable