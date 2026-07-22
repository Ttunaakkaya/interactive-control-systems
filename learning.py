"""
learning.py — Phase 3: residual model learning (Book Ch. 6, eq. 6.7 / 6.10-6.12)

True dynamics:      x+ = f(x, u)                     (unknown to the controller)
Nominal prior:      x+ = f_bar(x, u)                 (wrong mass, no friction)
Learn the residual: delta_f(x, u) = x+ - f_bar(x,u)  (a GP per affected state)

Only the two velocity states carry model error (positions integrate exactly),
so we learn 2 GPs on dims [1, 3] = [cart_vel, pole_angvel].
GP input features z = (cart_vel, theta, theta_dot, u): absolute cart position
cannot affect the physics residual, so excluding it saves data.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

RESIDUAL_DIMS = [1, 3]                  # cart_vel, pole_angvel


def features(x, u):
    """GP input: (cart_vel, theta, theta_dot, u). Drops absolute position."""
    x = np.atleast_2d(x); u = np.atleast_2d(u).reshape(len(x), -1)
    return np.hstack([x[:, [1, 2, 3]], u])


def collect_rollouts(true_env, nominal_env, K_lqr, n_rollouts=12, steps=150,
                     dt=0.02, excite_std=2.0, seed=0):
    """
    Closed-loop LQR rollouts from random initial states + exploration noise.
    Returns Z (features), R (residuals on RESIDUAL_DIMS), plus raw (X, U, Xn).
    """
    rng = np.random.default_rng(seed)
    X, U, Xn = [], [], []
    for _ in range(n_rollouts):
        x = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3),
                      rng.uniform(-0.25, 0.25), rng.uniform(-0.3, 0.3)])
        for _ in range(steps):
            u = float((-K_lqr @ x)[0]) + rng.normal(0.0, excite_std)
            u = float(np.clip(u, -15, 15))
            xn = true_env.step(x, u, dt)          # data comes from the TRUE plant
            X.append(x.copy()); U.append(u); Xn.append(xn.copy())
            x = xn
            if abs(np.degrees(x[2])) > 45:        # fell too far -> restart
                break
    X, U, Xn = np.array(X), np.array(U), np.array(Xn)
    # residual against the WRONG nominal one-step map
    Fbar = np.array([np.array(nominal_env.F(X[i], [U[i]])).flatten()
                     for i in range(len(X))])
    R = (Xn - Fbar)[:, RESIDUAL_DIMS]
    Z = features(X, U)
    return Z, R, (X, U, Xn)


class ResidualGP:
    """One exact GP per residual dimension (RBF-ARD + white noise)."""
    def __init__(self, max_points=300, seed=0):
        self.max_points = max_points
        self.seed = seed
        self.gps = []
        self.z_mean = None; self.z_std = None

    def fit(self, Z, R):
        rng = np.random.default_rng(self.seed)
        if len(Z) > self.max_points:                     # GP is O(n^3): subsample
            idx = rng.choice(len(Z), self.max_points, replace=False)
            Z, R = Z[idx], R[idx]
        self.z_mean, self.z_std = Z.mean(0), Z.std(0) + 1e-8
        Zn = (Z - self.z_mean) / self.z_std
        self.gps = []
        for d in range(R.shape[1]):
            kern = (ConstantKernel(1.0, (1e-3, 1e3))
                    * RBF(np.ones(Z.shape[1]), (1e-2, 1e4))
                    + WhiteKernel(1e-4, (1e-8, 1e0)))
            gp = GaussianProcessRegressor(kernel=kern, normalize_y=True,
                                          n_restarts_optimizer=2,
                                          random_state=self.seed)
            gp.fit(Zn, R[:, d])
            self.gps.append(gp)
        return self

    def predict(self, x, u):
        """mean, std of the residual on RESIDUAL_DIMS at (x, u)."""
        z = (features(x, u) - self.z_mean) / self.z_std
        mus, sds = [], []
        for gp in self.gps:
            m, s = gp.predict(z, return_std=True)
            mus.append(m); sds.append(s)
        return np.stack(mus, -1), np.stack(sds, -1)

    def corrected_step(self, nominal_env, x, u):
        """f_bar(x,u) + learned residual  (the Ch.6 corrected model, eq. 6.70)."""
        xn = np.array(nominal_env.F(x, [u])).flatten()
        mu, _ = self.predict(x, [u])
        xn[RESIDUAL_DIMS] += mu.flatten()
        return xn