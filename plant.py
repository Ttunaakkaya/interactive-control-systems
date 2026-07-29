import numpy as np
import casadi as ca


# ============================================================================ #
#  Symbolic model — the SINGLE source of truth                                  #
# ============================================================================ #
def build_symbolic_cartpole(m_c, m_p, l, g, dt=0.02):
    """
    Encode the cart-pole physics ONCE, symbolically (CasADi). Everything the
    controllers need is then derived from this one expression:

        f : xdot = f(x, u)        continuous nominal dynamics  (this is f̄)
        A : ∂f/∂x  ,  B : ∂f/∂u   exact analytic Jacobians (no hand derivation)
        F : x_next = F(x, u)      discrete RK4 one-step map  (feeds MPC / iLQR / GP-MPC)

    Why this matters
    ----------------
    1. The linear model A, B used for LQR / pole-placement / LQI is now the
       EXACT Jacobian of the nonlinear simulator, so the two can never disagree.
       (The previous hand-typed A, B used (m_c + m_p) where the point-mass model
        requires m_c in the denominator — a ~10% mismatch, fixed by construction.)
    2. f and F are symbolic and differentiable, which is exactly what the
       optimisation-based controllers in the later phases (MPC, iLQR, GP-MPC)
       require — a NumPy function returning numbers cannot be handed to an NLP.

    This mirrors the design of the Schoellig lab's `safe-control-gym`, which
    represents its a-priori dynamics symbolically (CasADi) for these same reasons.

    State:  x = [cart_pos, cart_vel, pole_angle, pole_angvel]   (pole_angle = 0 -> upright)
    Input:  u = horizontal force on the cart (N)
    """
    x = ca.SX.sym("x", 4)
    u = ca.SX.sym("u", 1)
    vel, theta, theta_dot, force = x[1], x[2], x[3], u[0]
    sin_th, cos_th = ca.sin(theta), ca.cos(theta)

    # Point-mass cart-pole, full nonlinear (no small-angle approximation).
    den = m_c + m_p * sin_th**2                                   # always > 0
    x_ddot = (force + m_p*l*theta_dot**2*sin_th
              - m_p*g*sin_th*cos_th) / den
    theta_ddot = ((m_c + m_p)*g*sin_th
                  - cos_th*(force + m_p*l*theta_dot**2*sin_th)) / (l*den)
    xdot = ca.vertcat(vel, x_ddot, theta_dot, theta_ddot)

    f = ca.Function("f", [x, u], [xdot], ["x", "u"], ["xdot"])
    A = ca.Function("A", [x, u], [ca.jacobian(xdot, x)], ["x", "u"], ["A"])
    B = ca.Function("B", [x, u], [ca.jacobian(xdot, u)], ["x", "u"], ["B"])

    k1 = f(x, u); k2 = f(x + dt/2*k1, u)
    k3 = f(x + dt/2*k2, u); k4 = f(x + dt*k3, u)
    F = ca.Function("F", [x, u], [x + dt/6*(k1 + 2*k2 + 2*k3 + k4)],
                    ["x", "u"], ["x_next"])
    return {"f": f, "A": A, "B": B, "F": F}


# ============================================================================ #
#  Cart-Pole plant                                                              #
# ============================================================================ #
class CartPolePlant:
    def __init__(self, m_c=1.0, m_p=0.1, l=0.5, g=9.81, d=0.0):
        """
        Cart-Pole plant with full nonlinear dynamics.

        The nonlinear SIMULATOR (step / _derivatives) stays in fast NumPy.
        The linear model (A, B), the continuous dynamics (f), and the discrete
        RK4 map (F) all come from ONE symbolic model so nothing can drift apart.

        Parameters
        ----------
        m_c : cart mass (kg)
        m_p : pendulum point-mass at tip (kg)
        l   : pendulum length, pivot -> tip (m)
        g   : gravity (m/s^2)
        d   : accepted for backward compatibility. A, B are now the exact
              Jacobian of the (frictionless) nonlinear model, so d no longer
              modifies the linear model. Physical friction is added to the
              nonlinear sim via set_friction().
        """
        self.m_c, self.m_p, self.l, self.g, self.d = m_c, m_p, l, g, d
        self._viscous = 0.0      # b_v  (N·s/m)
        self._coulomb = 0.0      # F_c  (N)
        self._dt_internal = 0.002  # 500 Hz physics sub-step

        # ---- ONE symbolic model feeds A, B, f, F --------------------------
        self._sym = build_symbolic_cartpole(m_c, m_p, l, g)
        self.f = self._sym["f"]   # continuous nominal dynamics  (iLQR / MPC linearization)
        self.F = self._sym["F"]   # discrete RK4 prediction map  (MPC / GP-MPC)

        x_eq, u_eq = ca.DM([0, 0, 0, 0]), ca.DM([0])
        self.A = np.array(self._sym["A"](x_eq, u_eq))   # exact Jacobian at upright
        self.B = np.array(self._sym["B"](x_eq, u_eq))
        self.C = np.array([[1.0, 0.0, 0.0, 0.0],        # measure cart_pos, pole_angle
                           [0.0, 0.0, 1.0, 0.0]])

    # ---------------------------------------------------------------------- #
    #  Friction API                                                            #
    # ---------------------------------------------------------------------- #
    def set_friction(self, cart_frictionloss: float = 0.0,
                     pole_frictionloss: float = 0.0,
                     cart_damping: float = 0.0):
        """
        Two-component rail friction for the nonlinear simulator:
            F_friction = -b_v·ẋ - F_c·sign(ẋ)
        b_v = cart_damping (viscous), F_c = cart_frictionloss (Coulomb).
        Note: friction is the part of the dynamics the nominal model f does NOT
        know about — i.e. exactly the residual δf learned by the GP model.
        """
        self._viscous = cart_damping
        self._coulomb = cart_frictionloss

    # ---------------------------------------------------------------------- #
    #  Fast NumPy simulator (same equations as the symbolic f; guarded below)  #
    # ---------------------------------------------------------------------- #
    def _derivatives(self, state: np.ndarray, u: float) -> np.ndarray:
        _, x_dot, theta, theta_dot = state
        m_c, m_p, L, g = self.m_c, self.m_p, self.l, self.g
        sin_th, cos_th = np.sin(theta), np.cos(theta)

        DEADBAND = 1e-4
        f_visc = self._viscous * x_dot
        f_coul = self._coulomb * np.sign(x_dot) if abs(x_dot) > DEADBAND else 0.0
        F_fric = f_visc + f_coul

        den = m_c + m_p * sin_th**2
        x_ddot = (u - F_fric + m_p*L*theta_dot**2*sin_th
                  - m_p*g*sin_th*cos_th) / den
        theta_ddot = ((m_c + m_p)*g*sin_th
                      - cos_th*(u - F_fric + m_p*L*theta_dot**2*sin_th)) / (L*den)
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    def _rk4_step(self, state: np.ndarray, u: float, dt: float) -> np.ndarray:
        k1 = self._derivatives(state, u)
        k2 = self._derivatives(state + 0.5*dt*k1, u)
        k3 = self._derivatives(state + 0.5*dt*k2, u)
        k4 = self._derivatives(state + dt*k3, u)
        return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def step(self, state: np.ndarray, u: float, dt: float) -> np.ndarray:
        """Advance one control step dt, sub-stepping at the 2 ms physics rate."""
        n_substeps = max(1, int(round(dt / self._dt_internal)))
        dt_sub = dt / n_substeps
        current = state.copy()
        for _ in range(n_substeps):
            current = self._rk4_step(current, u, dt_sub)
        return current

    # ---------------------------------------------------------------------- #
    #  Guardrail: the NumPy simulator must match the symbolic model exactly    #
    # ---------------------------------------------------------------------- #
    def assert_consistent(self, n_samples: int = 200, tol: float = 1e-9) -> bool:
        """
        Frictionless NumPy _derivatives must equal the symbolic f everywhere.
        Run this in a test so the two implementations can never silently drift.
        """
        rng = np.random.default_rng(0)
        v, c = self._viscous, self._coulomb
        self._viscous = self._coulomb = 0.0
        try:
            for _ in range(n_samples):
                x = rng.uniform(-2, 2, 4)
                u = float(rng.uniform(-10, 10))
                num = self._derivatives(x, u)
                sym = np.array(self.f(x, u)).flatten()
                assert np.max(np.abs(num - sym)) < tol, \
                    "simulator drifted from symbolic model!"
        finally:
            self._viscous, self._coulomb = v, c
        return True
