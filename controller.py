"""
controller.py — the complete controller suite for the cart-pole digital twin.

Classical:       PIDController, StateSpaceController (pole placement)
Optimal:         LQRController, LQIController
Estimation:      LuenbergerObserver, KalmanFilter (steady-state, ZOH + DARE)
Planning:        TrajectoryPlanner (quintic), solve_ilqr + iLQRController
Predictive:      MPCController (nonlinear, constrained, warm-started)
Learning-based:  GPMPCController (GP residual + chance/robust tightening)
Safe control:    MPSCFilter (certifies any policy) + RecklessPolicy (demo)

All controllers share the interface  compute(setpoint, current_state[, dt]);
optimisation-based ones also expose reset() and diagnostic attributes.
"""
import numpy as np
import control as ct
from scipy.signal import place_poles, cont2discrete
from scipy.linalg import solve_continuous_are, solve_discrete_are
import scipy.linalg as la
import casadi as ca


# ============================================================================ #
#  PID Controller — Classical SISO                                              #
# ============================================================================ #
class PIDController:
    """
    Single-Input Single-Output PID controller wired to CART POSITION error.

        u = Kp·e + Ki·∫e dt + Kd·ė      where  e = x_ref − x_cart  (metres)

    Fundamental limitation for the inverted pendulum:
        PID sees only ONE output (cart position). It has no explicit knowledge
        of the pole angle θ. Without θ feedback, it cannot stabilise the
        upright equilibrium — it can only move the cart. A high Kd can
        implicitly couple to angle dynamics via cart acceleration, but this
        is fragile and highly parameter-sensitive.

    Educational role:
        Demonstrates why SISO control is insufficient for a 4-state unstable
        system. Compare with Pole Placement to see the structural gap.
    """
    def __init__(self, kp: float = 0.0, ki: float = 0.0, kd: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_error = 0.0
        self.prev_error     = 0.0
        self._first_call    = True

    def reset(self):
        self.integral_error = 0.0
        self.prev_error     = 0.0
        self._first_call    = True

    def compute(self, setpoint: float, current_value: float, dt: float) -> float:
        """
        Parameters
        ----------
        setpoint      : target cart position (m)
        current_value : measured cart position (m)  ← position, NOT angle
        dt            : timestep (s)
        """
        error = setpoint - current_value
        P = self.kp * error
        self.integral_error += error * dt
        I = self.ki * self.integral_error
        if self._first_call:
            # no history yet: seeding prev_error avoids the classic
            # "derivative kick" (a one-step D spike of kd*e/dt at t=0)
            self.prev_error  = error
            self._first_call = False
        D = self.kd * (error - self.prev_error) / dt
        self.prev_error = error
        return P + I + D


# ============================================================================ #
#  Pole Placement (State-Space) Controller                                      #
# ============================================================================ #
class StateSpaceController:
    """
    Full-state feedback via direct pole placement.

    Design:
        Desired poles derived from 2nd-order prototype:
            p_{1,2} = −ζωn ± jωn√(1−ζ²)
        Plus two auxiliary real poles (p3, p4) placed well to the left.
        Feedback gain K solved via Ackermann (ct.place).
        Pre-filter Nr computed from closed-loop DC gain so steady-state
        cart position tracks the setpoint under zero disturbance.

    self.poles is exposed publicly so the Luenberger observer can
    automatically scale its poles to be obs_speed× faster.
    """
    def __init__(self, A, B, zeta: float = 0.7, wn: float = 4.0,
                 p3: float = -10.0, p4: float = -12.0):
        self.A = A
        self.B = B
        sigma = zeta * wn   # real part magnitude (always positive)

        if zeta < 1.0:
            # Underdamped: complex conjugate pair
            #   p_{1,2} = -σ ± j·ωd   where ωd = ωn·√(1-ζ²)
            omega_d = wn * np.sqrt(1.0 - zeta**2)
            p1 = complex(-sigma,  omega_d)
            p2 = complex(-sigma, -omega_d)

        elif zeta == 1.0:
            # Critically damped: repeated real pole at -σ.
            # ct.place (Ackermann) cannot handle repeated poles — split
            # by a minimal ε so the design intent is preserved.
            eps = max(0.01 * sigma, 0.01)
            p1 = complex(-sigma - eps, 0.0)
            p2 = complex(-sigma + eps, 0.0)

        else:
            # Overdamped (ζ > 1): two distinct real poles
            #   p_{1,2} = -σ ± ωn·√(ζ²-1)
            delta = wn * np.sqrt(zeta**2 - 1.0)
            p1 = complex(-sigma - delta, 0.0)
            p2 = complex(-sigma + delta, 0.0)
        self.desired_poles = [p1, p2, p3, p4]
        self.poles         = self.desired_poles
        self.K  = ct.place(self.A, self.B, self.desired_poles)
        C_track = np.array([[1.0, 0.0, 0.0, 0.0]])
        A_cl    = self.A - self.B @ self.K
        DC      = C_track @ np.linalg.inv(-A_cl) @ self.B
        self.Nr = 1.0 / DC[0, 0]

    def compute(self, setpoint: float, current_state: np.ndarray) -> float:
        return float(-np.dot(self.K, current_state)[0] + self.Nr * setpoint)


# ============================================================================ #
#  LQR Controller                                                               #
# ============================================================================ #
class LQRController:
    """
    Optimal full-state feedback via the Linear Quadratic Regulator.

    Minimises:  J = ∫₀^∞ (x'Qx + u'Ru) dt

    Q = diag([q_pos, 0.1, q_ang, 0.1])
    R = [[r_weight]]

    self.poles exposes closed-loop eigenvalues for observer scaling.
    Nr feedforward valid only under zero disturbance — use LQI otherwise.
    """
    def __init__(self, A, B, q_pos: float = 1.0, q_ang: float = 10.0,
                 r_weight: float = 1.0):
        self.A = A
        self.B = B
        self.Q = np.diag([q_pos, 0.1, q_ang, 0.1])
        self.R = np.array([[r_weight]])
        self.K, _, eigs = ct.lqr(self.A, self.B, self.Q, self.R)
        self.poles = list(eigs.flatten())
        C_track = np.array([[1.0, 0.0, 0.0, 0.0]])
        A_cl    = self.A - self.B @ self.K
        DC      = C_track @ np.linalg.inv(-A_cl) @ self.B
        self.Nr = 1.0 / DC[0, 0]

    def compute(self, setpoint: float, current_state: np.ndarray) -> float:
        return float(-np.dot(self.K, current_state)[0] + self.Nr * setpoint)


# ============================================================================ #
#  LQI Controller                                                               #
# ============================================================================ #
class LQIController:
    """
    LQR augmented with integral action on cart position error.

    Augmented state: x_aug = [x, ẋ, θ, θ̇, ∫e]ᵀ  (5×1)

    The integral state eliminates steady-state error under constant
    disturbances (wind, friction) via the Internal Model Principle.

    Q_int tuning: start low (~0.3×Q_pos), increase cautiously.
    High Q_int with a slow/unconverged observer → integral windup → instability.
    """
    def __init__(self, A, B, q_pos: float = 100.0, q_ang: float = 10.0,
                 q_int: float = 150.0, r_weight: float = 1.0):
        C_p = np.array([[1.0, 0.0, 0.0, 0.0]])
        A_aug = np.zeros((5, 5))
        A_aug[0:4, 0:4] = A
        A_aug[4,   0:4] = C_p
        B_aug = np.zeros((5, 1))
        B_aug[0:4, :] = B
        Q_aug = np.diag([q_pos, 0.1, q_ang, 0.1, q_int])
        R     = np.array([[r_weight]])
        P     = la.solve_continuous_are(A_aug, B_aug, Q_aug, R)
        K_aug = np.linalg.inv(R) @ B_aug.T @ P
        self.K   = K_aug[0, 0:4]
        self.K_i = K_aug[0, 4]
        self.integral_error = 0.0

    def compute(self, setpoint: float, current_state: np.ndarray,
                dt: float) -> float:
        self.integral_error += (current_state[0] - setpoint) * dt
        return float(-np.dot(self.K, current_state) - self.K_i * self.integral_error)

    def reset(self):
        self.integral_error = 0.0


# ============================================================================ #
#  Luenberger Observer                                                          #
# ============================================================================ #
class LuenbergerObserver:
    """
    Full-order deterministic observer (continuous, Euler-integrated).

    Observer ODE:  x̂̇ = A·x̂ + B·u + L·(y − C·x̂)

    L computed by dual pole placement:  place_poles(Aᵀ, Cᵀ, obs_poles).T

    Observer poles must be 2–5× faster (further left) than controller poles.
    Limitation: deterministic — amplifies sensor noise at high gains.
    Use Kalman Filter for noisy measurements.
    """
    def __init__(self, A, B, C, observer_poles):
        self.A = A
        self.B = B
        self.C = C
        res    = place_poles(A.T, C.T, observer_poles)
        self.L = res.gain_matrix.T
        self.x_hat = np.zeros(A.shape[0])

    def update(self, u: float, y: np.ndarray, dt: float) -> np.ndarray:
        innovation = y - self.C @ self.x_hat
        dx         = (self.A @ self.x_hat
                      + self.B.flatten() * u
                      + (self.L @ innovation).flatten())
        self.x_hat += dx * dt
        return self.x_hat

    def reset(self, initial_guess: np.ndarray):
        self.x_hat = initial_guess.copy()


# ============================================================================ #
#  Kalman Filter — Discrete-Time, Steady-State                                 #
# ============================================================================ #
class KalmanFilter:
    """
    Steady-state discrete-time Kalman filter with ZOH discretisation.

    Discretisation (ZOH at control rate dt):
        (A, B, C) → (Ad, Bd, Cd)   via scipy.signal.cont2discrete

    Steady-state gain from the Discrete Algebraic Riccati Equation (DARE):
        P  = Ad·P·Adᵀ − Ad·P·Cdᵀ·(Cd·P·Cdᵀ + Rd)⁻¹·Cd·P·Adᵀ + Qd
        L  = P·Cdᵀ·(Cd·P·Cdᵀ + Rd)⁻¹

    Predict–Correct update per step:
        x̂[k|k−1] = Ad·x̂[k−1] + Bd·u[k−1]          (predict)
        x̂[k|k]   = x̂[k|k−1] + L·(y[k] − Cd·x̂[k|k−1])  (correct)

    This is mathematically consistent — the DARE is solved at the same
    discrete rate as the controller, eliminating the CARE+Euler mismatch
    of a naive continuous implementation.

    Tuning ratio Qd/Rd:
        High Qd/Rd → trust sensors → fast but noisy estimate
        Low  Qd/Rd → trust model  → smooth but slow estimate
    """
    def __init__(self, A, B, C, Q_v: np.ndarray, R_w: np.ndarray,
                 dt: float = 0.02):
        sys_d      = cont2discrete((A, B, C, np.zeros((C.shape[0], 1))),
                                   dt, method='zoh')
        self.Ad    = sys_d[0]
        self.Bd    = sys_d[1]
        self.Cd    = sys_d[2]

        # Numerical guard: R_w too small → (Cd·P·Cdᵀ + R_w)⁻¹ → ∞ → DARE diverges.
        # Clamp diagonal to minimum 1e-3 so the matrix inversion stays finite.
        # Physically this means "at most 1000× trust in sensors" — still very aggressive.
        R_w_safe   = np.maximum(R_w, np.eye(R_w.shape[0]) * 1e-3)

        # Estimation Riccati via DUALITY: the filter DARE
        #     P = Ad·P·Adᵀ − Ad·P·Cdᵀ(Cd·P·Cdᵀ+R)⁻¹·Cd·P·Adᵀ + Q
        # maps onto scipy's control-form DARE  aᵀXa − X − aᵀXb(r+bᵀXb)⁻¹bᵀXa + q
        # only with a = Ad.T, b = Cd.T. Passing Ad untransposed solves the wrong
        # equation — it fails outright (LinAlgError) for common Q/R settings.
        P          = solve_discrete_are(self.Ad.T, self.Cd.T, Q_v, R_w_safe)
        S          = self.Cd @ P @ self.Cd.T + R_w_safe
        self.L     = P @ self.Cd.T @ np.linalg.inv(S)
        self.x_hat = np.zeros(A.shape[0])

    def update(self, u: float, y: np.ndarray, dt: float = None) -> np.ndarray:
        x_pred     = self.Ad @ self.x_hat + self.Bd.flatten() * u
        self.x_hat = x_pred + (self.L @ (y - self.Cd @ x_pred)).flatten()
        return self.x_hat

    def reset(self, initial_guess: np.ndarray):
        self.x_hat = initial_guess.copy()


# ============================================================================ #
#  Trajectory Planner — 5th-Order Quintic Polynomial                           #
# ============================================================================ #
class TrajectoryPlanner:
    """
    Minimum-jerk point-to-point trajectory via quintic polynomial.

    Boundary conditions (6 → degree-5):
        t=0: p=p_start, v=0, a=0
        t=T: p=p_end,   v=0, a=0

    Normalised form (τ = t/T):
        p(τ) = p₀ + Δp·(10τ³ − 15τ⁴ + 6τ⁵)
        v(τ) = (Δp/T)·(30τ² − 60τ³ + 30τ⁴)
        a(τ) = (Δp/T²)·(60τ − 180τ² + 120τ³)

    Acceleration a(t) feeds the 2-DOF feedforward:
        F_ff = (m_c + m_p) · a_ref
    """
    def __init__(self, p_start: float, p_end: float, duration: float):
        self.p_0 = p_start
        self.p_f = p_end
        self.T   = max(duration, 1e-6)

    def get_state(self, t: float):
        if t <= 0.0:
            return self.p_0, 0.0, 0.0
        if t >= self.T:
            return self.p_f, 0.0, 0.0
        tau  = t / self.T
        tau2, tau3, tau4, tau5 = tau**2, tau**3, tau**4, tau**5
        dp   = self.p_f - self.p_0
        p    = self.p_0 + dp * (10*tau3 - 15*tau4 + 6*tau5)
        v    = (dp / self.T)    * (30*tau2  - 60*tau3  + 30*tau4)
        a    = (dp / self.T**2) * (60*tau   - 180*tau2 + 120*tau3)
        return p, v, a
    

# ============================================================================ #
#  iLQR — Iterative LQR trajectory optimizer                                   #
#  ----------------------------------------------------------------------------#
#  WHY iLQR: LQR is a LOCAL controller (valid near upright). Swing-up from the  #
#  hanging position is a global nonlinear maneuver LQR structurally cannot do.  #
#  iLQR optimizes the full nonlinear trajectory: a backward pass builds a       #
#  quadratic approximation of the value function (giving a feedforward term k   #
#  and a time-varying feedback gain K), and a forward pass rolls the dynamics   #
#  out with a backtracking line search. Levenberg-Marquardt regularization on   #
#  Quu keeps the step well-defined when the quadratic model is non-convex.      #
# ============================================================================ #
from scipy.linalg import solve_discrete_are as _dare


def build_ilqr_funcs(F):
    """Vectorized discrete dynamics + Jacobians from the plant's RK4 map F."""
    x = ca.SX.sym("x", 4); u = ca.SX.sym("u", 1)
    xn = F(x, u)
    f_step = ca.Function("f_step", [x, u], [xn])
    Fx = ca.Function("Fx", [x, u], [ca.jacobian(xn, x)])   # ∂F/∂x
    Fu = ca.Function("Fu", [x, u], [ca.jacobian(xn, u)])   # ∂F/∂u
    return f_step, Fx, Fu


def solve_ilqr(F, x0, x_goal, N, Q, R, Qf, max_iters=100, tol=1e-5):
    """
    Iterative LQR. Returns (x_traj, u_traj, K_fb, cost).

      x_traj : (N+1, 4) optimal state trajectory
      u_traj : (N, 1)   optimal control sequence (the feedforward)
      K_fb   : (N, 1, 4) time-varying feedback gains (track x_traj robustly)

    Solve this ONCE and cache it (see app.py); it is the expensive step.
    """
    n, m = 4, 1
    x0 = np.asarray(x0, float); x_goal = np.asarray(x_goal, float)
    f_step, Fx, Fu = build_ilqr_funcs(F)
    Fx_map, Fu_map = Fx.map(N), Fu.map(N)     # evaluate all N Jacobians in one call

    # initial nominal trajectory: zero controls rolled forward
    u_traj = np.zeros((N, m))
    x_traj = np.zeros((N + 1, n)); x_traj[0] = x0
    for k in range(N):
        x_traj[k + 1] = np.array(f_step(x_traj[k], u_traj[k])).flatten()

    def cost(xt, ut):
        dx = xt[:N] - x_goal
        stage = 0.5*np.einsum('ki,ij,kj->', dx, Q, dx) + 0.5*np.einsum('ki,ij,kj->', ut, R, ut)
        dN = xt[N] - x_goal
        return stage + 0.5*dN @ Qf @ dN

    J = cost(x_traj, u_traj); mu = 1e-6
    for _ in range(max_iters):
        # ---- batched Jacobians along the current trajectory ----
        A_all = np.array(Fx_map(x_traj[:N].T, u_traj.T)).reshape(n, n, N, order='F')
        B_all = np.array(Fu_map(x_traj[:N].T, u_traj.T)).reshape(n, m, N, order='F')

        # ---- backward pass: value-function quadratic -> k_ff, K_fb ----
        Vx = Qf @ (x_traj[N] - x_goal); Vxx = Qf.copy()
        k_ff = np.zeros((N, m)); K_fb = np.zeros((N, m, n)); diverged = False
        for k in range(N - 1, -1, -1):
            A = A_all[:, :, k]; B = B_all[:, :, k]; dx = x_traj[k] - x_goal
            Qx  = Q @ dx + A.T @ Vx
            Qu  = R @ u_traj[k] + B.T @ Vx
            Qxx = Q + A.T @ Vxx @ A
            Quu = R + B.T @ Vxx @ B + mu*np.eye(m)     # regularized
            Qux = B.T @ Vxx @ A
            if Quu[0, 0] <= 0:                         # not positive-definite -> regularize more
                diverged = True; break
            Quu_inv = np.linalg.inv(Quu)
            k_ff[k] = (-Quu_inv @ Qu).flatten()        # feedforward
            K_fb[k] = -Quu_inv @ Qux                   # feedback gain
            Vx  = Qx + K_fb[k].T @ Quu @ k_ff[k] + K_fb[k].T @ Qu + Qux.T @ k_ff[k]
            Vxx = Qxx + K_fb[k].T @ Quu @ K_fb[k] + K_fb[k].T @ Qux + Qux.T @ K_fb[k]
            Vxx = 0.5*(Vxx + Vxx.T)
        if diverged:
            mu *= 10
            if mu > 1e10: break
            continue

        # ---- forward pass: backtracking line search on alpha ----
        improved = False
        for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            x_new = np.zeros_like(x_traj); x_new[0] = x0; u_new = np.zeros_like(u_traj)
            for k in range(N):
                u_new[k] = u_traj[k] + alpha*k_ff[k] + K_fb[k] @ (x_new[k] - x_traj[k])
                x_new[k + 1] = np.array(f_step(x_new[k], u_new[k])).flatten()
            J_new = cost(x_new, u_new)
            if J_new < J:
                improved = True; break
        if improved:
            done = abs(J - J_new) < tol
            x_traj, u_traj, J = x_new, u_new, J_new; mu = max(mu/2, 1e-8)
            if done: break
        else:
            mu *= 10
            if mu > 1e10: break
    return x_traj, u_traj, K_fb, J


class iLQRController:
    """
    Plays back an iLQR swing-up trajectory with its time-varying feedback gains,
    then switches to a terminal infinite-horizon LQR to HOLD the pole upright
    after the maneuver ends. Interface matches the other controllers:
    compute(setpoint, current_state).

    Usage:
        x_traj, u_traj, K_fb, _ = solve_ilqr(plant.F, x0, x_goal, N, Q, R, Qf)
        ctrl = iLQRController(plant, x_goal, x_traj, u_traj, K_fb)
    """
    def __init__(self, plant, x_goal, x_traj, u_traj, K_fb):
        self.x_goal = np.asarray(x_goal, float)
        self.x_traj, self.u_traj, self.K_fb = x_traj, u_traj, K_fb
        self.N = len(u_traj); self._k = 0
        # terminal discrete LQR at upright (the "balance" gain after swing-up)
        x = ca.SX.sym("x", 4); u = ca.SX.sym("u", 1); xn = plant.F(x, u)
        Ad = np.array(ca.Function("A", [x, u], [ca.jacobian(xn, x)])(self.x_goal, [0.0]))
        Bd = np.array(ca.Function("B", [x, u], [ca.jacobian(xn, u)])(self.x_goal, [0.0]))
        Qh, Rh = np.diag([10., 1., 50., 1.]), np.array([[0.1]])
        P = _dare(Ad, Bd, Qh, Rh)
        self.K = np.linalg.solve(Rh + Bd.T @ P @ Bd, Bd.T @ P @ Ad)   # 1x4 hold gain

    def reset(self):
        self._k = 0

    def compute(self, setpoint, current_state):
        x = np.asarray(current_state, float)
        if self._k < self.N:                       # swing-up phase: feedforward + tracking
            u = self.u_traj[self._k] + self.K_fb[self._k] @ (x - self.x_traj[self._k])
        else:                                       # balance phase: hold upright
            u = -self.K @ (x - self.x_goal)
        self._k += 1
        return float(np.atleast_1d(u)[0])


# ============================================================================ #
#  MPC — Nonlinear Model Predictive Control                                     #
#  ----------------------------------------------------------------------------#
#  WHY MPC: LQR applies one fixed gain and constraints are an afterthought      #
#  (clip after the fact). MPC solves, at EVERY step, a finite-horizon optimal   #
#  control problem over the full nonlinear model with the constraints INSIDE    #
#  the optimization:                                                            #
#      |u| <= max_force        actuator limit the plan respects by design      #
#      |cart_pos| <= limit     state constraint the plan steers around         #
#  Then it applies only the first input and re-solves (receding horizon).      #
#  Consequences you can demo:                                                   #
#    - under saturation, LQR+clip can destabilize; MPC plans WITH the limit    #
#    - near the rail end, MPC brakes early; LQR discovers the wall too late    #
#                                                                               #
#  Implementation notes:                                                        #
#    - multiple shooting: states X and inputs U are both decision variables,   #
#      dynamics x_{k+1} = F(x_k,u_k) enter as equality constraints             #
#    - terminal cost P from DARE at upright approximates the infinite tail     #
#      (the standard stability ingredient)                                     #
#    - the position constraint is SOFT (slack + heavy penalty) so IPOPT stays  #
#      feasible when a disturbance shoves the state outside the box            #
#    - the Opti problem is built ONCE (parametrized by x0 and x_ref) and       #
#      re-solved each step with the previous solution shifted as warm start    #
# ============================================================================ #


class MPCController:
    """
    Nonlinear MPC on the plant's discrete RK4 map F (multiple shooting).
    Interface matches the other controllers: compute(setpoint, current_state).

    Tuning intuition:
      horizon  : how far ahead it plans. Longer = more foresight at the
                 constraint boundary (>=40 recommended; 30 can oscillate and
                 drop the pole under sustained unmodeled disturbance).
      q_pos/q_ang/r_weight : same LQR-style trade-offs, applied per stage.
      max_force / track_limit : the constraints themselves.
    """
    def __init__(self, plant, horizon=40, q_pos=10.0, q_ang=50.0, r_weight=0.1,
                 max_force=15.0, track_limit=2.0):
        self.N = horizon
        n, m = 4, 1

        # --- terminal cost from DARE at the upright equilibrium
        x_eq, u_eq = ca.DM([0, 0, 0, 0]), ca.DM([0])
        xs = ca.SX.sym("x", 4); us = ca.SX.sym("u", 1); xn = plant.F(xs, us)
        Ad = np.array(ca.Function("A", [xs, us], [ca.jacobian(xn, xs)])(x_eq, u_eq))
        Bd = np.array(ca.Function("B", [xs, us], [ca.jacobian(xn, us)])(x_eq, u_eq))
        Q = np.diag([q_pos, 1.0, q_ang, 1.0])
        R = np.diag([r_weight])
        P = _dare(Ad, Bd, Q, R)

        # --- build the parametrized Opti problem ONCE ----------------------
        opti = ca.Opti()
        X = opti.variable(n, self.N + 1)          # planned state trajectory
        U = opti.variable(m, self.N)              # planned control sequence
        x0_p   = opti.parameter(n)                # current state  (set each step)
        xref_p = opti.parameter(n)                # reference      (set each step)

        cost = 0
        for k in range(self.N):
            dx = X[:, k] - xref_p
            cost += dx.T @ Q @ dx + U[:, k].T @ R @ U[:, k]
            opti.subject_to(X[:, k + 1] == plant.F(X[:, k], U[:, k]))   # dynamics
            opti.subject_to(opti.bounded(-max_force, U[0, k], max_force))
        dN = X[:, self.N] - xref_p
        cost += dN.T @ ca.DM(P) @ dN                                     # terminal

        # soft position constraint: slack eps >= 0, heavily penalized.
        eps = opti.variable(self.N + 1)
        opti.subject_to(eps >= 0)
        for k in range(self.N + 1):
            opti.subject_to(X[0, k] <=  track_limit + eps[k])
            opti.subject_to(X[0, k] >= -track_limit - eps[k])
        cost += 1e5 * ca.sumsqr(eps)

        opti.subject_to(X[:, 0] == x0_p)
        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False},
                    {"print_level": 0, "max_iter": 300, "acceptable_tol": 1e-6})

        self._opti, self._X, self._U = opti, X, U
        self._x0_p, self._xref_p = x0_p, xref_p
        self._prev_X = None; self._prev_U = None
        self.last_plan = None      # planned trajectory (for visualization later)

    def reset(self):
        self._prev_X = None; self._prev_U = None

    def compute(self, setpoint, current_state):
        x = np.asarray(current_state, float)
        xref = np.array([setpoint, 0.0, 0.0, 0.0])
        self._opti.set_value(self._x0_p, x)
        self._opti.set_value(self._xref_p, xref)

        # warm start: previous solution shifted one step forward
        if self._prev_X is not None:
            Xg = np.hstack([self._prev_X[:, 1:], self._prev_X[:, -1:]])
            Ug = np.vstack([self._prev_U[1:], self._prev_U[-1:]])
            Xg[:, 0] = x
            self._opti.set_initial(self._X, Xg)
            self._opti.set_initial(self._U, Ug.T)

        try:
            sol = self._opti.solve()
            Xs = np.array(sol.value(self._X))
            Us = np.atleast_2d(np.array(sol.value(self._U)))
            if Us.shape[0] != self.N:
                Us = Us.T
            if not np.all(np.isfinite(Us)):
                raise RuntimeError("non-finite solution")
            self._prev_X, self._prev_U = Xs, Us
            self.last_plan = Xs
            return float(Us[0, 0])
        except RuntimeError:
            # solver hiccup: fall back to the next input of the LAST good plan
            # (receding-horizon logic makes this the principled fallback)
            if self._prev_U is not None and len(self._prev_U) > 1:
                self._prev_U = np.vstack([self._prev_U[1:], self._prev_U[-1:]])
                return float(self._prev_U[0, 0])
            return 0.0
        
# ============================================================================ #
#  GP-MPC — Learning-Based MPC with uncertainty-aware constraint tightening     #
#  ----------------------------------------------------------------------------#
#  THE IDEA: the controller's prediction model is prior + learned residual,     #
#      x_{k+1} = f̄(x_k, u_k) + δf(x_k, u_k),                                   #
#  so the MPC plans with what the data says the plant ACTUALLY does — wrong     #
#  mass, unmodeled friction and all.                                            #
#                                                                               #
#  ARCHITECTURE (sequential / zeroth-order GP-MPC): embedding the GP kernels    #
#  into the NLP makes IPOPT ~4x slower. Instead, each step the GP residual      #
#  mean AND std are evaluated ALONG the warm-start plan with one batched        #
#  sklearn call (~1 ms) and passed to the NLP as PARAMETERS — the NLP stays     #
#  plain-MPC-sized (~45 ms warm). Re-solving from the newest state every 20 ms  #
#  lets the plan and the GP evaluation converge together in closed loop.        #
#  use_gp=False gives the ablation: same NLP, μ ≡ 0 (prior-only).               #
#                                                                               #
#  CONSTRAINT TIGHTENING:                                                       #
#  The position constraint is tightened per prediction step by a margin:        #
#      |cart_pos_k| <= track_limit - TIGHT_k                                    #
#  where TIGHT_k comes from first-order covariance propagation along the plan:  #
#      Σ_{k+1} = A_k Σ_k A_kᵀ + S·diag(w_k)·Sᵀ,     TIGHT_k = κ·sqrt(Σ_k[0,0]) #
#                                                                               #
#  constraint_mode selects where w_k comes from:                                #
#    'nominal' : TIGHT ≡ 0            (no margin — the honest baseline)         #
#    'chance'  : w_k = σ_GP(z_k)²     (LEARNED, state-dependent: the model's    #
#                own uncertainty tightens the constraint)                       #
#    'robust'  : w_k = w̄² fixed      (worst-case assumed bound, state-         #
#                independent -> conservative — simplified tube MPC)             #
#                                                                               #
#  Note the epistemic-vs-exogenous distinction this makes demonstrable:         #
#  chance mode protects against what the LEARNED MODEL is unsure of; it cannot  #
#  see disturbances absent from training data (e.g. wind) — robust mode covers  #
#  those via its a-priori bound. Both margins ride the same parameter channel   #
#  as the GP means, so the NLP stays standard-MPC-sized (~47 ms warm solves).  #
# ============================================================================ #


class GPMPCController:
    """
    Learning-based MPC:  x+ = F_nominal(x,u) + S·μ_GP,  with per-step
    constraint tightening from propagated uncertainty (see banner above).

    Exposes after each compute():
        last_plan      (4, N+1)  planned state trajectory
        last_pos_sigma (N+1,)    propagated position σ along the plan
        last_tight     (N+1,)    applied tightening κ·σ (or robust margin)
    """
    def __init__(self, nominal_plant, gp=None, use_gp=True, horizon=40,
                 q_pos=10.0, q_ang=50.0, r_weight=0.1,
                 max_force=15.0, track_limit=2.0,
                 constraint_mode='nominal', kappa=2.0, w_bound=0.05):
        self.N = horizon; self.gp = gp
        self.use_gp = bool(use_gp and gp is not None)
        self.mode = constraint_mode; self.kappa = kappa; self.w_bound = w_bound
        n, m = 4, 1

        xs = ca.SX.sym("x", 4); us = ca.SX.sym("u", 1)
        xn_ = nominal_plant.F(xs, us)
        # batched discrete Jacobian along the plan (for covariance propagation)
        self._Fx = ca.Function("Fx", [xs, us], [ca.jacobian(xn_, xs)]).map(self.N)
        x_eq, u_eq = ca.DM([0, 0, 0, 0]), ca.DM([0])
        Ad = np.array(ca.Function("A", [xs, us], [ca.jacobian(xn_, xs)])(x_eq, u_eq))
        Bd = np.array(ca.Function("B", [xs, us], [ca.jacobian(xn_, us)])(x_eq, u_eq))
        Q = np.diag([q_pos, 1.0, q_ang, 1.0])
        R = np.diag([r_weight])
        P = _dare(Ad, Bd, Q, R)

        opti = ca.Opti()
        X = opti.variable(n, self.N + 1)
        U = opti.variable(m, self.N)
        x0_p   = opti.parameter(n)
        xref_p = opti.parameter(n)
        MU     = opti.parameter(2, self.N)      # GP residual means along the plan
        TIGHT  = opti.parameter(self.N + 1)     # per-step constraint tightening

        cost = 0
        for k in range(self.N):
            dx = X[:, k] - xref_p
            cost += dx.T @ Q @ dx + U[:, k].T @ R @ U[:, k]
            corr = ca.vertcat(0, MU[0, k], 0, MU[1, k])
            opti.subject_to(X[:, k + 1] == nominal_plant.F(X[:, k], U[:, k]) + corr)
            opti.subject_to(opti.bounded(-max_force, U[0, k], max_force))
        dN = X[:, self.N] - xref_p
        cost += dN.T @ ca.DM(P) @ dN

        eps = opti.variable(self.N + 1)         # soft slack (keeps IPOPT feasible)
        opti.subject_to(eps >= 0)
        for k in range(self.N + 1):
            opti.subject_to(X[0, k] <=  track_limit - TIGHT[k] + eps[k])
            opti.subject_to(X[0, k] >= -track_limit + TIGHT[k] - eps[k])
        cost += 1e5 * ca.sumsqr(eps)

        opti.subject_to(X[:, 0] == x0_p)
        opti.minimize(cost)
        opti.solver("ipopt", {"print_time": False},
                    {"print_level": 0, "max_iter": 300, "acceptable_tol": 1e-6})

        self._opti, self._X, self._U = opti, X, U
        self._x0_p, self._xref_p, self._MU, self._TIGHT = x0_p, xref_p, MU, TIGHT
        self._prev_X = None; self._prev_U = None
        self.last_plan = None; self.last_sigma = None
        self.last_tight = None; self.last_pos_sigma = None

    def reset(self):
        self._prev_X = None; self._prev_U = None

    def _mu_sigma_along(self, Xplan, Uplan):
        """Batched GP residual mean AND std along the plan (one sklearn call)."""
        if not self.use_gp:
            return np.zeros((2, self.N)), np.zeros((2, self.N))
        mu, sd = self.gp.predict(Xplan[:self.N].copy(), Uplan.reshape(-1).copy())
        return mu.T, sd.T                                   # (2, N) each

    def _propagate_tightening(self, Xplan, Uplan, sigma_gp):
        """First-order covariance propagation along the plan -> κ·σ_pos per step."""
        if self.mode == 'nominal':
            self._last_sig_pos = np.zeros(self.N + 1)
            return np.zeros(self.N + 1)
        A_all = np.array(self._Fx(Xplan[:self.N].T, Uplan.T)).reshape(4, 4, self.N, order='F')
        if self.mode == 'chance':
            var_w = sigma_gp ** 2                 # learned, state-dependent (2, N)
        else:                                     # 'robust': fixed assumed bound
            var_w = np.full((2, self.N), self.w_bound ** 2)
        Sig = np.zeros((4, 4))
        tight = np.zeros(self.N + 1); sig_pos = np.zeros(self.N + 1)
        for k in range(self.N):
            A = A_all[:, :, k]
            Sig = A @ Sig @ A.T
            Sig[1, 1] += var_w[0, k]; Sig[3, 3] += var_w[1, k]
            sig_pos[k + 1] = np.sqrt(max(Sig[0, 0], 0.0))
            tight[k + 1] = self.kappa * sig_pos[k + 1]
        self._last_sig_pos = sig_pos
        return tight

    def compute(self, setpoint, current_state):
        x = np.asarray(current_state, float)
        xref = np.array([setpoint, 0.0, 0.0, 0.0])

        if self._prev_X is not None:
            Xg = np.hstack([self._prev_X[:, 1:], self._prev_X[:, -1:]]); Xg[:, 0] = x
            Ug = np.vstack([self._prev_U[1:], self._prev_U[-1:]])
        else:
            Xg = np.tile(x.reshape(-1, 1), (1, self.N + 1))
            Ug = np.zeros((self.N, 1))

        MUv, SDv = self._mu_sigma_along(Xg.T, Ug)
        TIv = self._propagate_tightening(Xg.T, Ug, SDv)
        self._opti.set_value(self._x0_p, x)
        self._opti.set_value(self._xref_p, xref)
        self._opti.set_value(self._MU, MUv)
        self._opti.set_value(self._TIGHT, TIv)
        self._opti.set_initial(self._X, Xg)
        self._opti.set_initial(self._U, Ug.T)

        try:
            sol = self._opti.solve()
            Xs = np.array(sol.value(self._X))
            Us = np.atleast_2d(np.array(sol.value(self._U)))
            if Us.shape[0] != self.N:
                Us = Us.T
            if not np.all(np.isfinite(Us)):
                raise RuntimeError("non-finite solution")
            self._prev_X, self._prev_U = Xs, Us
            self.last_plan = Xs; self.last_sigma = SDv; self.last_tight = TIv
            self.last_pos_sigma = getattr(self, '_last_sig_pos', np.zeros(self.N + 1))
            return float(Us[0, 0])
        except RuntimeError:
            if self._prev_U is not None and len(self._prev_U) > 1:
                self._prev_U = np.vstack([self._prev_U[1:], self._prev_U[-1:]])
                return float(self._prev_U[0, 0])
            return 0.0


# ============================================================================ #
#  MPSC — Model Predictive Safety Certification filter  +  Reckless demo policy #
#  ("certifiably safe RL"; cf. the Safe Learning in Robotics survey / MPSC)     #
#  ----------------------------------------------------------------------------#
#  THE IDEA: wrap ANY policy. Each step, given the proposed input u_prop,       #
#  solve:   min (u_0 - u_prop)^2   s.t. a SAFE tail exists:                     #
#           dynamics, |u|<=max_force, |pos|<=track_limit, |angle|<=angle_limit, #
#           terminal state inside a small recovery box.                         #
#  Safe proposal  -> returned unchanged (validated: max deviation 3e-4 N).      #
#  Unsafe proposal-> minimally modified, action concentrated at the boundary.   #
#                                                                               #
#  PRACTICAL NOTE (validated): construct with track_limit = 0.97 * physical     #
#  wall. The filter is optimally "lazy" (it defers braking to the last          #
#  feasible moment), so model/sim mismatch can eat a zero-margin boundary by    #
#  ~mm; the 3% standoff absorbs that. The terminal box stands in for a          #
#  certified invariant set (full MPSC uses an RPI/CLF set) — say so if asked.   #
# ============================================================================ #


class MPSCFilter:
    """Safety filter: least-restrictive certification of any proposed input.
    Call  u_safe = filter(state, u_prop)  every control step."""
    def __init__(self, plant, horizon=50, max_force=15.0, track_limit=1.5,
                 angle_limit=0.6, term_frac=0.7):
        self.N = horizon
        n, m = 4, 1
        opti = ca.Opti()
        X = opti.variable(n, self.N + 1); U = opti.variable(m, self.N)
        x0_p = opti.parameter(n); up_p = opti.parameter(1)

        obj = (U[0, 0] - up_p) ** 2 + 1e-4 * ca.sumsqr(U[0, 1:])
        for k in range(self.N):
            opti.subject_to(X[:, k + 1] == plant.F(X[:, k], U[:, k]))
            opti.subject_to(opti.bounded(-max_force, U[0, k], max_force))
        eps = opti.variable(self.N)               # path-constraint slack
        opti.subject_to(eps >= 0)
        for k in range(1, self.N + 1):
            opti.subject_to(X[0, k] <=  track_limit + eps[k - 1])
            opti.subject_to(X[0, k] >= -track_limit - eps[k - 1])
            opti.subject_to(X[2, k] <=  angle_limit + eps[k - 1])
            opti.subject_to(X[2, k] >= -angle_limit - eps[k - 1])
        obj += 1e6 * ca.sumsqr(eps)
        et = opti.variable(4)                     # terminal recovery-box slack
        opti.subject_to(et >= 0)
        tb = np.array([term_frac * track_limit, 1.0, 0.15, 1.0])
        for d in range(4):
            opti.subject_to(X[d, self.N] <=  tb[d] + et[d])
            opti.subject_to(X[d, self.N] >= -tb[d] - et[d])
        obj += 1e4 * ca.sumsqr(et)

        opti.subject_to(X[:, 0] == x0_p)
        opti.minimize(obj)
        opti.solver("ipopt", {"print_time": False},
                    {"print_level": 0, "max_iter": 300, "acceptable_tol": 1e-6})
        self._opti, self._X, self._U = opti, X, U
        self._x0_p, self._up_p = x0_p, up_p
        self._prev_X = None; self._prev_U = None
        self.max_force = max_force

    def reset(self):
        self._prev_X = None; self._prev_U = None

    def filter(self, state, u_prop):
        x = np.asarray(state, float)
        self._opti.set_value(self._x0_p, x)
        self._opti.set_value(self._up_p, float(u_prop))
        if self._prev_X is not None:
            Xg = np.hstack([self._prev_X[:, 1:], self._prev_X[:, -1:]]); Xg[:, 0] = x
            Ug = np.vstack([self._prev_U[1:], self._prev_U[-1:]])
            self._opti.set_initial(self._X, Xg); self._opti.set_initial(self._U, Ug.T)
        try:
            sol = self._opti.solve()
            Xs = np.array(sol.value(self._X))
            Us = np.atleast_2d(np.array(sol.value(self._U)))
            if Us.shape[0] != self.N:
                Us = Us.T
            if not np.all(np.isfinite(Us)):
                raise RuntimeError("non-finite")
            self._prev_X, self._prev_U = Xs, Us
            return float(Us[0, 0])
        except RuntimeError:
            if self._prev_U is not None and len(self._prev_U) > 1:
                self._prev_U = np.vstack([self._prev_U[1:], self._prev_U[-1:]])
                return float(self._prev_U[0, 0])
            return float(np.clip(u_prop, -self.max_force, self.max_force))


class RecklessPolicy:
    """Balances the pole competently but drives toward a point BEYOND the rail
    end — a stand-in for a policy trained without constraint knowledge.
    Interface matches the other controllers."""
    def __init__(self, plant, target_beyond_wall, u_max: float = 15.0):
        Q = np.diag([100.0, 1.0, 10.0, 1.0]); R = np.array([[1.0]])
        P = la.solve_continuous_are(plant.A, plant.B, Q, R)
        self.K = np.linalg.solve(R, plant.B.T @ P)
        self.xt = np.array([target_beyond_wall, 0.0, 0.0, 0.0])
        self.u_max = float(u_max)          # the policy's own actuator belief

    def compute(self, setpoint, current_state):
        x = np.asarray(current_state, float)
        return float(np.clip(float((-self.K @ (x - self.xt))[0]),
                             -self.u_max, self.u_max))
