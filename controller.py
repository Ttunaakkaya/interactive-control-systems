import numpy as np
import control as ct
from scipy.signal import place_poles, cont2discrete
from scipy.linalg import solve_continuous_are, solve_discrete_are
import scipy.linalg as la


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
        real_part = -zeta * wn
        imag_part =  wn * np.sqrt(max(1.0 - zeta**2, 0.0))
        p1 = complex(real_part,  imag_part)
        p2 = complex(real_part, -imag_part)
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
        P          = solve_discrete_are(self.Ad, self.Cd.T, Q_v, R_w)
        S          = self.Cd @ P @ self.Cd.T + R_w
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