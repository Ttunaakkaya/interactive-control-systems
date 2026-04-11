import numpy as np
import control as ct
from scipy.signal import place_poles
from scipy.linalg import solve_continuous_are
import scipy.linalg as la

class PIDController:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0):
        """
        Initializes the PID controller with specific tuning gains.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        # Memory variables for Integral and Derivative math
        self.integral_error = 0.0
        self.prev_error = 0.0

    def compute(self, setpoint, current_value, dt):
        """
        Calculates the control output (force u) based on the current error.
        """
        # 1. Calculate current error
        error = setpoint - current_value
        
        # 2. Proportional term
        P = self.kp * error
        
        # 3. Integral term (Area under the curve: error * time)
        self.integral_error += error * dt
        I = self.ki * self.integral_error
        
        # 4. Derivative term (Slope of the curve: change in error / time)
        derivative_error = (error - self.prev_error) / dt
        D = self.kd * derivative_error
        
        # 5. Save current error for the next loop's derivative calculation
        self.prev_error = error
        
        # Return the sum as the total control effort
        return P + I + D
    

class StateSpaceController:
    def __init__(self, A, B, zeta=0.7, wn=4.0, p3=-10.0, p4=-12.0):
        """
        Initializes the Full-State Feedback Controller using Pole Placement.
        zeta: Damping ratio (0.7 is a standard, smooth underdamped response)
        wn: Natural frequency (dictates speed. 4.0 rad/s is fairly responsive)
        p3, p4: The "fast" real poles pushed far into the left-half plane.
        """
        self.A = A
        self.B = B
        
        # 1. Calculate dominant complex poles using zeta and wn
        real_part = -zeta * wn
        imag_part = wn * np.sqrt(1 - zeta**2)
        
        p1 = complex(real_part, imag_part)
        p2 = complex(real_part, -imag_part)
        
        # 2. Define the exact desired locations for all 4 poles
        self.desired_poles = [p1, p2, p3, p4]
        
        # 3. Calculate the Feedback Gain Matrix (K) using Ackermann's formula
        self.K = ct.place(self.A, self.B, self.desired_poles)
        
        # 4. Calculate Feedforward Gain (Nr) for 2-DoF tracking
        # We want the cart to track a specific position on the track.
        # Position is the 1st state in our array, so C_track is [1, 0, 0, 0]
        C_track = np.array([[1.0, 0.0, 0.0, 0.0]])
        
        # Steady-state formula: Nr = -1 / ( C * inv(A - B*K) * B )
        sys_cl = self.A - self.B @ self.K
        DC_gain = C_track @ np.linalg.inv(sys_cl) @ self.B
        
        self.Nr = -1.0 / DC_gain[0, 0]

    def compute(self, setpoint, current_state):
        """
        Computes the 2-DoF control law: u = -Kx + Nr*r
        setpoint (r): Desired target position on the track (meters)
        current_state (x): The 4x1 state array [p, p_dot, theta, theta_dot]
        """
        # Feedback command: u_fb = -K * x
        # self.K is a 1x4 matrix, current_state is a 1D array. 
        u_feedback = -np.dot(self.K, current_state)[0]
        
        # Feedforward command: u_ff = Nr * r
        u_feedforward = self.Nr * setpoint
        
        # Total Actuator Force
        return u_feedback + u_feedforward
    

# ... (Keep your PIDController and StateSpaceController above this) ...

class LQRController:
    def __init__(self, A, B, q_pos=1.0, q_ang=10.0, r_weight=1.0):
        """
        Initializes the Linear Quadratic Regulator (Optimal Control).
        q_pos: Penalty for cart drifting away from target.
        q_ang: Penalty for pole falling away from perfectly upright.
        r_weight: Penalty for using the motor (actuator effort).
        """
        self.A = A
        self.B = B
        
        # 1. Construct the Q matrix (State Penalties)
        # We lightly penalize the velocities (0.1) just to dampen jitter, 
        # but the main focus is on Position (index 0) and Angle (index 2).
        self.Q = np.diag([q_pos, 0.1, q_ang, 0.1])
        
        # 2. Construct the R matrix (Actuation Penalty)
        self.R = np.array([[r_weight]])
        
        # 3. Calculate optimal feedback gain K using python-control's algebraic Riccati solver
        self.K, _, _ = ct.lqr(self.A, self.B, self.Q, self.R)
        
        # 4. Calculate Feedforward Gain (Nr) for 2-DoF tracking
        C_track = np.array([[1.0, 0.0, 0.0, 0.0]])
        sys_cl = self.A - self.B @ self.K
        DC_gain = C_track @ np.linalg.inv(sys_cl) @ self.B
        self.Nr = -1.0 / DC_gain[0, 0]

    def compute(self, setpoint, current_state):
        """
        Computes the optimal 2-DoF control law: u = -Kx + Nr*r
        """
        u_feedback = -np.dot(self.K, current_state)[0]
        u_feedforward = self.Nr * setpoint
        return u_feedback + u_feedforward
    

class LuenbergerObserver:
    def __init__(self, A, B, C, observer_poles):
        """
        A, B, C: System matrices
        observer_poles: Desired locations for the error dynamics (A - LC)
        """
        self.A = A
        self.B = B
        self.C = C
        
        # Calculate Observer Gain L
        # We use the dual system (A.T, C.T) to use the same pole placement math
        res = place_poles(self.A.T, self.C.T, observer_poles)
        self.L = res.gain_matrix.T
        
        # Initialize estimated state (can start with a guess/error)
        self.x_hat = np.zeros((A.shape[0],))

    def update(self, u, y, dt):
        """
        u: Control input applied to the plant
        y: Measured output from the plant (sensors)
        dt: Time step
        """
        # Calculate the innovation (prediction error)
        # y_hat is what the observer THINKS the sensor should read
        y_hat = self.C @ self.x_hat
        innovation = y - y_hat
        
        # Observer Dynamics: dx_hat/dt = A*x_hat + B*u + L*(y - y_hat)
        dx_hat = self.A @ self.x_hat + self.B.flatten() * u + (self.L @ innovation).flatten()
        
        # Simple Euler integration to update the estimate
        self.x_hat += dx_hat * dt
        
        return self.x_hat

    def reset(self, initial_guess):
        self.x_hat = initial_guess

class KalmanFilter:
    def __init__(self, A, B, C, Q_v, R_w):
        """
        A, B, C: System matrices
        Q_v: Process noise covariance matrix
        R_w: Measurement noise covariance matrix
        """
        self.A = A
        self.B = B
        self.C = C
        
        # Solve the Continuous Algebraic Riccati Equation (CARE) for the observer
        # We pass A.T and C.T because the estimator is the mathematical "dual" of the LQR controller
        P = solve_continuous_are(self.A.T, self.C.T, Q_v, R_w)
        
        # Calculate optimal Kalman Gain L (often denoted as K_f)
        # L = P * C^T * R_w^-1
        self.L = P @ self.C.T @ np.linalg.inv(R_w)
        
        # Initialize estimated state
        self.x_hat = np.zeros((A.shape[0],))

    def update(self, u, y, dt):
        """
        u: Control input
        y: Measured output
        dt: Time step
        """
        # Calculate what we *think* the sensor should see
        y_hat = self.C @ self.x_hat
        innovation = y - y_hat
        
        # dx_hat/dt = A*x_hat + B*u + L*(y - y_hat)
        dx_hat = self.A @ self.x_hat + self.B.flatten() * u + (self.L @ innovation).flatten()
        
        self.x_hat += dx_hat * dt
        
        return self.x_hat

    def reset(self, initial_guess):
        self.x_hat = initial_guess


class LQIController:
    def __init__(self, A, B, q_pos, q_ang, q_int, r_weight):
        # 1. Augment the system to include the integral of position error
        # x_aug = [p, p_dot, theta, theta_dot, integral_error]
        
        # C_p isolates the position state (p)
        C_p = np.array([[1.0, 0.0, 0.0, 0.0]])
        
        # Create 5x5 Augmented A matrix
        self.A_aug = np.zeros((5, 5))
        self.A_aug[0:4, 0:4] = A
        self.A_aug[4, 0:4] = C_p  # The derivative of the integral state is position
        
        # Create 5x1 Augmented B matrix
        self.B_aug = np.zeros((5, 1))
        self.B_aug[0:4, :] = B
        
        # 2. Setup Cost Matrices (Now 5x5)
        # We add q_int to heavily penalize steady-state error
        Q_aug = np.diag([q_pos, 0.1, q_ang, 0.1, q_int])
        R = np.array([[r_weight]])
        
        # 3. Solve CARE for the augmented system
        P = la.solve_continuous_are(self.A_aug, self.B_aug, Q_aug, R)
        K_aug = np.linalg.inv(R) @ self.B_aug.T @ P
        
        # Split the gains back out
        self.K = K_aug[0, 0:4]  # Standard state gains
        self.K_i = K_aug[0, 4]  # Integral gain
        
        # Initialize the running memory of the error
        self.integral_error = 0.0

    def compute(self, setpoint, current_state, dt):
        # Accumulate position error over time
        error = current_state[0] - setpoint
        self.integral_error += error * dt
        
        # Optimal Control Law with Integral Action
        u = -np.dot(self.K, current_state) - (self.K_i * self.integral_error)
        return float(u)
    
class TrajectoryPlanner:
    def __init__(self, p_start, p_end, duration):
        self.p_0 = p_start
        self.p_f = p_end
        self.T = duration

    def get_state(self, t):
        # Before the move starts
        if t <= 0:
            return self.p_0, 0.0, 0.0
        # After the move finishes
        if t >= self.T:
            return self.p_f, 0.0, 0.0
        
        # Normalized time (0.0 to 1.0)
        tau = t / self.T
        tau2 = tau**2
        tau3 = tau**3
        tau4 = tau**4
        tau5 = tau**5
        
        dp = self.p_f - self.p_0
        
        # Quintic Polynomial Equations
        p = self.p_0 + dp * (10*tau3 - 15*tau4 + 6*tau5)
        v = (dp / self.T) * (30*tau2 - 60*tau3 + 30*tau4)
        a = (dp / (self.T**2)) * (60*tau - 180*tau2 + 120*tau3)
        
        return p, v, a