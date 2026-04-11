import numpy as np
import control as ct
from scipy.integrate import solve_ivp

class CartPolePlant:
    def __init__(self, m_c=1.0, m_p=0.1, l=0.5, g=9.81, d=0.1):
        """
        Industry-standard Cart-Pole parameters.
        m_c: mass of cart
        m_p: mass of pendulum (point mass at the tip)
        l: length to the pendulum's center of mass
        """
        self.m_c = m_c
        self.m_p = m_p
        self.l = l
        self.g = g
        self.d = d

        # --- 1. THE BRAIN: python-control State-Space Representation ---
        # These are the linearized matrices (around the upright position theta = 0)
        # You will use these exact matrices for Pole Placement and LQR later.
        
        Mt = self.m_c + self.m_p
        
        # A matrix (System Dynamics)
        self.A = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -self.d/Mt, -(self.m_p*self.g)/Mt, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, self.d/(Mt*self.l), (Mt*self.g)/(Mt*self.l), 0.0]
        ])

        # B matrix (Input/Actuator matrix)
        self.B = np.array([
            [0.0],
            [1.0/Mt],
            [0.0],
            [-1.0/(Mt*self.l)]
        ])

        # C matrix (Output matrix - what sensors can measure)
        # Let's assume we can measure position (p) and angle (theta)
        self.C = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ])

        # D matrix (Feedforward matrix - usually zero in mechanical systems)
        self.D = np.array([
            [0.0], 
            [0.0]
        ])

        # Initialize the python-control linear system object
        self.linear_sys = ct.ss(self.A, self.B, self.C, self.D)

    # --- 2. THE MUSCLE: SciPy Non-Linear Dynamics ---
    def nonlinear_dynamics(self, t, state, u):
        """
        The absolute non-linear equations for the physics engine.
        SciPy requires the function signature to be f(t, y, args)
        """
        p, p_dot, theta, theta_dot = state
        s = np.sin(theta)
        c = np.cos(theta)
        Mt = self.m_c + self.m_p

        temp = (u + self.m_p * self.l * theta_dot**2 * s - self.d * p_dot) / Mt
        theta_acc = (self.g * s - c * temp) / (self.l * (4.0/3.0 - self.m_p * c**2 / Mt))
        p_acc = temp - (self.m_p  * self.l * theta_acc * c) / Mt

        return [p_dot, p_acc, theta_dot, theta_acc]

    def step(self, state, u, dt):
        """
        Advances the physics engine by dt using SciPy's professional RK45 solver.
        """
        # solve_ivp integrates the non-linear dynamics from t=0 to t=dt
        sol = solve_ivp(
            fun=self.nonlinear_dynamics,
            t_span=[0.0, dt],
            y0=state,
            args=(u,),       # Pass the control force 'u' into the dynamics
            method='RK45'    # Adaptive Runge-Kutta 4th/5th order
        )
        
        # Return the very last state calculated in this time step
        return sol.y[:, -1]