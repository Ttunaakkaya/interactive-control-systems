import numpy as np
import mujoco
import control as ct


class CartPolePlant:
    def __init__(self, m_c=1.0, m_p=0.1, l=0.5, g=9.81, d=0.0):
        """
        CartPole plant: MuJoCo nonlinear physics + linearised state-space model.

        Parameters
        ----------
        m_c : cart mass (kg)
        m_p : pendulum point-mass (kg)
        l   : pendulum half-length (m)  [tip is at 2·l in MuJoCo geom]
        g   : gravitational acceleration (m/s²)
        d   : viscous damping in the linearised A matrix ONLY.
              Default 0.0 → linear model matches the frictionless MuJoCo plant.
              To add physical friction call set_friction() after construction.
        """
        self.m_c = m_c
        self.m_p = m_p
        self.l   = l
        self.g   = g
        self.d   = d

        Mt = m_c + m_p

        # ------------------------------------------------------------------ #
        # Linearised model — Taylor expansion around θ=0, ẋ=0              #
        # State:  x = [cart_pos, cart_vel, pole_angle, pole_angvel]          #
        # Input:  u = horizontal force on cart (N)                           #
        # ------------------------------------------------------------------ #
        self.A = np.array([
            [0.0,          1.0,                0.0,  0.0],
            [0.0, -d / Mt,    -(m_p * g) / Mt, 0.0],
            [0.0,          0.0,                0.0,  1.0],
            [0.0,  d / (Mt*l),  (Mt * g) / (Mt*l),  0.0]
        ])
        self.B = np.array([[0.0], [1.0/Mt], [0.0], [-1.0/(Mt*l)]])
        self.C = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 0.0]])
        self.D = np.zeros((2, 1))
        self.linear_sys = ct.ss(self.A, self.B, self.C, self.D)

        # ------------------------------------------------------------------ #
        # MuJoCo nonlinear plant — RK4 at 2 ms internal timestep             #
        # Joints start frictionless; call set_friction() to add rail physics. #
        # ------------------------------------------------------------------ #
        xml = f"""
        <mujoco model="cart-pole">
            <option timestep="0.002" gravity="0 0 {-g}" integrator="RK4"/>
            <worldbody>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                <geom type="plane" size="10 1 0.1" rgba=".9 .9 .9 1"
                      contype="0" conaffinity="0"/>
                <body name="cart" pos="0 0 0.2">
                    <joint name="slider" type="slide" axis="1 0 0"
                           frictionloss="0.0" damping="0.0"/>
                    <geom type="box" size="0.2 0.1 0.1" mass="{m_c}"
                          rgba="0.8 0.6 0.1 1"/>
                    <body name="pole" pos="0 0 0">
                        <joint name="hinge" type="hinge" axis="0 1 0"
                               frictionloss="0.0" damping="0.0"/>
                        <geom type="capsule" fromto="0 0 0 0 0 {2*l}"
                              size="0.02" mass="{m_p}"
                              rgba="0.8 0.2 0.2 1"
                              contype="0" conaffinity="0"/>
                    </body>
                </body>
            </worldbody>
            <actuator>
                <motor joint="slider" name="cart_motor"
                       gear="1" ctrllimited="false"/>
            </actuator>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data  = mujoco.MjData(self.model)

    def set_friction(self, cart_frictionloss: float = 0.0,
                     pole_frictionloss: float = 0.0,
                     cart_damping: float = 0.0):
        """
        Set friction/damping directly on MuJoCo joints (correct physics layer).

        Parameters
        ----------
        cart_frictionloss : Coulomb friction on slider joint (N).
                            MuJoCo constraint solver enforces this as a
                            max static/kinetic friction force on the cart.
        pole_frictionloss : Coulomb friction on hinge joint (N·m).
        cart_damping      : Viscous damping on slider DOF (N·s/m).
                            Corresponds to 'd' in the A matrix; setting this
                            equal to 'd' closes the linear–nonlinear model gap.
        """
        self.model.joint("slider").frictionloss[0] = cart_frictionloss
        self.model.joint("hinge").frictionloss[0]  = pole_frictionloss
        self.model.dof_damping[0]                  = cart_damping

    def step(self, state: np.ndarray, u: float, dt: float) -> np.ndarray:
        """
        Advance MuJoCo by one control timestep dt (must be multiple of 0.002 s).

        Returns next state [cart_pos, cart_vel, pole_angle, pole_angvel].
        """
        self.data.qpos[0] = state[0]
        self.data.qpos[1] = state[2]
        self.data.qvel[0] = state[1]
        self.data.qvel[1] = state[3]
        self.data.ctrl[0] = u
        n_steps = int(round(dt / self.model.opt.timestep))
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)
        return np.array([self.data.qpos[0], self.data.qvel[0],
                         self.data.qpos[1], self.data.qvel[1]])