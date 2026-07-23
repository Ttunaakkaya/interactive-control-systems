"""Mathematical invariant and optimizer smoke tests for controller.py."""

import numpy as np

from controller import (
    GPMPCController,
    KalmanFilter,
    LQIController,
    LQRController,
    LuenbergerObserver,
    MPCController,
    PIDController,
    RecklessPolicy,
    StateSpaceController,
    TrajectoryPlanner,
)
from plant import CartPolePlant


def _max_pole_matching_error(actual, desired):
    remaining = list(actual)
    errors = []
    for target in desired:
        closest = int(np.argmin([abs(value - target) for value in remaining]))
        errors.append(abs(remaining.pop(closest) - target))
    return max(errors)


def test_pid_avoids_first_step_derivative_kick_and_resets():
    controller = PIDController(kp=2.0, ki=1.0, kd=3.0)
    assert controller.compute(1.0, 0.0, 0.02) == 2.02
    assert np.isfinite(controller.compute(1.0, 0.1, 0.02))

    controller.reset()
    assert controller._first_call
    assert controller.integral_error == 0.0


def test_pole_placement_matches_requested_poles_and_dc_tracking():
    plant = CartPolePlant()
    controller = StateSpaceController(plant.A, plant.B, zeta=0.7, wn=3.5)
    actual_poles = np.linalg.eigvals(plant.A - plant.B @ controller.K)

    assert (
        _max_pole_matching_error(actual_poles, controller.desired_poles) < 1e-7
    )
    steady_state = np.linalg.solve(
        -(plant.A - plant.B @ controller.K),
        plant.B.flatten() * controller.Nr,
    )
    assert abs(steady_state[0] - 1.0) < 1e-10


def test_lqr_closed_loop_is_stable_and_tracks_dc_reference():
    plant = CartPolePlant()
    controller = LQRController(
        plant.A, plant.B, q_pos=100.0, q_ang=10.0, r_weight=1.0
    )
    closed_loop = plant.A - plant.B @ controller.K

    assert np.max(np.real(np.linalg.eigvals(closed_loop))) < 0.0
    steady_state = np.linalg.solve(
        -closed_loop, plant.B.flatten() * controller.Nr
    )
    assert abs(steady_state[0] - 1.0) < 1e-10


def test_lqi_augmented_closed_loop_is_stable():
    plant = CartPolePlant()
    controller = LQIController(plant.A, plant.B)
    augmented_a = np.zeros((5, 5))
    augmented_a[:4, :4] = plant.A
    augmented_a[4, 0] = 1.0
    augmented_b = np.zeros((5, 1))
    augmented_b[:4] = plant.B
    augmented_k = np.r_[controller.K, controller.K_i][None, :]

    poles = np.linalg.eigvals(augmented_a - augmented_b @ augmented_k)
    assert np.max(np.real(poles)) < 0.0


def test_luenberger_error_dynamics_match_requested_poles():
    plant = CartPolePlant()
    desired = np.array([-8.0, -9.0, -10.0, -11.0])
    observer = LuenbergerObserver(plant.A, plant.B, plant.C, desired)
    actual = np.linalg.eigvals(plant.A - observer.L @ plant.C)

    assert _max_pole_matching_error(actual, desired) < 1e-7


def test_kalman_discrete_error_dynamics_are_stable():
    plant = CartPolePlant()
    estimator = KalmanFilter(
        plant.A,
        plant.B,
        plant.C,
        np.eye(4) * 0.1,
        np.eye(2),
        dt=0.02,
    )
    error_dynamics = (
        np.eye(4) - estimator.L @ estimator.Cd
    ) @ estimator.Ad

    assert np.max(np.abs(np.linalg.eigvals(error_dynamics))) < 1.0
    assert np.all(np.isfinite(estimator.L))


def test_quintic_trajectory_satisfies_boundary_conditions():
    planner = TrajectoryPlanner(-0.4, 1.2, 2.5)

    assert np.allclose(planner.get_state(0.0), (-0.4, 0.0, 0.0))
    assert np.allclose(planner.get_state(2.5), (1.2, 0.0, 0.0))
    position, velocity, acceleration = planner.get_state(1.25)
    assert position == 0.4
    assert velocity > 0.0
    assert abs(acceleration) < 1e-12


def test_mpc_respects_input_bound_dynamics_and_warm_start():
    plant = CartPolePlant()
    controller = MPCController(
        plant, horizon=12, max_force=15.0, track_limit=2.0
    )
    state = np.array([0.0, 0.0, 0.05, 0.0])

    first_input = controller.compute(0.0, state)
    assert np.isfinite(first_input)
    assert abs(first_input) <= 15.0 + 1e-6
    assert controller.last_plan.shape == (4, 13)

    residual = max(
        np.linalg.norm(
            controller.last_plan[:, step + 1]
            - np.array(
                plant.F(
                    controller.last_plan[:, step],
                    [controller._prev_U[step, 0]],
                )
            ).ravel(),
            np.inf,
        )
        for step in range(12)
    )
    assert residual < 1e-6

    next_state = plant.step(state, first_input, 0.02)
    assert np.isfinite(controller.compute(0.0, next_state))


def test_gp_mpc_exposes_nonnegative_chance_tightening():
    class FakeGP:
        def predict(self, states, inputs):
            samples = len(states)
            return np.zeros((samples, 2)), np.full((samples, 2), 0.02)

    plant = CartPolePlant()
    controller = GPMPCController(
        plant,
        FakeGP(),
        use_gp=True,
        horizon=12,
        max_force=15.0,
        track_limit=2.0,
        constraint_mode="chance",
        kappa=2.0,
    )
    control = controller.compute(0.0, np.array([0.0, 0.0, 0.05, 0.0]))

    assert np.isfinite(control)
    assert abs(control) <= 15.0 + 1e-6
    assert controller.last_plan.shape == (4, 13)
    assert controller.last_tight.shape == (13,)
    assert np.all(controller.last_tight >= 0.0)
    assert np.max(controller.last_tight) > 0.0


def test_reckless_policy_still_respects_its_own_actuator_limit():
    plant = CartPolePlant()
    controller = RecklessPolicy(plant, target_beyond_wall=3.3, u_max=15.0)

    control = controller.compute(0.0, np.zeros(4))
    assert np.isfinite(control)
    assert abs(control) <= 15.0
