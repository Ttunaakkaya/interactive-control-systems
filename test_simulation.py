"""Fast regression tests for the Streamlit-independent simulation runtime."""

import numpy as np
import pytest

from controller import LQRController
from plant import CartPolePlant
from simulation import (
    ControllerMode,
    DisturbanceProfile,
    SimulationConfig,
    SimulationError,
    run_simulation,
)


class _StaticPlant:
    m_c = 1.0
    m_p = 0.1
    l = 0.5
    g = 9.81

    def __init__(self):
        self.applied_inputs = []

    def step(self, state, control, dt):
        self.applied_inputs.append(float(control))
        return np.asarray(state, dtype=float).copy()


class _ConstantStateController:
    def __init__(self, value):
        self.value = value

    def compute(self, reference, state):
        return self.value


def _legacy_state_loop(config, plant, controller, initial_state):
    """Pre-refactor deterministic loop used as an exact parity oracle."""

    current_state = np.asarray(initial_state, dtype=float).copy()
    previous_measurement = np.array([current_state[0], current_state[2]])
    states, estimates, references, inputs = [], [], [], []

    for _ in range(config.steps):
        states.append(current_state.copy())
        reference = config.target_position
        references.append(reference)

        measurement = np.array([current_state[0], current_state[2]])
        feedback_state = np.zeros(4)
        feedback_state[0] = measurement[0]
        feedback_state[2] = measurement[1]
        feedback_state[1] = (
            measurement[0] - previous_measurement[0]
        ) / config.dt
        feedback_state[3] = (
            measurement[1] - previous_measurement[1]
        ) / config.dt
        estimates.append(feedback_state.copy())
        previous_measurement = measurement.copy()

        if abs(current_state[0]) > config.track_limit:
            break

        control = float(controller.compute(reference, feedback_state))
        inputs.append(control)
        current_state = plant.step(current_state, control, config.dt)

    return tuple(np.asarray(values) for values in (states, estimates, references, inputs))


def test_extracted_runtime_matches_pre_refactor_loop_exactly():
    config = SimulationConfig(
        total_time=1.0,
        target_position=0.3,
        track_limit=2.8,
        controller_mode=ControllerMode.STATE,
    )
    initial_state = np.array([0.0, 0.0, np.radians(5.7), 0.0])

    legacy_plant = CartPolePlant()
    legacy_controller = LQRController(
        legacy_plant.A, legacy_plant.B, q_pos=100.0, q_ang=10.0, r_weight=1.0
    )
    expected = _legacy_state_loop(
        config, legacy_plant, legacy_controller, initial_state
    )

    runtime_plant = CartPolePlant()
    runtime_controller = LQRController(
        runtime_plant.A, runtime_plant.B, q_pos=100.0, q_ang=10.0, r_weight=1.0
    )
    actual = run_simulation(
        config=config,
        plant=runtime_plant,
        controller=runtime_controller,
        initial_state=initial_state,
    )

    for actual_values, expected_values in zip(
        (actual.states, actual.estimates, actual.references, actual.inputs),
        expected,
    ):
        assert np.array_equal(actual_values, expected_values)


def test_safety_filter_runs_before_actuator_saturation():
    plant = _StaticPlant()

    class SafetyFilter:
        def filter(self, state, proposed):
            assert proposed == 10.0
            return 7.0

    result = run_simulation(
        config=SimulationConfig(
            total_time=0.02,
            target_position=0.0,
            track_limit=2.0,
            actuator_saturation=True,
            max_force=5.0,
        ),
        plant=plant,
        controller=_ConstantStateController(10.0),
        initial_state=np.zeros(4),
        safety_filter=SafetyFilter(),
    )

    assert result.inputs.tolist() == [5.0]
    assert result.safety_filter_log == ((0, 10.0, 7.0),)
    assert result.metrics.saturation_percentage == 100.0
    assert plant.applied_inputs == [5.0]


def test_impulse_disturbance_is_applied_after_command_logging():
    plant = _StaticPlant()
    result = run_simulation(
        config=SimulationConfig(
            total_time=0.06,
            target_position=0.0,
            track_limit=2.0,
            disturbance_profile=DisturbanceProfile.IMPULSE,
            disturbance_magnitude=3.0,
            disturbance_start=0.02,
            impulse_duration=0.02,
        ),
        plant=plant,
        controller=_ConstantStateController(2.0),
        initial_state=np.zeros(4),
    )

    assert result.inputs.tolist() == [2.0, 2.0, 2.0]
    assert plant.applied_inputs == [2.0, 5.0, 5.0]


def test_seeded_sensor_noise_is_reproducible():
    config = SimulationConfig(
        total_time=0.1,
        target_position=0.0,
        track_limit=2.0,
        sensor_noise_std_deg=0.5,
        random_seed=42,
    )

    first = run_simulation(
        config=config,
        plant=_StaticPlant(),
        controller=_ConstantStateController(0.0),
        initial_state=np.zeros(4),
    )
    second = run_simulation(
        config=config,
        plant=_StaticPlant(),
        controller=_ConstantStateController(0.0),
        initial_state=np.zeros(4),
    )

    assert np.array_equal(first.estimates, second.estimates)
    assert np.any(first.estimates[:, 2] != 0.0)


def test_out_of_bounds_initial_state_terminates_without_control():
    result = run_simulation(
        config=SimulationConfig(
            total_time=0.1,
            target_position=0.0,
            track_limit=1.0,
        ),
        plant=_StaticPlant(),
        controller=_ConstantStateController(0.0),
        initial_state=np.array([1.1, 0.0, 0.0, 0.0]),
    )

    assert result.terminated
    assert result.termination_reason == "track_limit"
    assert result.inputs.size == 0
    assert not result.metrics.position_stable


def test_runtime_error_identifies_stage_and_step():
    with pytest.raises(SimulationError, match="controller.*step 0") as error:
        run_simulation(
            config=SimulationConfig(
                total_time=0.02,
                target_position=0.0,
                track_limit=1.0,
            ),
            plant=_StaticPlant(),
            controller=_ConstantStateController(np.nan),
            initial_state=np.zeros(4),
        )

    assert error.value.stage == "controller"
    assert isinstance(error.value.cause, ValueError)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"dt": 0.0}, "dt must be positive"),
        ({"track_limit": 0.0}, "track_limit must be positive"),
        ({"sensor_noise_std_deg": -1.0}, "cannot be negative"),
    ],
)
def test_config_rejects_invalid_runtime_values(kwargs, message):
    values = dict(total_time=1.0, target_position=0.0, track_limit=2.0)
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        SimulationConfig(**values)
