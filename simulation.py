"""Streamlit-independent simulation runtime for the cart-pole application.

The UI builds a plant, controller, optional estimator/planner/safety filter, and
an immutable :class:`SimulationConfig`.  This module owns the runtime loop and
returns arrays plus derived metrics in :class:`SimulationResult`.

Keeping Streamlit out of this file makes the exact same simulation reusable by
unit tests, offline benchmarks, notebooks, and the interactive app.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


MEASUREMENT_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],  # cart position
        [0.0, 0.0, 1.0, 0.0],  # pole angle
    ]
)


class ControllerMode(str, Enum):
    """Controller call signature used by the runtime adapter."""

    STATE = "state"
    POSITION_PID = "position_pid"
    STATE_WITH_DT = "state_with_dt"


class DisturbanceProfile(str, Enum):
    """Supported exogenous-force profiles."""

    NONE = "none"
    IMPULSE = "impulse"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class SimulationConfig:
    """All scalar/runtime choices needed for one reproducible simulation."""

    total_time: float
    target_position: float
    track_limit: float
    controller_mode: ControllerMode = ControllerMode.STATE
    dt: float = 0.02

    use_trajectory: bool = False
    move_start: float = 0.0

    feedback_linearization: bool = False

    actuator_saturation: bool = False
    max_force: float = np.inf

    sensor_noise_std_deg: float = 0.0
    random_seed: int | None = None

    disturbance_profile: DisturbanceProfile = DisturbanceProfile.NONE
    disturbance_magnitude: float = 0.0
    disturbance_start: float = 0.0
    impulse_duration: float = 0.1

    target_tolerance: float = 0.05
    angle_stable_deg: float = 1.0
    linearization_limit_deg: float = 20.0

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.total_time < self.dt:
            raise ValueError("total_time must contain at least one control step")
        if self.track_limit <= 0.0:
            raise ValueError("track_limit must be positive")
        if self.actuator_saturation and (
            not np.isfinite(self.max_force) or self.max_force <= 0.0
        ):
            raise ValueError(
                "max_force must be finite and positive when saturation is enabled"
            )
        if self.sensor_noise_std_deg < 0.0:
            raise ValueError("sensor_noise_std_deg cannot be negative")
        if self.impulse_duration < 0.0:
            raise ValueError("impulse_duration cannot be negative")
        if self.target_tolerance < 0.0:
            raise ValueError("target_tolerance cannot be negative")
        if self.angle_stable_deg < 0.0:
            raise ValueError("angle_stable_deg cannot be negative")

    @property
    def steps(self) -> int:
        return int(self.total_time / self.dt)


@dataclass(frozen=True)
class SimulationMetrics:
    """Scalar diagnostics derived from one simulation."""

    final_position: float
    final_angle_deg: float
    position_stable: bool
    angle_stable: bool
    rms_tracking_error: float
    steady_state_error: float
    total_energy: float
    peak_force: float
    saturation_percentage: float
    max_angle_deg: float
    safety_intervention_percentage: float
    safety_max_deviation: float


@dataclass(frozen=True)
class SimulationResult:
    """Time-series output and diagnostics returned by :func:`run_simulation`."""

    dt: float
    states: np.ndarray
    estimates: np.ndarray
    references: np.ndarray
    inputs: np.ndarray
    gpmpc_snapshots: tuple[tuple[Any, ...], ...]
    safety_filter_log: tuple[tuple[int, float, float], ...]
    terminated: bool
    termination_reason: str | None
    metrics: SimulationMetrics

    @property
    def time(self) -> np.ndarray:
        return np.arange(len(self.states), dtype=float) * self.dt

    @property
    def position(self) -> np.ndarray:
        return self.states[:, 0]

    @property
    def angle_deg(self) -> np.ndarray:
        return np.degrees(self.states[:, 2])

    @property
    def angular_velocity_deg(self) -> np.ndarray:
        return np.degrees(self.states[:, 3])

    @property
    def cart_velocity(self) -> np.ndarray:
        return self.states[:, 1]


class SimulationError(RuntimeError):
    """Adds runtime stage and timestep context to a simulation failure."""

    def __init__(self, step: int, time: float, stage: str, cause: Exception):
        self.step = step
        self.time = time
        self.stage = stage
        self.cause = cause
        super().__init__(
            f"Simulation failed during {stage} at step {step} "
            f"(t={time:.3f} s): {cause}"
        )


def _compute_feedback(
    controller: Any,
    mode: ControllerMode,
    reference: float,
    feedback_state: np.ndarray,
    dt: float,
) -> float:
    if mode is ControllerMode.POSITION_PID:
        value = controller.compute(reference, float(feedback_state[0]), dt)
    elif mode is ControllerMode.STATE_WITH_DT:
        value = controller.compute(reference, feedback_state, dt)
    else:
        value = controller.compute(reference, feedback_state)

    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"controller returned a non-finite input: {value}")
    return value


def _disturbance_force(config: SimulationConfig, time: float) -> float:
    if (
        config.disturbance_profile is DisturbanceProfile.IMPULSE
        and config.disturbance_start
        <= time
        <= config.disturbance_start + config.impulse_duration
    ):
        return config.disturbance_magnitude
    if (
        config.disturbance_profile is DisturbanceProfile.CONTINUOUS
        and time >= config.disturbance_start
    ):
        return config.disturbance_magnitude
    return 0.0


def run_simulation(
    *,
    config: SimulationConfig,
    plant: Any,
    controller: Any,
    initial_state: np.ndarray,
    estimator: Any | None = None,
    planner: Any | None = None,
    safety_filter: Any | None = None,
    measurement_matrix: np.ndarray = MEASUREMENT_MATRIX,
) -> SimulationResult:
    """Run one closed-loop simulation without any Streamlit dependency.

    Parameters are injected rather than constructed here.  This keeps controller
    design, model training, and UI caching separate from the deterministic
    runtime orchestration.
    """

    current_state = np.asarray(initial_state, dtype=float).copy()
    if current_state.shape != (4,):
        raise ValueError("initial_state must have shape (4,)")
    if not np.all(np.isfinite(current_state)):
        raise ValueError("initial_state must contain only finite values")

    measurement_matrix = np.asarray(measurement_matrix, dtype=float)
    if measurement_matrix.shape != (2, 4):
        raise ValueError("measurement_matrix must have shape (2, 4)")
    if config.use_trajectory and planner is None:
        raise ValueError("planner is required when use_trajectory is enabled")

    rng = np.random.default_rng(config.random_seed)
    states: list[np.ndarray] = []
    estimates: list[np.ndarray] = []
    references: list[float] = []
    inputs: list[float] = []
    gpmpc_snapshots: list[tuple[Any, ...]] = []
    safety_filter_log: list[tuple[int, float, float]] = []

    total_energy = 0.0
    tracking_error_sq = 0.0
    saturation_steps = 0
    terminated = False
    termination_reason: str | None = None
    previous_measurement = measurement_matrix @ current_state

    for step_index in range(config.steps):
        time = step_index * config.dt
        stage = "state recording"
        try:
            states.append(current_state.copy())
            cart_position = float(current_state[0])

            stage = "reference generation"
            if config.use_trajectory:
                position_reference, _, acceleration_reference = planner.get_state(
                    time - config.move_start
                )
                feedforward = (
                    float(plant.m_c) + float(plant.m_p)
                ) * acceleration_reference
            else:
                position_reference = config.target_position
                feedforward = 0.0
            position_reference = float(position_reference)
            references.append(position_reference)

            stage = "measurement"
            measurement = measurement_matrix @ current_state
            if config.sensor_noise_std_deg > 0.0:
                angle_noise_deg = rng.normal(0.0, config.sensor_noise_std_deg)
                measurement[1] += np.radians(angle_noise_deg)

            stage = "state estimation"
            if estimator is not None:
                previous_input = inputs[-1] if inputs else 0.0
                estimated_state = np.asarray(
                    estimator.update(previous_input, measurement, config.dt),
                    dtype=float,
                )
                feedback_state = estimated_state.copy()
            else:
                feedback_state = np.zeros(4, dtype=float)
                feedback_state[0] = measurement[0]
                feedback_state[2] = measurement[1]
                feedback_state[1] = (
                    measurement[0] - previous_measurement[0]
                ) / config.dt
                feedback_state[3] = (
                    measurement[1] - previous_measurement[1]
                ) / config.dt
            if feedback_state.shape != (4,) or not np.all(
                np.isfinite(feedback_state)
            ):
                raise ValueError("state estimate must be a finite vector of shape (4,)")
            estimates.append(feedback_state.copy())
            previous_measurement = measurement.copy()

            stage = "track-limit check"
            if abs(cart_position) > config.track_limit:
                terminated = True
                termination_reason = "track_limit"
                break

            stage = "controller"
            control = _compute_feedback(
                controller,
                config.controller_mode,
                position_reference,
                feedback_state,
                config.dt,
            )
            control += feedforward

            last_plan = getattr(controller, "last_plan", None)
            last_position_sigma = getattr(controller, "last_pos_sigma", None)
            last_tightening = getattr(controller, "last_tight", None)
            if (
                last_plan is not None
                and last_position_sigma is not None
                and last_tightening is not None
            ):
                gpmpc_snapshots.append(
                    (
                        step_index,
                        float(feedback_state[0]),
                        last_plan.copy(),
                        last_position_sigma.copy(),
                        last_tightening.copy(),
                    )
                )

            stage = "feedback linearization"
            if config.feedback_linearization:
                theta, omega = feedback_state[2], feedback_state[3]
                control += (
                    -float(plant.m_p)
                    * float(plant.l)
                    * omega**2
                    * np.sin(theta)
                )
                control -= float(plant.m_p) * float(plant.g) * (
                    theta - np.sin(theta) * np.cos(theta)
                )

            stage = "safety filter"
            if safety_filter is not None:
                proposed_control = float(control)
                control = float(safety_filter.filter(feedback_state, proposed_control))
                safety_filter_log.append(
                    (step_index, proposed_control, float(control))
                )

            stage = "actuator saturation"
            if config.actuator_saturation:
                clipped_control = float(
                    np.clip(control, -config.max_force, config.max_force)
                )
                if abs(clipped_control) < abs(control):
                    saturation_steps += 1
                control = clipped_control
            if not np.isfinite(control):
                raise ValueError(f"control path produced a non-finite input: {control}")

            total_energy += control**2 * config.dt
            tracking_error_sq += (
                feedback_state[0] - position_reference
            ) ** 2 * config.dt
            inputs.append(float(control))

            stage = "plant integration"
            effective_control = control + _disturbance_force(config, time)
            next_state = np.asarray(
                plant.step(current_state, effective_control, config.dt),
                dtype=float,
            )
            if next_state.shape != (4,) or not np.all(np.isfinite(next_state)):
                raise ValueError("plant returned a non-finite state or invalid shape")
            current_state = next_state
        except SimulationError:
            raise
        except Exception as exc:
            raise SimulationError(step_index, time, stage, exc) from exc

    state_array = np.asarray(states, dtype=float)
    estimate_array = np.asarray(estimates, dtype=float)
    reference_array = np.asarray(references, dtype=float)
    input_array = np.asarray(inputs, dtype=float)

    final_position = float(state_array[-1, 0])
    final_angle_deg = float(np.degrees(state_array[-1, 2]))
    steady_state_error = abs(final_position - config.target_position)
    rms_tracking_error = float(
        np.sqrt(tracking_error_sq / max(len(state_array) * config.dt, 1e-9))
    )
    peak_force = float(np.max(np.abs(input_array))) if len(input_array) else 0.0
    saturation_percentage = 100.0 * saturation_steps / max(len(input_array), 1)
    max_angle_deg = float(np.max(np.abs(np.degrees(state_array[:, 2]))))

    if safety_filter_log:
        safety_deviations = np.asarray(
            [abs(proposed - certified) for _, proposed, certified in safety_filter_log]
        )
        safety_intervention_percentage = 100.0 * float(
            np.mean(safety_deviations > 0.01)
        )
        safety_max_deviation = float(np.max(safety_deviations))
    else:
        safety_intervention_percentage = 0.0
        safety_max_deviation = 0.0

    metrics = SimulationMetrics(
        final_position=final_position,
        final_angle_deg=final_angle_deg,
        position_stable=steady_state_error <= config.target_tolerance
        and not terminated,
        angle_stable=abs(final_angle_deg) <= config.angle_stable_deg,
        rms_tracking_error=rms_tracking_error,
        steady_state_error=steady_state_error,
        total_energy=float(total_energy),
        peak_force=peak_force,
        saturation_percentage=float(saturation_percentage),
        max_angle_deg=max_angle_deg,
        safety_intervention_percentage=safety_intervention_percentage,
        safety_max_deviation=safety_max_deviation,
    )

    return SimulationResult(
        dt=config.dt,
        states=state_array,
        estimates=estimate_array,
        references=reference_array,
        inputs=input_array,
        gpmpc_snapshots=tuple(gpmpc_snapshots),
        safety_filter_log=tuple(safety_filter_log),
        terminated=terminated,
        termination_reason=termination_reason,
        metrics=metrics,
    )
