"""Reproducible, Streamlit-independent controller benchmarks.

The benchmark deliberately measures simulated control performance, not wall
clock speed. Wall clock timings depend on the machine and background load;
closed-loop trajectories and metrics are the reproducible scientific result.

Examples
--------
Run the standard protocol and write JSON/CSV artifacts::

    python benchmark.py

Run the short protocol with selected controllers and seeds::

    python benchmark.py --suite quick --controllers lqr lqi --seeds 7 11
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from importlib import metadata
import io
import json
import math
from pathlib import Path
import platform
from typing import Any, Sequence

import numpy as np

from controller import (
    LQIController,
    LQRController,
    MPCController,
    PIDController,
    StateSpaceController,
)
from plant import CartPolePlant
from simulation import (
    ControllerMode,
    DisturbanceProfile,
    SimulationConfig,
    SimulationResult,
    run_simulation,
)


BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_PROTOCOL_VERSION = "cartpole-benchmark-v1"
DEFAULT_CONTROLLERS = ("pole", "lqr", "lqi")
DEFAULT_SEEDS = (0, 1, 2)
IMPLEMENTATION_FILES = (
    "benchmark.py",
    "simulation.py",
    "plant.py",
    "controller.py",
)


def _validate_identifier(value: str, field_name: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_"
    if not value or any(character not in allowed for character in value):
        raise ValueError(
            f"{field_name} must contain only lowercase letters, digits, and underscores"
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _implementation_sha256() -> str:
    """Fingerprint runtime source with line endings normalized by read_text."""

    source_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for filename in IMPLEMENTATION_FILES:
        digest.update(filename.encode("ascii"))
        source = (source_root / filename).read_text(encoding="utf-8")
        digest.update(source.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class BenchmarkScenario:
    """A completely specified physical and stochastic benchmark scenario."""

    scenario_id: str
    description: str
    total_time: float
    target_position: float
    initial_state: tuple[float, float, float, float]
    track_limit: float = 2.8
    dt: float = 0.02
    actuator_saturation: bool = True
    max_force: float = 15.0
    sensor_noise_std_deg: float = 0.0
    disturbance_profile: DisturbanceProfile = DisturbanceProfile.NONE
    disturbance_magnitude: float = 0.0
    disturbance_start: float = 0.0
    impulse_duration: float = 0.1
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    pole_length: float = 0.5
    gravity: float = 9.81
    viscous_friction: float = 0.0
    coulomb_friction: float = 0.0

    def __post_init__(self) -> None:
        _validate_identifier(self.scenario_id, "scenario_id")
        initial_state = np.asarray(self.initial_state, dtype=float)
        if initial_state.shape != (4,) or not np.all(np.isfinite(initial_state)):
            raise ValueError("initial_state must be a finite four-state tuple")
        for name in ("cart_mass", "pole_mass", "pole_length", "gravity"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.viscous_friction < 0.0 or self.coulomb_friction < 0.0:
            raise ValueError("friction coefficients cannot be negative")

        # Reuse the runtime's validation as the single source of truth.
        self.simulation_config(seed=0)

    def simulation_config(
        self,
        *,
        seed: int,
        controller_mode: ControllerMode = ControllerMode.STATE,
    ) -> SimulationConfig:
        return SimulationConfig(
            total_time=self.total_time,
            target_position=self.target_position,
            track_limit=self.track_limit,
            controller_mode=controller_mode,
            dt=self.dt,
            actuator_saturation=self.actuator_saturation,
            max_force=self.max_force,
            sensor_noise_std_deg=self.sensor_noise_std_deg,
            random_seed=seed,
            disturbance_profile=self.disturbance_profile,
            disturbance_magnitude=self.disturbance_magnitude,
            disturbance_start=self.disturbance_start,
            impulse_duration=self.impulse_duration,
        )

    def to_protocol_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "total_time": self.total_time,
            "target_position": self.target_position,
            "initial_state": list(self.initial_state),
            "track_limit": self.track_limit,
            "dt": self.dt,
            "actuator_saturation": self.actuator_saturation,
            "max_force": self.max_force,
            "sensor_noise_std_deg": self.sensor_noise_std_deg,
            "disturbance_profile": self.disturbance_profile.value,
            "disturbance_magnitude": self.disturbance_magnitude,
            "disturbance_start": self.disturbance_start,
            "impulse_duration": self.impulse_duration,
            "plant": {
                "cart_mass": self.cart_mass,
                "pole_mass": self.pole_mass,
                "pole_length": self.pole_length,
                "gravity": self.gravity,
                "viscous_friction": self.viscous_friction,
                "coulomb_friction": self.coulomb_friction,
            },
        }


@dataclass(frozen=True)
class ControllerDefinition:
    """A named controller recipe with parameters recorded in the protocol."""

    controller_id: str
    label: str
    mode: ControllerMode
    parameters: tuple[tuple[str, float | int], ...]

    def __post_init__(self) -> None:
        _validate_identifier(self.controller_id, "controller_id")

    def to_protocol_dict(self) -> dict[str, Any]:
        return {
            "controller_id": self.controller_id,
            "label": self.label,
            "mode": self.mode.value,
            "parameters": dict(self.parameters),
        }


CONTROLLERS: dict[str, ControllerDefinition] = {
    "pid": ControllerDefinition(
        controller_id="pid",
        label="Position PID",
        mode=ControllerMode.POSITION_PID,
        parameters=(("kp", 10.0), ("ki", 1.0), ("kd", 5.0)),
    ),
    "pole": ControllerDefinition(
        controller_id="pole",
        label="Pole Placement",
        mode=ControllerMode.STATE,
        parameters=(
            ("zeta", 0.7),
            ("wn", 3.5),
            ("p3", -10.0),
            ("p4", -12.0),
        ),
    ),
    "lqr": ControllerDefinition(
        controller_id="lqr",
        label="LQR",
        mode=ControllerMode.STATE,
        parameters=(
            ("q_pos", 100.0),
            ("q_ang", 10.0),
            ("r_weight", 1.0),
        ),
    ),
    "lqi": ControllerDefinition(
        controller_id="lqi",
        label="LQI",
        mode=ControllerMode.STATE_WITH_DT,
        parameters=(
            ("q_pos", 100.0),
            ("q_ang", 10.0),
            ("q_int", 150.0),
            ("r_weight", 1.0),
        ),
    ),
    "mpc": ControllerDefinition(
        controller_id="mpc",
        label="Nonlinear MPC",
        mode=ControllerMode.STATE,
        parameters=(
            ("horizon", 20),
            ("q_pos", 10.0),
            ("q_ang", 50.0),
            ("r_weight", 0.1),
            ("max_force", 15.0),
            ("track_margin", 0.97),
        ),
    ),
}


SCENARIOS: dict[str, BenchmarkScenario] = {
    "nominal_recovery": BenchmarkScenario(
        scenario_id="nominal_recovery",
        description="Recover the upright equilibrium from an 8 degree tilt.",
        total_time=5.0,
        target_position=0.0,
        initial_state=(0.0, 0.0, math.radians(8.0), 0.0),
    ),
    "setpoint_tracking": BenchmarkScenario(
        scenario_id="setpoint_tracking",
        description="Track a one metre cart-position step from a small tilt.",
        total_time=8.0,
        target_position=1.0,
        initial_state=(0.0, 0.0, math.radians(5.7), 0.0),
    ),
    "noisy_recovery": BenchmarkScenario(
        scenario_id="noisy_recovery",
        description="Recover upright with 0.5 degree Gaussian angle noise.",
        total_time=5.0,
        target_position=0.0,
        initial_state=(0.0, 0.0, math.radians(8.0), 0.0),
        sensor_noise_std_deg=0.5,
    ),
    "wind_rejection": BenchmarkScenario(
        scenario_id="wind_rejection",
        description="Track one metre while rejecting a persistent 2 N wind.",
        total_time=8.0,
        target_position=1.0,
        initial_state=(0.0, 0.0, math.radians(5.7), 0.0),
        disturbance_profile=DisturbanceProfile.CONTINUOUS,
        disturbance_magnitude=2.0,
        disturbance_start=2.0,
    ),
}


SUITES: dict[str, tuple[str, ...]] = {
    "quick": ("nominal_recovery", "noisy_recovery"),
    "standard": tuple(SCENARIOS),
}


def _controller_parameters(definition: ControllerDefinition) -> dict[str, Any]:
    return dict(definition.parameters)


def _build_controller(
    definition: ControllerDefinition,
    plant: CartPolePlant,
    scenario: BenchmarkScenario,
) -> Any:
    parameters = _controller_parameters(definition)
    if definition.controller_id == "pid":
        return PIDController(**parameters)
    if definition.controller_id == "pole":
        return StateSpaceController(plant.A, plant.B, **parameters)
    if definition.controller_id == "lqr":
        return LQRController(plant.A, plant.B, **parameters)
    if definition.controller_id == "lqi":
        return LQIController(plant.A, plant.B, **parameters)
    if definition.controller_id == "mpc":
        track_margin = parameters.pop("track_margin")
        return MPCController(
            plant,
            track_limit=track_margin * scenario.track_limit,
            **parameters,
        )
    raise ValueError(f"unsupported controller: {definition.controller_id}")


def _trajectory_sha256(result: SimulationResult) -> str:
    digest = hashlib.sha256()
    for name, values in (
        ("states", result.states),
        ("estimates", result.estimates),
        ("references", result.references),
        ("inputs", result.inputs),
    ):
        array = np.ascontiguousarray(values, dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(_canonical_json(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class BenchmarkRun:
    """Serializable result of one scenario/controller/seed combination."""

    scenario_id: str
    controller_id: str
    seed: int
    success: bool
    terminated: bool
    termination_reason: str | None
    steps: int
    simulated_time_s: float
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
    trajectory_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class BenchmarkAggregate:
    """Seed aggregate for one scenario/controller pair."""

    scenario_id: str
    controller_id: str
    runs: int
    success_rate: float
    termination_rate: float
    rms_tracking_error_mean: float
    rms_tracking_error_std: float
    steady_state_error_mean: float
    total_energy_mean: float
    peak_force_mean: float
    max_angle_deg_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class BenchmarkReport:
    """A deterministic benchmark artifact plus minimal provenance."""

    suite: str
    scenario_ids: tuple[str, ...]
    controller_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    implementation_sha256: str
    protocol_sha256: str
    results_sha256: str
    environment: tuple[tuple[str, str], ...]
    runs: tuple[BenchmarkRun, ...]
    aggregates: tuple[BenchmarkAggregate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "protocol_version": BENCHMARK_PROTOCOL_VERSION,
            "implementation_sha256": self.implementation_sha256,
            "protocol_sha256": self.protocol_sha256,
            "results_sha256": self.results_sha256,
            "suite": self.suite,
            "seeds": list(self.seeds),
            "environment": dict(self.environment),
            "controllers": [
                CONTROLLERS[controller_id].to_protocol_dict()
                for controller_id in self.controller_ids
            ],
            "scenarios": [
                SCENARIOS[scenario_id].to_protocol_dict()
                for scenario_id in self.scenario_ids
            ],
            "runs": [run.to_dict() for run in self.runs],
            "aggregates": [
                aggregate.to_dict() for aggregate in self.aggregates
            ],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seeds must be integers")
    normalized = int(seed)
    if not 0 <= normalized <= 2**32 - 1:
        raise ValueError("seeds must be in the range [0, 2**32 - 1]")
    return normalized


def run_benchmark_case(
    *,
    scenario: BenchmarkScenario,
    controller_id: str,
    seed: int,
) -> BenchmarkRun:
    """Run one fully isolated benchmark case."""

    seed = _validate_seed(seed)
    try:
        definition = CONTROLLERS[controller_id]
    except KeyError as exc:
        raise ValueError(f"unknown controller: {controller_id}") from exc

    plant = CartPolePlant(
        m_c=scenario.cart_mass,
        m_p=scenario.pole_mass,
        l=scenario.pole_length,
        g=scenario.gravity,
    )
    plant.set_friction(
        cart_frictionloss=scenario.coulomb_friction,
        cart_damping=scenario.viscous_friction,
    )
    controller = _build_controller(definition, plant, scenario)
    result = run_simulation(
        config=scenario.simulation_config(
            seed=seed,
            controller_mode=definition.mode,
        ),
        plant=plant,
        controller=controller,
        initial_state=np.asarray(scenario.initial_state, dtype=float),
    )
    metrics = result.metrics
    success = (
        not result.terminated
        and metrics.position_stable
        and metrics.angle_stable
    )

    return BenchmarkRun(
        scenario_id=scenario.scenario_id,
        controller_id=controller_id,
        seed=seed,
        success=success,
        terminated=result.terminated,
        termination_reason=result.termination_reason,
        steps=len(result.states),
        simulated_time_s=len(result.states) * result.dt,
        final_position=metrics.final_position,
        final_angle_deg=metrics.final_angle_deg,
        position_stable=metrics.position_stable,
        angle_stable=metrics.angle_stable,
        rms_tracking_error=metrics.rms_tracking_error,
        steady_state_error=metrics.steady_state_error,
        total_energy=metrics.total_energy,
        peak_force=metrics.peak_force,
        saturation_percentage=metrics.saturation_percentage,
        max_angle_deg=metrics.max_angle_deg,
        safety_intervention_percentage=metrics.safety_intervention_percentage,
        safety_max_deviation=metrics.safety_max_deviation,
        trajectory_sha256=_trajectory_sha256(result),
    )


def _normalized_controller_ids(
    controller_ids: Sequence[str] | None,
) -> tuple[str, ...]:
    requested = (
        set(DEFAULT_CONTROLLERS)
        if controller_ids is None
        else set(controller_ids)
    )
    unknown = requested.difference(CONTROLLERS)
    if unknown:
        raise ValueError(f"unknown controllers: {', '.join(sorted(unknown))}")
    if not requested:
        raise ValueError("at least one controller is required")
    return tuple(
        controller_id
        for controller_id in CONTROLLERS
        if controller_id in requested
    )


def _aggregate_runs(
    runs: tuple[BenchmarkRun, ...],
    scenario_ids: tuple[str, ...],
    controller_ids: tuple[str, ...],
) -> tuple[BenchmarkAggregate, ...]:
    aggregates: list[BenchmarkAggregate] = []
    for scenario_id in scenario_ids:
        for controller_id in controller_ids:
            group = [
                run
                for run in runs
                if run.scenario_id == scenario_id
                and run.controller_id == controller_id
            ]
            rms = np.asarray(
                [run.rms_tracking_error for run in group],
                dtype=float,
            )
            aggregates.append(
                BenchmarkAggregate(
                    scenario_id=scenario_id,
                    controller_id=controller_id,
                    runs=len(group),
                    success_rate=float(np.mean([run.success for run in group])),
                    termination_rate=float(
                        np.mean([run.terminated for run in group])
                    ),
                    rms_tracking_error_mean=float(np.mean(rms)),
                    rms_tracking_error_std=float(np.std(rms)),
                    steady_state_error_mean=float(
                        np.mean([run.steady_state_error for run in group])
                    ),
                    total_energy_mean=float(
                        np.mean([run.total_energy for run in group])
                    ),
                    peak_force_mean=float(
                        np.mean([run.peak_force for run in group])
                    ),
                    max_angle_deg_mean=float(
                        np.mean([run.max_angle_deg for run in group])
                    ),
                )
            )
    return tuple(aggregates)


def _environment() -> tuple[tuple[str, str], ...]:
    versions = [("python", platform.python_version())]
    for distribution in ("numpy", "scipy", "control", "casadi"):
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            version = "unknown"
        versions.append((distribution, version))
    return tuple(versions)


def run_benchmark(
    *,
    suite: str = "standard",
    controller_ids: Sequence[str] | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> BenchmarkReport:
    """Execute a canonical suite with deterministic ordering and isolation."""

    try:
        scenario_ids = SUITES[suite]
    except KeyError as exc:
        raise ValueError(f"unknown suite: {suite}") from exc

    normalized_controllers = _normalized_controller_ids(controller_ids)
    normalized_seeds = tuple(sorted({_validate_seed(seed) for seed in seeds}))
    if not normalized_seeds:
        raise ValueError("at least one seed is required")

    protocol = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "suite": suite,
        "seeds": list(normalized_seeds),
        "controllers": [
            CONTROLLERS[controller_id].to_protocol_dict()
            for controller_id in normalized_controllers
        ],
        "scenarios": [
            SCENARIOS[scenario_id].to_protocol_dict()
            for scenario_id in scenario_ids
        ],
    }
    protocol_sha256 = _sha256(protocol)

    runs = tuple(
        run_benchmark_case(
            scenario=SCENARIOS[scenario_id],
            controller_id=controller_id,
            seed=seed,
        )
        for scenario_id in scenario_ids
        for controller_id in normalized_controllers
        for seed in normalized_seeds
    )
    aggregates = _aggregate_runs(
        runs,
        scenario_ids,
        normalized_controllers,
    )
    results_sha256 = _sha256(
        {
            "runs": [run.to_dict() for run in runs],
            "aggregates": [
                aggregate.to_dict() for aggregate in aggregates
            ],
        }
    )

    return BenchmarkReport(
        suite=suite,
        scenario_ids=scenario_ids,
        controller_ids=normalized_controllers,
        seeds=normalized_seeds,
        implementation_sha256=_implementation_sha256(),
        protocol_sha256=protocol_sha256,
        results_sha256=results_sha256,
        environment=_environment(),
        runs=runs,
        aggregates=aggregates,
    )


CSV_FIELDS = (
    "scenario_id",
    "controller_id",
    "seed",
    "success",
    "terminated",
    "termination_reason",
    "steps",
    "simulated_time_s",
    "final_position",
    "final_angle_deg",
    "position_stable",
    "angle_stable",
    "rms_tracking_error",
    "steady_state_error",
    "total_energy",
    "peak_force",
    "saturation_percentage",
    "max_angle_deg",
    "safety_intervention_percentage",
    "safety_max_deviation",
    "trajectory_sha256",
)


def _atomic_write(path: Path, content: str) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8", newline="")
    temporary_path.replace(path)


def write_benchmark_report(
    report: BenchmarkReport,
    output_dir: str | Path,
    *,
    stem: str = "benchmark_results",
) -> tuple[Path, Path]:
    """Write stable JSON and CSV artifacts, replacing prior files atomically."""

    _validate_identifier(stem, "stem")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{stem}.json"
    csv_path = output_path / f"{stem}.csv"

    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=CSV_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    for run in report.runs:
        writer.writerow(run.to_dict())

    _atomic_write(json_path, f"{report.to_json()}\n")
    _atomic_write(csv_path, csv_buffer.getvalue())
    return json_path, csv_path


def _print_summary(report: BenchmarkReport) -> None:
    print(
        f"implementation={report.implementation_sha256[:12]} "
        f"protocol={report.protocol_sha256[:12]} "
        f"results={report.results_sha256[:12]} "
        f"runs={len(report.runs)}"
    )
    print(
        f"{'scenario':<20} {'controller':<10} {'success':>8} "
        f"{'rms':>10} {'ss error':>10} {'energy':>10} {'max angle':>10}"
    )
    for aggregate in report.aggregates:
        print(
            f"{aggregate.scenario_id:<20} "
            f"{aggregate.controller_id:<10} "
            f"{aggregate.success_rate:>7.0%} "
            f"{aggregate.rms_tracking_error_mean:>10.4f} "
            f"{aggregate.steady_state_error_mean:>10.4f} "
            f"{aggregate.total_energy_mean:>10.2f} "
            f"{aggregate.max_angle_deg_mean:>9.2f}°"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic cart-pole controller benchmarks through the "
            "same runtime used by the Streamlit app."
        )
    )
    parser.add_argument(
        "--suite",
        choices=tuple(SUITES),
        default="standard",
        help="quick is a smoke benchmark; standard exercises all scenarios",
    )
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=tuple(CONTROLLERS),
        default=list(DEFAULT_CONTROLLERS),
        help="controller recipes to compare",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="explicit RNG seeds (deduplicated and sorted)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="directory for deterministic JSON and CSV artifacts",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print results without writing artifacts",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list available suites/controllers and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.list:
        print("suites:")
        for suite, scenario_ids in SUITES.items():
            print(f"  {suite}: {', '.join(scenario_ids)}")
        print("controllers:")
        for controller_id, definition in CONTROLLERS.items():
            suffix = " (default)" if controller_id in DEFAULT_CONTROLLERS else ""
            print(f"  {controller_id}: {definition.label}{suffix}")
        return 0

    report = run_benchmark(
        suite=args.suite,
        controller_ids=args.controllers,
        seeds=args.seeds,
    )
    _print_summary(report)
    if not args.no_write:
        json_path, csv_path = write_benchmark_report(
            report,
            args.output_dir,
        )
        print(f"json={json_path}")
        print(f"csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
