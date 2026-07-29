# Inverted Pendulum Digital Twin — from PID to Certifiably Safe Learning-Based Control

An interactive cart-pole simulator that walks the entire arc of modern control on a
single system: classical PID, pole placement, LQR/LQI, iLQR swing-up, constrained
nonlinear MPC, Gaussian-Process learning-based MPC with chance constraints, and a
model-predictive safety filter that can certify *any* policy at runtime — including a
deliberately reckless one.

Every method runs on the **same plant, the same symbolic model, and the same 20 ms
control loop**. When one controller fails and another succeeds, the difference is
the method—not a hidden change in the simulation setup.

**Live demo:** _add your Streamlit Cloud URL here_

---

## Why this exists

Most control tutorials show one method on one toy problem. This project instead
compares a broad set of methods on the same plant so their structural strengths,
limitations and trade-offs become directly observable.

| Method | Core capability | Engineering value |
|---|---|---|
| PID | error feedback | transparent SISO baseline on an unstable plant |
| Pole Placement / LQR / LQI | full-state feedback | stability, optimality and offset rejection |
| iLQR | nonlinear trajectory optimisation | swing-up from the hanging state |
| MPC | constrained receding-horizon control | limits live *inside* the optimiser |
| Robust MPC mode | fixed uncertainty margin | worst-case constraint protection |
| GP-MPC | learned model residual | corrects model error from data |
| Chance-constraint mode | propagated learned uncertainty | data-dependent constraint margins |
| MPSC safety filter | runtime safety certification | minimally filters any proposed policy |

Two companion studies live outside the app as scripts:
**residual GP model learning** (`experiment_phase3.py`) and **dynamic programming /
value iteration** (`experiment_value_iteration.py`).

---

## The demos that deserve attention

These are all reproducible in the app with the sliders; numbers below are from my runs.

**Learning recovers what model knowledge can't.** With rail friction on (viscous 3.0,
Coulomb 1.0) and a target at 1.0 m, an MPC that knows the *true* masses but not the
friction parks at **1.217 m and hunts**. Give the controller a pole mass that's 100 %
wrong *plus* a GP residual trained on 36 s of data, and it settles at **0.982 m,
0.04°**. The interesting part: the true-mass friction-blind MPC fails exactly like
the wrong-mass one — friction dominates, and no amount of parameter knowledge fixes
it. Only learning does.

**Conservatism you can watch shrink.** Ask the controller to park at 1.60 m when the
wall is at 1.50 m (an intentionally infeasible request). Nominal MPC parks **6 mm past
the wall** — a real violation. Chance-constrained GP-MPC with an immature model
(3 training rollouts) stays **14 mm inside**; give it 12 rollouts and the margin
shrinks toward the wall as the model earns confidence. Robust mode parks **28 cm
away, forever**, regardless of data. This makes the core trade-off—robust protection
is conservative while learning can reduce conservatism—visible in one slider sweep. There's an
animated prediction tube in the app (🔮) where you can literally watch the
uncertainty band get thinner as you add rollouts.

**An honest failure, kept on purpose.** Under a constant wind the GP never saw in
training, chance mode behaves like nominal mode and hits the wall, while robust mode
survives. That's not a bug — GP-σ measures *epistemic* uncertainty (what the learned
model is unsure about), and an exogenous disturbance absent from the data is
invisible to it. I could have hidden this; I think it's the most instructive result
in the project.

**Certifying a reckless policy without retraining it.** The "Reckless" brain balances
the pole with perfectly good LQR gains but drives toward a point *beyond the rail
end* — my stand-in for an RL policy trained without constraint knowledge. Unfiltered,
it's out of bounds in about a second. Wrapped with the MPSC safety filter, it runs
indefinitely, held ~2 cm inside the wall with the pole upright. And the filter's
other half matters just as much: wrapping a *safe* LQR, the filter's maximum
deviation over a whole run is **0.0003 N with zero interventions** — certification
with no performance tax.

**DP rediscovers LQR.** Value iteration over 43,875 discretised states (nonuniform
sinh-spaced grids — uniform grids chatter) converges in 621 sweeps and produces a
value function that correlates **0.98** with LQR's analytic `x'Px` near the origin,
with the same policy structure. The remaining ~2× closed-loop cost gap is the price
of discretisation: dynamic programming is exact in principle but becomes expensive
as the state space grows. Analytic linear solutions and function approximation offer
two different ways around that scaling problem.

---

## What's under the hood

**One symbolic model, everywhere.** `plant.py` defines the cart-pole dynamics once
in CasADi. From that single source: the continuous `f`, an RK4 discrete map `F`,
and *exact* Jacobians `A, B` by automatic differentiation. The simulator keeps a
frozen, fast NumPy path that is asserted bit-consistent with the symbolic model —
so the controllers and the "reality" they act on can never silently drift apart.
Checking hand-derived Jacobians against automatic differentiation caught a ~10 %
algebra error before it could propagate into the controllers.

**Sequential GP-MPC.** Embedding 300 GP kernel terms per timestep into the NLP made
IPOPT roughly 4× slower, so the GP mean *and* std are evaluated along the
warm-started plan with one batched call and enter the NLP as parameters. The NLP
stays plain-MPC-sized (~45 ms warm solves) while still planning with the learned
model. Chance and robust constraint tightening ride the same parameter channel:
first-order covariance propagation along the plan, `|pos_k| ≤ L − κ·σ_pos,k`.

**MPSC as a wrapper, not a controller.** The filter solves
`min (u₀ − u_prop)²` subject to dynamics, input/state limits and a terminal
recovery box — it has no opinion about performance, only about whether a safe
future still exists. It wraps whichever brain is active via a checkbox.

**One runtime, every interface.** `simulation.py` owns the complete closed-loop
runtime — measurement, estimation, reference generation, control, safety
filtering, saturation, disturbance injection and plant integration. It has no
Streamlit dependency. The app, tests and reproducible batch benchmark therefore
run the same simulation code and consume the same typed `SimulationResult`.

**Honest simplifications, stated as such:** the MPSC terminal box stands in for a
certified invariant set (a full implementation uses an RPI/CLF set); the robust
mode is a simplified tube MPC (first-order propagation, no RPI computation); state
constraints are soft (heavily penalised slack) so IPOPT stays feasible under
disturbances; and the filter enforces 0.97·L because it is *optimally lazy* — it
defers braking to the last feasible moment, so a small standoff has to absorb
model/step mismatch. I'd rather name these than pretend they aren't there.

**A war story.** While auditing `controller.py` I found my steady-state Kalman
filter had been calling `solve_discrete_are(Ad, Cd.T, ...)` — missing the
transpose that the estimation/control duality requires. The wrong equation didn't
just degrade the filter; it made the DARE *fail outright* on default settings.
One character (`Ad.T`), verified by checking the estimator error dynamics'
eigenvalues, and the filter finally does what its docstring always promised.

---

## Running it

```bash
git clone https://github.com/Ttunaakkaya/interactive-control-systems.git
cd interactive-control-systems
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The offline studies:

```bash
python experiment_phase3.py            # residual GP: 12x one-step, ~600x rollout improvement
python experiment_value_iteration.py   # DP vs LQR: value/policy comparison plots
```

Both write interactive Plotly HTML reports next to the scripts.

### Application structure

The Streamlit interface is split into three focused product surfaces through
native multipage navigation:

- **Live Simulation** — the full 50 Hz digital twin, controller configuration,
  animation and diagnostic telemetry.
- **Controller Benchmarks** — reproducible suite configuration, cached execution,
  comparison charts, aggregate/raw tables and JSON/CSV downloads.
- **Theory & Method Guide** — the model, estimator, controller, learning and
  safety concepts arranged as a progressive engineering reference.

`app.py` is intentionally a small application shell. Page implementations live
under `pages/`, while `ui_components.py` owns the shared visual system. Numerical
work remains in `simulation.py` and `benchmark.py`; the pages only orchestrate and
present those APIs.

### Reproducible benchmark mode

The benchmark CLI runs the same `simulation.py` runtime as the app, but with
versioned scenarios, explicit random seeds and a fresh plant/controller for every
case. It intentionally measures simulated control quality rather than wall-clock
speed, which would depend on the machine and background load.

```bash
python benchmark.py
python benchmark.py --suite quick --controllers lqr lqi --seeds 7 11
python benchmark.py --list
```

The default protocol compares Pole Placement, LQR and LQI under nominal recovery,
setpoint tracking, sensor noise and continuous wind. PID and nonlinear MPC are
available through `--controllers`, but excluded from the default set: PID does not
observe the pole angle, while MPC makes the otherwise fast suite substantially
slower.

Each invocation prints an aggregate table and writes:

- `benchmark_results/benchmark_results.json` — complete versioned protocol,
  package provenance, raw runs, seed aggregates and integrity hashes.
- `benchmark_results/benchmark_results.csv` — one stable row per
  scenario/controller/seed for analysis in pandas, Excel or MATLAB.

There is deliberately no timestamp or wall-clock duration in either artifact.
Running the same protocol in the same dependency environment produces byte-stable
JSON/CSV and identical trajectory SHA-256 values.

**A note on speed:** MPC, GP-MPC and the safety filter each solve a real NLP every
20 ms of simulated time, so a 10 s run computes for ~20–25 s. That's the method,
not a bug — the app shows a ⏳ note wherever it applies. First GP-MPC run also
trains the GP once (~20 s), then it's cached per configuration.

Run the automated checks with the development dependencies:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite covers plant/model consistency, controller and observer invariants,
MPC/GP-MPC optimizer smoke tests, exact pre/post-refactor runtime parity, runtime
ordering and failure diagnostics, benchmark reproducibility/artifact contracts,
and a Streamlit rendering smoke test.

## Repo map

```
app.py                          native Streamlit navigation + application shell
pages/live_simulation.py        50 Hz digital twin, controls, animation, telemetry
pages/benchmark_dashboard.py    benchmark runner, charts, tables, JSON/CSV export
pages/theory_guide.py            structured model/method/safety reference
ui_components.py                shared styling and page-heading primitives
simulation.py                   UI-independent closed-loop runtime + typed results/metrics
benchmark.py                    versioned seeded suites + deterministic JSON/CSV reports
plant.py                        single symbolic cart-pole model (CasADi) + fast simulator
controller.py                   PID → ... → GP-MPC → MPSC, one shared interface
learning.py                     rollout collection + residual GP (2 GPs on velocity states)
value_iteration.py              discretised DP: grids, interpolation operator, VI, greedy policy
experiment_phase3.py            residual model-learning study
experiment_value_iteration.py   DP-rediscovers-LQR study
test_*.py                       regression, invariant, optimizer and UI smoke tests
requirements.txt / requirements-dev.txt
```

## Where this is going

Next on my list: a runtime out-of-distribution monitor built on the GP's
uncertainty signal (the in- vs out-of-distribution σ ratio is already ~14× in the
residual-learning data—it can become an alarm that triggers the safety filter),
followed by a data-efficiency comparison of GP-MPC against DQN/PPO/SAC on the same
plant.

---

*Screenshots/GIFs: (add 3–4 here — the swing-up, the wind demo, the prediction
tube at 3 vs 30 rollouts, and the safety-filter telemetry chart make the best set.)*
