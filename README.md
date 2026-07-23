# Inverted Pendulum Digital Twin — from PID to Certifiably Safe Learning-Based Control

An interactive cart-pole simulator that walks the entire arc of modern control on a
single system: classical PID, pole placement, LQR/LQI, iLQR swing-up, constrained
nonlinear MPC, Gaussian-Process learning-based MPC with chance constraints, and a
model-predictive safety filter that can certify *any* policy at runtime — including a
deliberately reckless one.

I built this alongside the *Optimal Control and Decision Making* course at TUM. The
rule I set myself: every method had to run on the **same plant, the same symbolic
model, and the same 20 ms control loop**, so that when one controller fails and the
next one doesn't, the difference is the *method* — not the setup.

**Live demo:** _add your Streamlit Cloud URL here_

---

## Why this exists

Most control tutorials show you one method on one toy problem. What I wanted to see
(and show) is the *ladder*: each technique exists because the previous one has a
structural limitation you can actually watch happen.

| Method | Course ref. | What it fixes |
|---|---|---|
| PID | baseline | — (and why SISO fails on a 4-state unstable plant) |
| Pole Placement / LQR / LQI | Ch. 2 | full-state, *optimal* feedback |
| iLQR | Ch. 4 | nonlinear trajectory optimisation — swing-up from hanging |
| MPC | Ch. 5 | constraints live *inside* the optimiser |
| Robust MPC mode | Ch. 5.5 | worst-case constraint tightening |
| GP-MPC | Ch. 6 | *learns* the model error from data |
| Chance-constraint mode | Ch. 6.5 | tightening from the model's own uncertainty |
| MPSC safety filter | safe-RL literature | certifies any policy, minimally |

Two companion studies live outside the app as scripts:
**residual GP model learning** (`experiment_phase3.py`, Ch. 6.1–6.3) and
**dynamic programming / value iteration** (`experiment_value_iteration.py`,
Ch. 1.2 — which is also Ch. 7.3.1, because model-based RL *is* the DP algorithm).

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
away, forever**, regardless of data. That's the textbook argument — "robustness is
conservative; learning reduces conservatism" — in one slider sweep. There's an
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
of discretisation, which is precisely Chapter 1's lesson: DP is exact in principle
and cursed in practice — the reason the linear case gets solved analytically and the
deep-RL chapter replaces the table with a network.

---

## What's under the hood

**One symbolic model, everywhere.** `plant.py` defines the cart-pole dynamics once
in CasADi. From that single source: the continuous `f`, an RK4 discrete map `F`,
and *exact* Jacobians `A, B` by automatic differentiation. The simulator keeps a
frozen, fast NumPy path that is asserted bit-consistent with the symbolic model —
so the controllers and the "reality" they act on can never silently drift apart.
(Writing the Jacobians by hand first and then checking them against autodiff caught
a ~10 % error in my own algebra. Lesson absorbed.)

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
Streamlit dependency. The app, tests and future batch benchmarks therefore run
the same simulation code and consume the same typed `SimulationResult`.

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

**A note on speed:** MPC, GP-MPC and the safety filter each solve a real NLP every
20 ms of simulated time, so a 10 s run computes for ~20–25 s. That's the method,
not a bug — the app shows a ⏳ note wherever it applies. First GP-MPC run also
trains the GP once (~20 s), then it's cached per configuration.

The numerical simulation and all telemetry remain at 50 Hz. Only the cart-pole
animation is uniformly sampled to 25 FPS before it is sent to the browser; its
first/last state and total playback time are preserved. This keeps the UI payload
smaller without changing controller results, metrics, or diagnostic plots.

Run the automated checks with the development dependencies:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The suite covers plant/model consistency, controller and observer invariants,
MPC/GP-MPC optimizer smoke tests, exact pre/post-refactor runtime parity, runtime
ordering and failure diagnostics, and a Streamlit rendering smoke test.

## Repo map

```
app.py                          Streamlit app — 8 controllers, safety filter, telemetry
simulation.py                   UI-independent closed-loop runtime + typed results/metrics
plant.py                        single symbolic cart-pole model (CasADi) + fast simulator
controller.py                   PID → ... → GP-MPC → MPSC, one shared interface
learning.py                     rollout collection + residual GP (2 GPs on velocity states)
value_iteration.py              discretised DP: grids, interpolation operator, VI, greedy policy
experiment_phase3.py            model-learning study (Ch. 6.1–6.3)
experiment_value_iteration.py   DP-rediscovers-LQR study (Ch. 1.2 / 7.3.1)
test_*.py                       regression, invariant, optimizer and UI smoke tests
requirements.txt / requirements-dev.txt
```

## Where this is going

Next on my list: a runtime out-of-distribution monitor built on the GP's
uncertainty signal (the in- vs out-of-distribution σ ratio is already ~14× in the
Phase-3 data — it wants to become an alarm that triggers the safety filter), and,
once the RL chapters land, a data-efficiency comparison of GP-MPC against
DQN/PPO/SAC on the same plant.

---

*Screenshots/GIFs: (add 3–4 here — the swing-up, the wind demo, the prediction
tube at 3 vs 30 rollouts, and the safety-filter telemetry chart make the best set.)*
