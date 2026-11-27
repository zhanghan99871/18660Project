# Optimizing Swarm Intelligence

Learning generalizable swarm intelligence through adaptive local rule optimization
(18-660 Optimization final project). Python + JAX.

**Status:** Phase 0 (environment), Phase 1 (maps, episodes, datasets) and
Phase 2 (differentiable simulator, metrics, hand-tuned baseline, benchmark) are
done. Optimizers and experiments come next.

**[User guide](docs/user_guide.md)** covers setup, datasets, the simulator API,
evaluation, animation, the hand-tuned protocol, the benchmark and the tests.

## Setup

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` installs the CPU build of JAX. For an NVIDIA GPU, install
`jax[cuda12]==0.12.1` instead. Consumer GeForce/RTX cards run float64 far slower
than float32, which matters for the L-BFGS-B path.

**Precision.** The project runs in float64 (`numerics.precision` in
`configs/base.yaml`). `jax_enable_x64` must be set before any JAX array is
created, so every entry point calls `swarm.config.configure_jax(...)` first.

## Commands

```bash
.venv/bin/python -m pytest
```

```bash
.venv/bin/python -m experiments.build_datasets
```

```bash
.venv/bin/python -m experiments.preview_maps
```

```bash
.venv/bin/python -m experiments.hand_tuned
```

```bash
.venv/bin/python -m experiments.benchmark
```

```bash
.venv/bin/python -m visualization.animate --split val_id --index 0 --frozen
```

`build_datasets` writes every split to `data/<split>.npz` + `.json` (about 13 s,
2.5 MB). These files are the common random numbers shared by all optimizers.
Each carries a hash of the settings that produced it; loading a split whose
settings have changed raises `StaleDatasetError` (rebuild with `--force`).
`preview_maps` renders every scenario type to `results/figures/map_preview.png`.
`hand_tuned` runs the logged hand-tuning trials and writes `configs/frozen.yaml`.
`benchmark` times every objective path to size the per-run budget. `animate`
writes a GIF/MP4 of one rollout.
`data/` and `results/` are gitignored.

## Layout

```text
configs/
  base.yaml              static model constants (doc §10) and loss weights (§5.3)
  data.yaml              scenario generators, layout rules, dataset splits
  hand_tuned.yaml        logged hand-tuning trials (<= 20)
  frozen.yaml            generated: hand-tuned rule and feature centering mu_z
docs/user_guide.md       how to use everything
swarm/
  config.py              strict SimParams loader, config hashing, precision switch
  geometry.py            differentiable norms, clipnorm, smoothstep, swept geometry (JAX)
  rng.py                 independent random streams per quantity
  maps.py                MapSpec and scenario generators
  feasibility.py         capacity rule, clearances, grid connectivity check
  episodes.py            initial states, nested informed masks, padded EpisodeBatch
  datasets.py            split building, caching, staleness checks
  descriptors.py         map descriptor psi(E) for Exp3 and its standardizer
  perception.py          hard/soft neighbors; separation, alignment, cohesion
  forces.py              goal, obstacle (normal + tangential), boundary forces
  observation.py         goal observation with partial-information masking
  features.py            local features z = [rho, o, gamma, kappa]
  controllers.py         fixed / adaptive / time-25 weight maps, masks, theta layout
  model.py               Episode, SimSpec, full control law
  dynamics.py            damped semi-implicit Euler with limits
  losses.py              swept task losses
  rollout.py             scan rollouts, batch/population objectives, gradients
  metrics.py             evaluation: contacts/events, arrival, deadlock, completion
experiments/
  common.py              run directories with config snapshot and git hash
  build_datasets.py      generate and cache all splits
  preview_maps.py        visual check of the generators
  hand_tuned.py          hand-tuning protocol, lambda check, mu_z
  benchmark.py           timings, c_grad, budget planning
visualization/
  plots.py               map drawing
  animate.py             rollout animation
tests/                   data and infrastructure tests
tests/simulation/        simulator tests
```

## Data conventions

These are choices the project doc left open, made while implementing Phase 1:

- **Layout.** The start box is on the left and the goal disk on the right of the
  domain. The start box area puts the agents at an area fraction of 0.20–0.35
  with 0.25 minimum spacing. Large swarms get taller boxes (width ≤ 25% of the
  domain).
- **Walls.** Bottleneck walls are vertical chains of overlapping circles
  (radius 0.3, spacing 0.3) that end on the boundaries. The gap is measured
  surface to surface, and centered within ±1 of where the start→goal line
  crosses the wall (generator version 2). A gap anywhere along the wall acts as
  a trap for agents that sense only 1 unit around them.
- **Large obstacles (OOD).** Radius 1.3–2.0, above every training radius
  (max 1.2); 2–4 obstacles per map.
- **Fixed density.** The domain side and horizon scale by sqrt(N/30), and
  random-obstacle counts scale with area.
- **Scenario mixture.** Scenarios are assigned to maps in balanced proportions
  before rejection sampling, so rejections can't tilt the mixture.
- **Informed agents.** Datasets store an informed *order*. The mask for fraction
  p is its first max(1, floor(pN + 0.5)) agents, so masks are nested across p.
- **Leakage.** Initial positions, velocities and the informed order never read
  the goal or obstacles, and each comes from its own random stream.
- **Corridor gap.** The descriptor's "corridor gap" is the maximin passage width
  between the start box and the goal. A literal pairwise minimum gap breaks on
  overlapping wall chains.
