# Optimizing Swarm Intelligence

Learning generalizable swarm intelligence through adaptive local rule optimization
(18-660 Optimization final project). Python + JAX.

**Status:** Phase 0 (environment) and Phase 1 (maps, episodes, datasets) are done.
The forward model, optimizers and experiments come next.

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

`build_datasets` writes every split to `data/<split>.npz` + `.json` (about 13 s,
2.5 MB). These files are the common random numbers shared by all optimizers.
Each carries a hash of the settings that produced it; loading a split whose
settings have changed raises `StaleDatasetError` (rebuild with `--force`).
`preview_maps` renders every scenario type to `results/figures/map_preview.png`.
`data/` and `results/` are gitignored.

## Layout

```text
configs/
  base.yaml              static model constants (doc §10) and loss weights (§5.3)
  data.yaml              scenario generators, layout rules, dataset splits
swarm/
  config.py              strict SimParams loader, config hashing, precision switch
  geometry.py            differentiable norms, clipnorm, smoothstep, swept geometry (JAX)
  rng.py                 independent random streams per quantity
  maps.py                MapSpec and scenario generators
  feasibility.py         capacity rule, clearances, grid connectivity check
  episodes.py            initial states, nested informed masks, padded EpisodeBatch
  datasets.py            split building, caching, staleness checks
  descriptors.py         map descriptor psi(E) for Exp3 and its standardizer
experiments/
  common.py              run directories with config snapshot and git hash
  build_datasets.py      generate and cache all splits
  preview_maps.py        visual check of the generators
visualization/plots.py   map drawing
tests/                   unit tests (pytest)
```

## Data conventions

These are choices the project doc left open, made while implementing Phase 1:

- **Layout.** The start box is on the left and the goal disk on the right of the
  domain. The start box area puts the agents at an area fraction of 0.20–0.35
  with 0.25 minimum spacing. Large swarms get taller boxes (width ≤ 25% of the
  domain).
- **Walls.** Bottleneck walls are vertical chains of overlapping circles
  (radius 0.3, spacing 0.3) that end on the boundaries. The gap is measured
  surface to surface.
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
