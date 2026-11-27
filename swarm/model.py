"""The full control law and the containers the simulator runs on.

`SimSpec` is static (hashable, a jit static argument): physical constants,
controller kind and mask, hard/soft neighbors, horizon. Everything per episode
is dynamic and lives in an `Episode` pytree, so one compiled function serves
every episode with the same (N, K, T).

Control law for agent i (doc §3.4, §4.5):

    u_raw_i = sum_j w_ij f_ij / nu_j,   j in (sep, ali, coh, goal, obs)

with w_i from the controller (swarm.controllers) and the goal force gated by
the informed flag.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from swarm.config import SimParams
from swarm.controllers import ControllerSpec, behavior_weights, preactivation
from swarm.episodes import EpisodeBatch
from swarm.features import local_features
from swarm.forces import boundary_force, force_scales, obstacle_force
from swarm.maps import MapSpec
from swarm.observation import local_goal_observation
from swarm.perception import local_interactions


# --------------------------------------------------------------------------- #
# Containers
# --------------------------------------------------------------------------- #
class Episode(NamedTuple):
    """One episode (or a batch, with a leading axis on every field)."""

    lower: jnp.ndarray        # (2,)
    upper: jnp.ndarray        # (2,)
    centers: jnp.ndarray      # (K, 2), padded
    radii: jnp.ndarray        # (K,)
    valid: jnp.ndarray        # (K,) 1 = real obstacle
    goal: jnp.ndarray         # (2,)  never read by uninformed agents' control
    goal_radius: jnp.ndarray  # ()
    x0: jnp.ndarray           # (N, 2)
    v0: jnp.ndarray           # (N, 2)
    informed: jnp.ndarray     # (N,) bool


def episodes_from_batch(batch: EpisodeBatch, p: float = 1.0, k_max: int | None = None) -> Episode:
    """Batched Episode (leading axis B) from a cached split, informed fraction p."""
    if k_max is not None:
        batch = batch.with_k_max(k_max)
    informed = np.ones(batch.order.shape, dtype=bool) if p == 1.0 else batch.informed(p)
    return Episode(
        lower=jnp.asarray(batch.lower),
        upper=jnp.asarray(batch.upper),
        centers=jnp.asarray(batch.centers),
        radii=jnp.asarray(batch.radii),
        valid=jnp.asarray(batch.valid),
        goal=jnp.asarray(batch.goal),
        goal_radius=jnp.asarray(batch.goal_radius),
        x0=jnp.asarray(batch.x0),
        v0=jnp.asarray(batch.v0),
        informed=jnp.asarray(informed),
    )


def episode_from_map(m: MapSpec, x0, v0=None, informed=None, k_max: int | None = None) -> Episode:
    """Single Episode from a MapSpec and an initial state (handy for tests and demos)."""
    n = len(x0)
    k = max(1, m.n_obstacles) if k_max is None else k_max
    if k < m.n_obstacles:
        raise ValueError(f"k_max={k} smaller than the obstacle count {m.n_obstacles}")
    centers, radii, valid = np.zeros((k, 2)), np.zeros(k), np.zeros(k)
    centers[: m.n_obstacles], radii[: m.n_obstacles], valid[: m.n_obstacles] = m.centers, m.radii, 1.0
    return Episode(
        lower=jnp.asarray(m.lower),
        upper=jnp.asarray(m.upper),
        centers=jnp.asarray(centers),
        radii=jnp.asarray(radii),
        valid=jnp.asarray(valid),
        goal=jnp.asarray(m.goal),
        goal_radius=jnp.asarray(m.goal_radius),
        x0=jnp.asarray(x0, dtype=jnp.result_type(float)),
        v0=jnp.zeros((n, 2)) if v0 is None else jnp.asarray(v0, dtype=jnp.result_type(float)),
        informed=jnp.ones(n, dtype=bool) if informed is None else jnp.asarray(informed, dtype=bool),
    )


def take(episodes: Episode, idx) -> Episode:
    """Index a batched Episode (an int gives a single episode)."""
    return Episode(*(f[idx] for f in episodes))


@dataclasses.dataclass(frozen=True)
class SimSpec:
    """Static simulation settings.

    horizon        rollout length T (default: base config; use the split's
                   metadata horizon for fixed-density scale sets)
    clock_horizon  T used by the Time-25 basis omega(t); defaults to `horizon`.
                   Pass the full T during horizon continuation.
    gamma_missing  goal-distance feature for uninformed agents (0 or 1)
    remat          wrap the step in jax.checkpoint. Only affects the backward pass.
                   Default on: in the benchmark it cut the gradient's temporary
                   memory from 5.4 GB to 26 MB (M=32, N=30, T=400) and ran faster.
    """

    sim: SimParams
    controller: ControllerSpec
    hard: bool = True
    horizon: int | None = None
    clock_horizon: int | None = None
    gamma_missing: float = 0.0
    remat: bool = True

    def __post_init__(self):
        if self.horizon is not None and self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.gamma_missing not in (0.0, 1.0):
            raise ValueError("gamma_missing must be 0 or 1")

    @property
    def T(self) -> int:
        return self.sim.simulation.horizon if self.horizon is None else self.horizon

    @property
    def T_clock(self) -> int:
        return self.T if self.clock_horizon is None else self.clock_horizon

    def replace(self, **changes) -> "SimSpec":
        return dataclasses.replace(self, **changes)


# --------------------------------------------------------------------------- #
# Control law
# --------------------------------------------------------------------------- #
class Control(NamedTuple):
    u_raw: jnp.ndarray      # (N, 2) unclipped command
    weights: jnp.ndarray    # (N, 5) behavior weights
    forces: jnp.ndarray     # (N, 5, 2) normalized behavior forces f_j / nu_j
    features: jnp.ndarray   # (N, 4) z = [rho, o, gamma, kappa]


def compute_control(theta, x, v, t, ep: Episode, spec: SimSpec, eta, mu_z) -> Control:
    """Controller output for every agent at state (x, v) and step t."""
    sim = spec.sim
    p, f, eps, r_a = sim.perception, sim.forces, sim.numerics.eps, sim.agent.radius

    sep, ali, coh, nb = local_interactions(
        x, v, p.sensor_radius, p.sep_radius, eta, p.sep_softening, spec.hard, eps
    )
    obs = local_goal_observation(x, v, ep.goal, ep.goal_radius, ep.informed, nb.a, sim, spec.gamma_missing)
    f_obs, d_near = obstacle_force(
        x, obs.heading, ep.centers, ep.radii, ep.valid, r_a, f.obstacle_influence,
        f.tangential_gain, f.tangential_sharpness, f.tangential_bias, eps,
    )
    f_wall, d_wall = boundary_force(x, ep.lower, ep.upper, r_a, f.obstacle_influence)

    scales = jnp.asarray(force_scales(sim), x.dtype)
    forces = jnp.stack([sep, ali, coh, obs.goal_force, f_obs + f_wall], axis=1) / scales[None, :, None]
    z = local_features(nb.count, d_near, d_wall, obs.gamma, coh, p.sensor_radius, sim.features.n_ref, eps)
    s = preactivation(theta, spec.controller, z, t, mu_z, spec.T_clock)
    w = behavior_weights(s, sim.controller.w_max, spec.controller.mask)
    u_raw = jnp.sum(w[..., None] * forces, axis=1)
    return Control(u_raw, w, forces, z)
