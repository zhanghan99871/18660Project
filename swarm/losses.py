"""Dimensionless task losses (doc §5.2–5.3).

Per-step terms are accumulated as sums inside the rollout and normalized once
at the end. Step t covers the segment x^t -> x^{t+1} (t = 0..T-1, including the
first step from the initial state), so safety uses swept geometry and never
misses a contact between samples.

  J_g = (1/N) sum_i ([N(x_i^T - g) - R_g]_+ / L0)^2
  J_t = (1/NT) sum_{t,i} [N(x_i^{t+1} - g) - R_g]_+ / L0
  J_c = (1/NT) sum_t sum_{i<j} [(2 r_a + delta - D_ij) / r_a]_+^2
  J_o = (1/NT) sum_{t,i,k valid} [(delta - D_ik) / r_a]_+^2
  J_b = (1/NT) sum_{t,i} [(delta - D_i^wall) / r_a]_+^2
  J_e = (1/NT) sum_{t,i} |u_i^t|^2 / a_max^2

N(.) is the zero-at-zero regularized norm. The margin delta only penalizes
early; true contacts are counted in swarm.metrics with raw geometry.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from swarm.config import SimParams
from swarm.geometry import swept_geometry, zero_norm


class StepTerms(NamedTuple):
    """Unnormalized per-step sums."""

    progress: jnp.ndarray
    pair: jnp.ndarray
    obstacle: jnp.ndarray
    wall: jnp.ndarray
    energy: jnp.ndarray


class LossTerms(NamedTuple):
    goal: jnp.ndarray
    time: jnp.ndarray
    collision: jnp.ndarray
    obstacle: jnp.ndarray
    boundary: jnp.ndarray
    energy: jnp.ndarray
    total: jnp.ndarray     # lambda-weighted sum, without controller regularization


def initial_scale(x0, goal, floor: float):
    """L0: mean initial distance to the goal center, floored (constant in theta)."""
    return jnp.maximum(jnp.mean(jnp.linalg.norm(x0 - goal, axis=-1)), floor)


def goal_excess(x, goal, goal_radius, eps: float):
    """[N(x_i - g) - R_g]_+ per agent: distance outside the goal disk."""
    return jnp.maximum(zero_norm(x - goal, eps=eps) - goal_radius, 0.0)


def swept_safety(x0, x1, centers, radii, valid, lower, upper, r_a, margin, eps: float):
    """Unnormalized pair / obstacle / wall penalty sums for one step (training only)."""
    pair, obs, wall = swept_geometry(x0, x1, centers, lower, upper, r_a)
    n = x0.shape[0]
    upper_tri = jnp.triu(jnp.ones((n, n), dtype=bool), k=1)
    d_pair = zero_norm(pair, eps=eps)
    d_obs = zero_norm(obs, eps=eps) - radii[None, :] - r_a
    cp = jnp.maximum((2 * r_a + margin - d_pair) / r_a, 0.0) ** 2
    co = jnp.maximum((margin - d_obs) / r_a, 0.0) ** 2
    cb = jnp.maximum((margin - wall) / r_a, 0.0) ** 2
    return (
        jnp.sum(jnp.where(upper_tri, cp, 0.0)),
        jnp.sum(jnp.where(valid[None, :] > 0, co, 0.0)),
        jnp.sum(cb),
    )


def step_terms(x, x_next, u, ep, L0, sim: SimParams) -> StepTerms:
    """Per-step sums for the segment x -> x_next with applied control u."""
    eps, r_a = sim.numerics.eps, sim.agent.radius
    pair, obs, wall = swept_safety(
        x, x_next, ep.centers, ep.radii, ep.valid, ep.lower, ep.upper, r_a, sim.loss.safety_margin, eps
    )
    progress = jnp.sum(goal_excess(x_next, ep.goal, ep.goal_radius, eps)) / L0
    energy = jnp.sum(u * u) / sim.agent.a_max**2
    return StepTerms(progress, pair, obs, wall, energy)


def combine(sums: StepTerms, x_final, ep, L0, n_agents: int, n_steps: int, sim: SimParams) -> LossTerms:
    """Normalize the accumulated step sums and add the terminal goal term."""
    w = sim.loss.weights
    nt = n_agents * n_steps
    goal = jnp.mean((goal_excess(x_final, ep.goal, ep.goal_radius, sim.numerics.eps) / L0) ** 2)
    time, coll, obst, bound, energy = (s / nt for s in (sums.progress, sums.pair, sums.obstacle,
                                                         sums.wall, sums.energy))
    total = (w.goal * goal + w.time * time + w.collision * coll + w.obstacle * obst
             + w.boundary * bound + w.energy * energy)
    return LossTerms(goal, time, coll, obst, bound, energy, total)
