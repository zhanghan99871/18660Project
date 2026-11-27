"""Goal observations: the only place the true goal enters the controller (doc §4.5).

Informed agents (h_i = 1) see the goal center and radius. Uninformed agents see
neither: their goal force is zero, their goal-distance feature is a fixed
missing value, and their obstacle-avoidance heading is the local mean velocity.
No other code path may read the goal for control. The training loss and the
evaluator hold the true goal separately (centralized training, decentralized
execution).

With every agent informed this reduces exactly to the fully informed model.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from swarm.config import SimParams
from swarm.forces import goal_force
from swarm.geometry import unit_direction, zero_norm


class GoalObservation(NamedTuple):
    goal_force: jnp.ndarray   # (N, 2), zero for uninformed agents
    gamma: jnp.ndarray        # (N,) remaining goal distance feature in [0, 1)
    heading: jnp.ndarray      # (N, 2) direction used by the tangential obstacle term


def local_goal_observation(x, v, goal, goal_radius, informed, a, sim: SimParams, gamma_missing: float):
    """Build each agent's goal observation.

    x, v: (N, 2); goal: (2,); goal_radius: scalar; informed: (N,) bool;
    a: (N, N) neighbor weights of the current (hard or soft) model, zero diagonal.
    gamma_missing: gamma for uninformed agents (0 by default; 1 = "still en route").
    """
    eps = sim.numerics.eps
    h = informed[:, None]
    g_obs = jnp.where(h, goal[None, :], x)
    rg_obs = jnp.where(informed, goal_radius, 0.0)
    # l_g depends on the true R_g, so uninformed agents get a placeholder scale;
    # their goal output is masked to zero below.
    ell_g = jnp.where(informed, sim.goal.length_scale_factor * goal_radius, 1.0)[:, None]
    f_goal = jnp.where(h, goal_force(x, v, g_obs, sim.agent.v_max, ell_g, eps), 0.0)

    remaining = jnp.maximum(zero_norm(g_obs - x, eps=eps) - rg_obs, 0.0)
    gamma = jnp.where(informed, jnp.tanh(remaining / sim.features.goal_distance_scale), gamma_missing)

    local_v = (v + a @ v) / (1.0 + jnp.sum(a, axis=1, keepdims=True))
    heading = jnp.where(h, unit_direction(g_obs - x, eps), unit_direction(local_v, eps))
    return GoalObservation(f_goal, gamma, heading)
