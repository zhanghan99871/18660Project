"""Goal, obstacle and boundary forces (doc §3.3), and the force scales nu_j (§3.4).

Obstacles are circles padded to a fixed count K; `valid` (K,) is 1 for real
obstacles and 0 for padding, whose (finite) geometry is ignored.

The boundary force is a normal-only push with the same [d_s - d]_+ profile as
obstacles, based on the signed clearance from each wall. It is added to the
obstacle force and shares the obstacle weight w_o.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from swarm.config import SimParams
from swarm.geometry import DEFAULT_EPS, unit_direction, zero_norm

# Behavior order used everywhere: separation, alignment, cohesion, goal, obstacle.
SEP, ALI, COH, GOAL, OBS = range(5)

# Inward normals of the left, bottom, right and top walls.
_WALL_NORMALS = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])


def force_scales(sim: SimParams) -> np.ndarray:
    """nu = (1, v_max, r, v_max, d_s): makes the five behaviors comparable."""
    return np.array(
        [1.0, sim.agent.v_max, sim.perception.sensor_radius, sim.agent.v_max, sim.forces.obstacle_influence]
    )


def goal_force(x, v, g, v_max, ell_g, eps: float = DEFAULT_EPS):
    """f_g = v_des - v, v_des = v_max tanh(R(g - x) / l_g) U(g - x).

    The desired speed decays at the goal *center*, so arrived agents keep a weak
    inward pull. `g` and `ell_g` broadcast against x (N, 2).
    """
    gvec = g - x
    dist = jnp.sqrt(jnp.sum(gvec * gvec, axis=-1, keepdims=True) + eps**2)
    return v_max * jnp.tanh(dist / ell_g) * gvec / dist - v


def obstacle_force(x, heading, centers, radii, valid, r_a, d_s, k_t, c_t, b_t, eps: float = DEFAULT_EPS):
    """Normal push plus a tangential term that steers around the goal-ward side.

        n_i = sum_k [d_s - d_ik]_+ n_ik                       (= -grad of U_obs)
        f_o = n_i + k_t |n_i| tanh(c_t (t_i . q_i + b_t)) t_i,   t_i = R90 n_i / |n_i|

    with d_ik the clearance of agent i's disk from obstacle k. The bias b_t
    breaks the head-on symmetry.

    Deviation from doc §3.3: the doc applies the tangential term per obstacle,
    sum_k push_k (n_ik + k_t s_ik t_ik). For an isolated obstacle the two are
    identical. For a wall built from overlapping circles, the per-circle
    tangents of the neighboring circles point *through* the wall, and agents
    tunnel through it. Using the tangent of the combined repulsion slides agents
    along the wall instead.

    Returns (force (N, 2), nearest valid clearance (N,), +inf if none).
    """
    diff = x[:, None, :] - centers[None, :, :]                     # (N, K, 2)
    norm = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + eps**2)
    n_hat = diff / norm[..., None]
    clearance = norm - radii[None, :] - r_a
    push = jnp.maximum(d_s - clearance, 0.0) * valid[None, :]
    normal = jnp.sum(push[..., None] * n_hat, axis=1)              # (N, 2)
    magnitude = zero_norm(normal, eps=eps)
    n_dir = unit_direction(normal, eps)
    t_dir = jnp.stack([-n_dir[:, 1], n_dir[:, 0]], axis=-1)
    side = jnp.tanh(c_t * (jnp.sum(t_dir * heading, axis=-1) + b_t))
    force = normal + (k_t * magnitude * side)[:, None] * t_dir
    d_near = jnp.min(jnp.where(valid[None, :] > 0, clearance, jnp.inf), axis=1)
    return force, d_near


def boundary_force(x, lower, upper, r_a, d_s):
    """Normal push [d_s - d_w]_+ from each wall, d_w = signed clearance of the disk.

    Returns (force (N, 2), min wall clearance (N,)).
    """
    clear = jnp.concatenate([x - lower, upper - x], axis=-1) - r_a   # left, bottom, right, top
    push = jnp.maximum(d_s - clear, 0.0)
    return push @ jnp.asarray(_WALL_NORMALS, dtype=x.dtype), jnp.min(clear, axis=-1)
