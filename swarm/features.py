"""Local features z_i = [rho, o, gamma, kappa] in [0, 1]^4 (doc §4.2)."""

from __future__ import annotations

import jax.numpy as jnp

from swarm.geometry import DEFAULT_EPS, zero_norm

FEATURE_NAMES = ("density", "obstacle_distance", "goal_distance", "isolation")
N_FEATURES = len(FEATURE_NAMES)


def local_features(count, d_near, d_wall, gamma, cohesion, r, n_ref, eps: float = DEFAULT_EPS):
    """Stack the four local features, shape (N, 4).

    rho   = tanh(n_i / n_ref)                    local density
    o     = clip(min(d_obs, d_wall) / r, 0, 1)   nearest obstacle/wall; 1 if none in range
    gamma = (from the goal observation)          remaining goal distance; 0 inside the goal
    kappa = N_eps(f_c) / r                       distance to the local centroid (edge / isolation)

    d_near is +inf when no valid obstacle exists; clipping keeps o in [0, 1] even
    for agents inside an obstacle or outside the domain.
    """
    density = jnp.tanh(count / n_ref)
    obs = jnp.clip(jnp.minimum(d_near, d_wall) / r, 0.0, 1.0)
    kappa = zero_norm(cohesion, eps=eps) / r
    return jnp.stack([density, obs, gamma, kappa], axis=-1)
