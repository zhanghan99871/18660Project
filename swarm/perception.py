"""Local perception and the three Boids interactions (doc §3.2–3.3, §9.2).

Two neighbor radii: r (alignment, cohesion, obstacle sensing) and r_sep < r
(separation). Weights are either hard indicators or a compact-support
smoothstep of relative width eta, which is exactly zero beyond the radius and
exactly one inside (1 - eta) * radius, so soft weights never use information
from outside the sensing range. eta -> 0 recovers the hard model.

`hard` must be a Python bool (static under jit). `eta` may be a traced scalar,
so changing it between continuation stages does not recompile.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from swarm.geometry import DEFAULT_EPS, smoothstep


def neighbor_weights(dist, radius, eta, hard: bool):
    """Hard: 1[dist < radius]. Soft: S((radius - dist) / (eta * radius))."""
    if hard:
        return (dist < radius).astype(dist.dtype)
    return smoothstep((radius - dist) / (eta * radius))


class Neighborhood(NamedTuple):
    delta: jnp.ndarray    # (N, N, 2) x_i - x_j
    d2: jnp.ndarray       # (N, N) squared distances
    a: jnp.ndarray        # (N, N) weights within r, zero diagonal
    s: jnp.ndarray        # (N, N) weights within r_sep, zero diagonal
    count: jnp.ndarray    # (N,) n_i = sum_j a_ij


def neighborhood(x, r, r_sep, eta, hard: bool, eps: float = DEFAULT_EPS) -> Neighborhood:
    n = x.shape[0]
    delta = x[:, None, :] - x[None, :, :]
    d2 = jnp.sum(delta * delta, axis=-1)
    dist = jnp.sqrt(d2 + eps**2)          # R_eps: smooth at coincidence
    off = 1.0 - jnp.eye(n, dtype=x.dtype)
    a = neighbor_weights(dist, r, eta, hard) * off
    s = neighbor_weights(dist, r_sep, eta, hard) * off
    return Neighborhood(delta, d2, a, s, a.sum(axis=1))


def separation(nb: Neighborhood, r_sep, ell_s):
    """f_s = r_sep * sum_j s_ij (x_i - x_j) / (|x_i - x_j|^2 + l_s^2).

    Summed, not averaged, so a close neighbor is never diluted by others. Each
    pair contributes at most r_sep / (2 l_s); the Jacobian at coincidence is
    (r_sep / l_s^2) I.
    """
    return r_sep * jnp.sum(nb.s[..., None] * nb.delta / (nb.d2 + ell_s**2)[..., None], axis=1)


def alignment(nb: Neighborhood, v):
    """f_a = (1 / (n_i + 1)) sum_j a_ij (v_j - v_i) (self counted with weight 1)."""
    m = (nb.count + 1.0)[:, None]
    return (nb.a @ v - nb.count[:, None] * v) / m


def cohesion(nb: Neighborhood):
    """f_c = (1 / (n_i + 1)) sum_j a_ij (x_j - x_i); |f_c| < r."""
    m = (nb.count + 1.0)[:, None]
    return -jnp.sum(nb.a[..., None] * nb.delta, axis=1) / m


def local_interactions(x, v, r, r_sep, eta, ell_s, hard: bool, eps: float = DEFAULT_EPS):
    """Returns (separation, alignment, cohesion, neighborhood)."""
    nb = neighborhood(x, r, r_sep, eta, hard, eps)
    return separation(nb, r_sep, ell_s), alignment(nb, v), cohesion(nb), nb
