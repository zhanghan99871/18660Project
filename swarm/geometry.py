"""Differentiable geometry primitives (project doc §3.1, §5.2, §9.2).

Every norm that takes part in autodiff goes through `safe_norm` (R_eps) or
`zero_norm` (N_eps). Never use `jnp.linalg.norm(z) + eps`: its gradient is NaN
at z = 0. Pure evaluation metrics use raw geometry instead (see the
contact checks in Phase 2), so regularization never changes a contact decision.
"""

from __future__ import annotations

import jax.numpy as jnp

DEFAULT_EPS = 1e-6


def safe_norm(x, axis: int = -1, keepdims: bool = False, eps: float = DEFAULT_EPS):
    """R_eps(z) = sqrt(|z|^2 + eps^2): smooth everywhere, >= eps."""
    return jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=keepdims) + eps**2)


def zero_norm(x, axis: int = -1, keepdims: bool = False, eps: float = DEFAULT_EPS):
    """N_eps(z) = R_eps(z) - eps: zero at z = 0, underestimates |z| by at most eps."""
    return safe_norm(x, axis, keepdims, eps) - eps


def unit_direction(x, eps: float = DEFAULT_EPS):
    """U_eps(z) = z / R_eps(z): zero at z = 0, norm < 1 everywhere."""
    return x / safe_norm(x, keepdims=True, eps=eps)


def clipnorm(x, max_norm, eps: float = DEFAULT_EPS):
    """z * min(1, a / R_eps(z)): output norm <= a, finite gradient at zero."""
    return x * jnp.minimum(1.0, max_norm / safe_norm(x, keepdims=True, eps=eps))


def smoothstep(q):
    """S(q) = q^2 (3 - 2q) on [0, 1], with the input clipped to [0, 1]."""
    q = jnp.clip(q, 0.0, 1.0)
    return q * q * (3.0 - 2.0 * q)


def swept_closest(p0, p1):
    """Closest point to the origin on the segment p0 -> p1 (relative positions).

    s* = clip(-p0.q / q.q, 0, 1) with q = p1 - p0, and s* = 0 when q = 0. The
    zero-motion branch uses a safe denominator, so neither branch divides by
    zero and gradients stay finite. Shapes: (..., 2) -> (..., 2).
    """
    q = p1 - p0
    q2 = jnp.sum(q * q, axis=-1)
    moving = q2 > 0.0
    den = jnp.where(moving, q2, 1.0)
    s = jnp.where(moving, jnp.clip(-jnp.sum(p0 * q, axis=-1) / den, 0.0, 1.0), 0.0)
    return p0 + s[..., None] * q


def wall_clearance(x, lower, upper, r_a):
    """Signed clearance of agent disks from a rectangle's walls.

    min over the four axis-aligned signed distances, minus the agent radius.
    Negative means the disk crosses a wall. x: (..., 2) -> (...,).
    """
    return jnp.min(jnp.concatenate([x - lower, upper - x], axis=-1), axis=-1) - r_a


def swept_geometry(x0, x1, centers, lower, upper, r_a):
    """Per-step swept geometry for N agents moving linearly from x0 to x1.

    Returns
      pair: (N, N, 2) closest relative position x_i - x_j along the step,
      obs:  (N, K, 2) closest position relative to each obstacle center,
      wall: (N,)      min wall clearance (attained at a segment endpoint).
    Obstacles are static, so their relative segment is x_i(t) - c_k.
    """
    pair = swept_closest(x0[:, None, :] - x0[None, :, :], x1[:, None, :] - x1[None, :, :])
    obs = swept_closest(x0[:, None, :] - centers[None, :, :], x1[:, None, :] - centers[None, :, :])
    wall = jnp.minimum(wall_clearance(x0, lower, upper, r_a), wall_clearance(x1, lower, upper, r_a))
    return pair, obs, wall
