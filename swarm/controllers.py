"""Controller parameterizations sharing one weight map (doc §4.1).

    w_i = m * w_max * sigmoid(s_i)

| kind     | pre-activation s_i            | parameters          | dim (full mask) |
|----------|-------------------------------|---------------------|-----------------|
| fixed    | b                             | b                   | 5               |
| adaptive | A (z_i - mu_z) + b            | A (5x4), b          | 25              |
| time     | A (omega(t) - mu_omega) + b   | A (5x4), b          | 25              |

omega(t) = [t/T, (t/T)^2, sin(2 pi t/T), cos(2 pi t/T)] uses the *clock*
horizon T (the full horizon, also during horizon continuation).

The behavior mask m (separation, alignment, cohesion, goal, obstacle) is fixed
and not optimized. Masked behaviors have weight exactly 0, and their rows of
(A, b) are not parameters. Optimizers see a flat vector theta holding only the
active entries: fixed -> b[active]; adaptive/time -> [A[active].ravel(), b[active]].
With A = 0 the adaptive controller equals the fixed one with the same b.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from swarm.config import BEHAVIORS, SimParams
from swarm.features import N_FEATURES

CONTROLLER_KINDS = ("fixed", "adaptive", "time")
N_BEHAVIORS = len(BEHAVIORS)
N_TIME_FEATURES = 4

MASKS = {
    "full": (1, 1, 1, 1, 1),
    "no_alignment": (1, 0, 1, 1, 1),
    "no_cohesion": (1, 1, 0, 1, 1),
    "no_alignment_cohesion": (1, 0, 0, 1, 1),
}


@dataclasses.dataclass(frozen=True)
class ControllerSpec:
    kind: str = "adaptive"
    mask: tuple[int, ...] = MASKS["full"]

    def __post_init__(self):
        if self.kind not in CONTROLLER_KINDS:
            raise ValueError(f"controller kind must be one of {CONTROLLER_KINDS}, got {self.kind!r}")
        mask = tuple(int(m) for m in self.mask)
        if len(mask) != N_BEHAVIORS or any(m not in (0, 1) for m in mask):
            raise ValueError(f"mask must be {N_BEHAVIORS} zeros/ones, got {self.mask!r}")
        if not any(mask):
            raise ValueError("mask disables every behavior")
        object.__setattr__(self, "mask", mask)

    @property
    def active(self) -> tuple[int, ...]:
        return tuple(i for i, m in enumerate(self.mask) if m)

    @property
    def n_active(self) -> int:
        return len(self.active)

    @property
    def n_inputs(self) -> int:
        return {"fixed": 0, "adaptive": N_FEATURES, "time": N_TIME_FEATURES}[self.kind]

    @property
    def n_params(self) -> int:
        return self.n_active * (self.n_inputs + 1)

    @property
    def has_matrix(self) -> bool:
        return self.n_inputs > 0


# --------------------------------------------------------------------------- #
# Flat parameter vector <-> (A, b)
# --------------------------------------------------------------------------- #
def unpack(theta, spec: ControllerSpec):
    """theta (n_params,) -> (A (5, n_inputs), b (5,)), zeros in masked rows."""
    theta = jnp.asarray(theta)
    if theta.shape != (spec.n_params,):
        raise ValueError(f"theta must have shape ({spec.n_params},), got {theta.shape}")
    rows = np.asarray(spec.active)
    split = spec.n_active * spec.n_inputs
    A = jnp.zeros((N_BEHAVIORS, spec.n_inputs), theta.dtype)
    if spec.has_matrix:
        A = A.at[rows].set(theta[:split].reshape(spec.n_active, spec.n_inputs))
    b = jnp.zeros(N_BEHAVIORS, theta.dtype).at[rows].set(theta[split:])
    return A, b


def pack(A, b, spec: ControllerSpec):
    """Inverse of `unpack`: keep only the active rows."""
    rows = np.asarray(spec.active)
    b = jnp.asarray(b)
    parts = [jnp.asarray(A)[rows].reshape(-1)] if spec.has_matrix else []
    return jnp.concatenate(parts + [b[rows]])


def bounds(spec: ControllerSpec, sim: SimParams) -> tuple[np.ndarray, np.ndarray]:
    """Search box shared by every optimizer: [-4, 4]^d by default."""
    lo, hi = sim.controller.search_box
    return np.full(spec.n_params, lo), np.full(spec.n_params, hi)


def regularizer(theta, spec: ControllerSpec):
    """||A||_F^2 (zero for the fixed controller)."""
    if not spec.has_matrix:
        return jnp.zeros((), jnp.asarray(theta).dtype)
    A, _ = unpack(theta, spec)
    return jnp.sum(A * A)


def warm_start(b_logits, spec: ControllerSpec):
    """theta with A = 0 and the given 5 logits (doc §4.1: theta_A^0 = (0, b_F*))."""
    b = jnp.asarray(b_logits)
    if b.shape != (N_BEHAVIORS,):
        raise ValueError("b_logits must hold all 5 behavior logits")
    return pack(jnp.zeros((N_BEHAVIORS, spec.n_inputs), b.dtype), b, spec)


def logits_from_weights(weights, w_max) -> np.ndarray:
    """Inverse of w = w_max * sigmoid(b), for weights strictly inside (0, w_max)."""
    p = np.asarray(weights, dtype=np.float64) / np.asarray(w_max, dtype=np.float64)
    if np.any(p <= 0) or np.any(p >= 1):
        raise ValueError("weights must lie strictly between 0 and w_max")
    return np.log(p / (1 - p))


def weights_from_logits(b, w_max, mask=MASKS["full"]) -> np.ndarray:
    return np.asarray(mask) * np.asarray(w_max) / (1 + np.exp(-np.asarray(b, dtype=np.float64)))


# --------------------------------------------------------------------------- #
# Pre-activation and weights
# --------------------------------------------------------------------------- #
def time_basis(t, clock_horizon: int):
    """omega(t) for a (possibly traced) step index t."""
    s = jnp.asarray(t, dtype=jnp.result_type(float)) / clock_horizon
    return jnp.stack([s, s * s, jnp.sin(2 * jnp.pi * s), jnp.cos(2 * jnp.pi * s)])


def time_basis_mean(clock_horizon: int) -> np.ndarray:
    """mu_omega: the mean of omega(t) over t = 0..T-1 (a constant per T)."""
    s = np.arange(clock_horizon) / clock_horizon
    return np.stack([s, s * s, np.sin(2 * np.pi * s), np.cos(2 * np.pi * s)], axis=-1).mean(axis=0)


def preactivation(theta, spec: ControllerSpec, z, t, mu_z, clock_horizon: int):
    """s (N, 5) for features z (N, 4) at step t."""
    A, b = unpack(theta, spec)
    n = z.shape[0]
    if spec.kind == "fixed":
        return jnp.broadcast_to(b, (n, N_BEHAVIORS))
    if spec.kind == "adaptive":
        return (z - mu_z) @ A.T + b
    omega = time_basis(t, clock_horizon) - jnp.asarray(time_basis_mean(clock_horizon), z.dtype)
    return jnp.broadcast_to(A @ omega + b, (n, N_BEHAVIORS))


def behavior_weights(s, w_max, mask):
    """w = m * w_max * sigmoid(s); masked behaviors are exactly zero."""
    return jnp.asarray(mask, s.dtype) * jnp.asarray(w_max, s.dtype) * jax.nn.sigmoid(s)
