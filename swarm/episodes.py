"""Initial states, informed masks and padded episode batches (doc §4.5, §5.1, §8.1).

An episode is xi = (E, x0, v0, h). Here we store the informed *order* (a random
permutation) instead of h itself: the informed set for fraction p is the first
n_I = max(1, floor(p N + 0.5)) agents of that order, so masks for different p are
nested and paired comparisons across p reuse the same episodes.

Initial-state sampling never reads the goal or the obstacles. It reads only the
start box, and draws from its own random streams, so x0, v0 and the order are
independent of the goal by construction.
"""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple, Sequence

import numpy as np

from swarm.maps import SCENARIO_ID, SCENARIOS, MapSpec
from swarm.rng import Stream, stream_rng


# --------------------------------------------------------------------------- #
# Initial state
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True, eq=False)
class InitialState:
    x0: np.ndarray      # (N, 2)
    v0: np.ndarray      # (N, 2)
    order: np.ndarray   # (N,) permutation; informed agents are a prefix


@dataclasses.dataclass(frozen=True)
class VelocityParams:
    mode: str           # "zero" | "isotropic"
    max_speed: float

    def __post_init__(self):
        if self.mode not in ("zero", "isotropic"):
            raise ValueError(f"initial velocity mode must be zero or isotropic, got {self.mode!r}")
        if self.max_speed < 0:
            raise ValueError("max_speed must be non-negative")


def sample_positions(
    rng: np.random.Generator,
    start_lo: np.ndarray,
    start_hi: np.ndarray,
    n: int,
    spacing: float,
    r_a: float,
    max_attempts_per_agent: int = 2000,
) -> np.ndarray:
    """Dart throwing: n points in the start box (shrunk by r_a), pairwise >= spacing."""
    lo, hi = start_lo + r_a, start_hi - r_a
    if np.any(hi <= lo):
        raise ValueError("start box too small for the agent radius")
    pts = np.empty((n, 2))
    count, attempts = 0, 0
    budget = max_attempts_per_agent * n
    while count < n:
        if attempts >= budget:
            raise RuntimeError(f"placed only {count}/{n} agents; start box too small for spacing")
        attempts += 1
        p = rng.uniform(lo, hi)
        if count == 0 or np.min(np.sum((pts[:count] - p) ** 2, axis=-1)) >= spacing**2:
            pts[count] = p
            count += 1
    return pts


def sample_velocities(rng: np.random.Generator, n: int, params: VelocityParams) -> np.ndarray:
    """v0 = 0 or isotropic (uniform direction, speed ~ U[0, max_speed]); never goal-aware."""
    if params.mode == "zero":
        return np.zeros((n, 2))
    angle = rng.uniform(0.0, 2 * math.pi, n)
    speed = rng.uniform(0.0, params.max_speed, n)
    return np.stack([speed * np.cos(angle), speed * np.sin(angle)], axis=-1)


def sample_initial_state(
    m: MapSpec,
    seed: int,
    split: str,
    map_index: int,
    episode: int,
    spacing: float,
    r_a: float,
    velocity: VelocityParams,
) -> InitialState:
    """Initial state for one episode; depends only on the start box and the streams."""
    n = m.n_agents
    x0 = sample_positions(
        stream_rng(seed, split, map_index, Stream.INIT_POS, episode), m.start_lo, m.start_hi, n, spacing, r_a
    )
    v0 = sample_velocities(stream_rng(seed, split, map_index, Stream.INIT_VEL, episode), n, velocity)
    order = stream_rng(seed, split, map_index, Stream.INFORMED, episode).permutation(n)
    return InitialState(x0=x0, v0=v0, order=order)


# --------------------------------------------------------------------------- #
# Informed masks
# --------------------------------------------------------------------------- #
def n_informed(p: float, n: int) -> int:
    """n_I = max(1, floor(p N + 0.5)) (§4.5)."""
    if not 0.0 < p <= 1.0:
        raise ValueError(f"informed fraction must be in (0, 1], got {p}")
    return max(1, math.floor(p * n + 0.5))


def informed_mask(order: np.ndarray, p: float) -> np.ndarray:
    """Boolean mask (..., N): agent i is informed iff its rank in `order` < n_I."""
    order = np.asarray(order)
    rank = np.argsort(order, axis=-1)   # inverse permutation
    return rank < n_informed(p, order.shape[-1])


# --------------------------------------------------------------------------- #
# Padded batches
# --------------------------------------------------------------------------- #
class EpisodeBatch(NamedTuple):
    """B episodes with equal N, obstacles padded to K. A JAX pytree (NamedTuple).

    Padded obstacles have center (0, 0), radius 0 and valid 0: finite values that
    the simulator masks out with `valid`.
    """

    lower: np.ndarray          # (B, 2)
    upper: np.ndarray          # (B, 2)
    centers: np.ndarray        # (B, K, 2)
    radii: np.ndarray          # (B, K)
    valid: np.ndarray          # (B, K) float 0/1
    start_lo: np.ndarray       # (B, 2)
    start_hi: np.ndarray       # (B, 2)
    goal: np.ndarray           # (B, 2)
    goal_radius: np.ndarray    # (B,)
    x0: np.ndarray             # (B, N, 2)
    v0: np.ndarray             # (B, N, 2)
    order: np.ndarray          # (B, N) int
    scenario_id: np.ndarray    # (B,) int, index into maps.SCENARIOS
    map_index: np.ndarray      # (B,) int
    episode_index: np.ndarray  # (B,) int, episode number within its map
    role: np.ndarray           # (B,) int, index into the split's role names

    @property
    def n_episodes(self) -> int:
        return self.x0.shape[0]

    @property
    def n_agents(self) -> int:
        return self.x0.shape[1]

    @property
    def k_max(self) -> int:
        return self.radii.shape[1]

    def select(self, idx) -> "EpisodeBatch":
        """Sub-batch by index array or boolean mask."""
        return EpisodeBatch(*(np.asarray(a)[idx] for a in self))

    def informed(self, p: float) -> np.ndarray:
        """(B, N) informed masks for fraction p (nested across p)."""
        return informed_mask(self.order, p)

    def with_k_max(self, k: int) -> "EpisodeBatch":
        """Re-pad obstacles to exactly k slots, e.g. so every split shares one
        compiled shape. Only trailing all-padding slots can be dropped."""
        valid = np.asarray(self.valid)
        need = max(1, int(valid.sum(axis=1).max()))
        if k < need:
            raise ValueError(f"k={k} smaller than the largest obstacle count {need}")
        if k <= self.k_max:
            if np.any(valid[:, k:] > 0):
                raise ValueError("cannot drop slots that hold real obstacles")
            return self._replace(centers=self.centers[:, :k], radii=self.radii[:, :k], valid=valid[:, :k])
        extra = k - self.k_max
        B = self.n_episodes
        return self._replace(
            centers=np.concatenate([self.centers, np.zeros((B, extra, 2))], axis=1),
            radii=np.concatenate([self.radii, np.zeros((B, extra))], axis=1),
            valid=np.concatenate([valid, np.zeros((B, extra))], axis=1),
        )


def stack_episodes(
    maps: Sequence[MapSpec],
    states: Sequence[InitialState],
    map_index: Sequence[int],
    episode_index: Sequence[int],
    role: Sequence[int] | None = None,
    k_max: int | None = None,
) -> EpisodeBatch:
    """Stack equal-N episodes; pad obstacles to k_max (default: largest K, >= 1)."""
    if not maps or len(maps) != len(states):
        raise ValueError("need one MapSpec per InitialState, at least one")
    n_agents = {m.n_agents for m in maps} | {s.x0.shape[0] for s in states}
    if len(n_agents) != 1:
        raise ValueError(f"a batch needs a single agent count, got {sorted(n_agents)}")
    need = max(1, max(m.n_obstacles for m in maps))
    k = need if k_max is None else k_max
    if k < need:
        raise ValueError(f"k_max={k} smaller than the largest obstacle count {need}")

    B = len(maps)
    centers = np.zeros((B, k, 2))
    radii = np.zeros((B, k))
    valid = np.zeros((B, k))
    for b, m in enumerate(maps):
        centers[b, : m.n_obstacles] = m.centers
        radii[b, : m.n_obstacles] = m.radii
        valid[b, : m.n_obstacles] = 1.0
    return EpisodeBatch(
        lower=np.stack([m.lower for m in maps]),
        upper=np.stack([m.upper for m in maps]),
        centers=centers,
        radii=radii,
        valid=valid,
        start_lo=np.stack([m.start_lo for m in maps]),
        start_hi=np.stack([m.start_hi for m in maps]),
        goal=np.stack([m.goal for m in maps]),
        goal_radius=np.array([m.goal_radius for m in maps]),
        x0=np.stack([s.x0 for s in states]),
        v0=np.stack([s.v0 for s in states]),
        order=np.stack([s.order for s in states]).astype(np.int64),
        scenario_id=np.array([SCENARIO_ID[m.scenario] for m in maps], dtype=np.int64),
        map_index=np.asarray(map_index, dtype=np.int64),
        episode_index=np.asarray(episode_index, dtype=np.int64),
        role=np.zeros(B, dtype=np.int64) if role is None else np.asarray(role, dtype=np.int64),
    )


def map_from_batch(batch: EpisodeBatch, b: int) -> MapSpec:
    """Reconstruct the unpadded MapSpec of episode b."""
    keep = np.asarray(batch.valid[b]) > 0
    return MapSpec(
        scenario=SCENARIOS[int(batch.scenario_id[b])],
        n_agents=batch.n_agents,
        lower=batch.lower[b],
        upper=batch.upper[b],
        centers=np.asarray(batch.centers[b])[keep],
        radii=np.asarray(batch.radii[b])[keep],
        start_lo=batch.start_lo[b],
        start_hi=batch.start_hi[b],
        goal=batch.goal[b],
        goal_radius=float(batch.goal_radius[b]),
    )


def to_jax(batch: EpisodeBatch) -> EpisodeBatch:
    """Device copy of every field (float fields follow JAX's default precision)."""
    import jax.numpy as jnp

    return EpisodeBatch(*(jnp.asarray(a) for a in batch))
