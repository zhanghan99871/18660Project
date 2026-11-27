"""Map descriptor psi(E) for Exp3 (doc §8.1), plus its standardizer.

psi(E) uses global map information but is only used offline to pick parameters;
the runtime controller stays strictly local.

Feature 3, the "minimum gap in the start-goal corridor", is computed as the
maximin passage width: the widest clearance w such that some path from the start
box to the goal disk keeps distance >= w/2 from every obstacle and boundary.
It equals the narrowest obstacle-obstacle or obstacle-boundary gap on the best
route. A literal min over pairwise gaps fails on bottleneck walls: the circles
in a wall chain overlap, and pairwise gaps between a wall circle and a boundary
look open even though other circles fill them.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy import ndimage

from swarm.config import SimParams
from swarm.feasibility import clearance_grid
from swarm.maps import MapSpec

DESCRIPTOR_NAMES = (
    "n_obstacles",
    "obstacle_area_fraction",
    "passage_width",          # maximin passage width / sensor radius r
    "start_goal_distance",    # |g - start center| / domain side
    "log_n_agents",
    "initial_density",        # N / start-box area
)


def obstacle_area_fraction(m: MapSpec, resolution: float) -> float:
    """Fraction of the domain covered by the union of obstacles (grid estimate)."""
    if m.n_obstacles == 0:
        return 0.0
    clr, _ = clearance_grid(m, resolution, include_walls=False)
    return float(np.mean(clr < 0.0))


def passage_width(m: MapSpec, resolution: float, cap: float, iters: int = 30) -> float:
    """Maximin passage width between the start box and the goal disk, capped at `cap`.

    Binary search on the clearance threshold tau: the start and goal regions are
    connected through cells with clearance > tau. Accuracy is about one cell.
    """
    clr, grid = clearance_grid(m, resolution)
    X, Y = grid.mesh()
    start = (X >= m.start_lo[0]) & (X <= m.start_hi[0]) & (Y >= m.start_lo[1]) & (Y <= m.start_hi[1])
    goal = np.hypot(X - m.goal[0], Y - m.goal[1]) <= m.goal_radius
    if not start.any() or not goal.any():
        raise ValueError("start box or goal disk smaller than one grid cell")

    def connected(tau: float) -> bool:
        labels, _ = ndimage.label((clr > tau) | start | goal)
        return bool(np.intersect1d(labels[start], labels[goal]).size)

    lo, hi = 0.0, cap / 2.0
    if connected(hi):
        return cap
    if not connected(lo):
        return 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if connected(mid) else (lo, mid)
    return 2.0 * lo


def map_descriptor(m: MapSpec, sim: SimParams, resolution: float = 0.05, gap_cap: float = 4.0) -> np.ndarray:
    """psi(E) in raw units (standardize with DescriptorScaler)."""
    side = float(np.max(m.size))
    return np.array(
        [
            float(m.n_obstacles),
            obstacle_area_fraction(m, resolution),
            passage_width(m, resolution, gap_cap) / sim.perception.sensor_radius,
            float(np.linalg.norm(m.goal - m.start_center)) / side,
            math.log(m.n_agents),
            m.n_agents / m.start_area,
        ]
    )


@dataclasses.dataclass(frozen=True)
class DescriptorScaler:
    """Per-feature standardization fitted on training maps.

    A feature that is constant on the training maps (e.g. log N when all maps
    have N = 30) gets scale 1, so it maps to 0 in training and keeps its raw
    deviation on other maps instead of exploding.
    """

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, min_std: float = 1e-8) -> "DescriptorScaler":
        X = np.asarray(X, dtype=np.float64)
        std = X.std(axis=0)
        return cls(mean=X.mean(axis=0), scale=np.where(std > min_std, std, 1.0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X) - self.mean) / self.scale

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z) * self.scale + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "DescriptorScaler":
        return cls(mean=np.asarray(d["mean"]), scale=np.asarray(d["scale"]))
