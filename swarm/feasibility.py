"""Map feasibility checks (project doc §8.1).

A map is accepted only if:
  * the goal radius satisfies the capacity rule R_g >= c * r_sep * sqrt(N);
  * the start box (shrunk by r_a) and the goal disk lie inside the domain;
  * no obstacle comes within r_a + delta of the start box or overlaps the goal;
  * a path exists from the start-box center to the goal center on an occupancy
    grid where obstacles and walls are inflated by r_a + delta.

The connectivity test labels 4-connected free cells (equivalent to a BFS, but in
C via scipy.ndimage), so it is fast enough to run on every generated map.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import ndimage

from swarm.config import SimParams
from swarm.maps import MapSpec, point_box_distance


@dataclasses.dataclass(frozen=True)
class FeasibilityReport:
    ok: bool
    reasons: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Grid:
    """Cell-center coordinates of a regular grid covering the domain."""

    xs: np.ndarray   # (W,)
    ys: np.ndarray   # (H,)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.ys), len(self.xs)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.xs, self.ys)   # each (H, W)

    def index(self, p) -> tuple[int, int]:
        """(row, col) of the cell whose center is nearest to point p."""
        return int(np.argmin(np.abs(self.ys - p[1]))), int(np.argmin(np.abs(self.xs - p[0])))


def make_grid(m: MapSpec, resolution: float) -> Grid:
    nx, ny = (int(np.ceil(s / resolution)) for s in m.size)
    xs = m.lower[0] + (np.arange(nx) + 0.5) * (m.size[0] / nx)
    ys = m.lower[1] + (np.arange(ny) + 0.5) * (m.size[1] / ny)
    return Grid(xs, ys)


def clearance_grid(m: MapSpec, resolution: float, include_walls: bool = True):
    """Exact signed clearance of each cell center from obstacles (and walls).

    Returns (clearance (H, W), grid). Negative inside an obstacle; +inf for an
    open map without walls.
    """
    grid = make_grid(m, resolution)
    X, Y = grid.mesh()
    if include_walls:
        clr = np.minimum.reduce([X - m.lower[0], m.upper[0] - X, Y - m.lower[1], m.upper[1] - Y])
    else:
        clr = np.full(X.shape, np.inf)
    for (cx, cy), r in zip(m.centers, m.radii):   # loop keeps memory at O(H W)
        np.minimum(clr, np.hypot(X - cx, Y - cy) - r, out=clr)
    return clr, grid


def same_component(free: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if cells a and b are free and 4-connected through free cells."""
    labels, _ = ndimage.label(free)
    return bool(labels[a] != 0 and labels[a] == labels[b])


def path_exists(m: MapSpec, inflate: float, resolution: float) -> bool:
    clr, grid = clearance_grid(m, resolution)
    return same_component(clr > inflate, grid.index(m.start_center), grid.index(m.goal))


def check_map(m: MapSpec, sim: SimParams, resolution: float) -> FeasibilityReport:
    r_a = sim.agent.radius
    inflate = r_a + sim.loss.safety_margin
    reasons: list[str] = []

    if m.goal_radius < sim.goal_radius(m.n_agents) - 1e-12:
        reasons.append(
            f"goal radius {m.goal_radius:.3f} below capacity rule {sim.goal_radius(m.n_agents):.3f}"
        )
    if np.any(m.start_lo < m.lower + r_a) or np.any(m.start_hi > m.upper - r_a):
        reasons.append("start box leaves the domain")
    if np.any(m.goal - m.goal_radius < m.lower) or np.any(m.goal + m.goal_radius > m.upper):
        reasons.append("goal disk leaves the domain")
    if m.n_obstacles:
        to_box = point_box_distance(m.centers, m.start_lo, m.start_hi) - m.radii
        if np.any(to_box < inflate):
            reasons.append("obstacle within r_a + delta of the start box")
        to_goal = np.linalg.norm(m.centers - m.goal, axis=-1) - m.radii
        if np.any(to_goal < m.goal_radius):
            reasons.append("obstacle overlaps the goal disk")
    if not reasons and not path_exists(m, inflate, resolution):
        reasons.append("no collision-free path from start to goal")
    return FeasibilityReport(ok=not reasons, reasons=tuple(reasons))
