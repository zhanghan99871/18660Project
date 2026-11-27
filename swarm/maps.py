"""Map representation and scenario generators (project doc §8.1).

A map is a bounded rectangle with circular obstacles, an axis-aligned start box
and a circular goal. Walls and bottlenecks are chains of overlapping circles, so
all obstacle geometry is circular (§3.1).

Generation is offline NumPy code (rejection sampling, variable obstacle counts).
The JAX simulator only sees the padded, stacked arrays built in
`swarm.episodes`. Feasibility (connectivity, capacity) is checked separately in
`swarm.feasibility`.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable

import numpy as np

from swarm.config import SimParams, from_dict

# Order defines scenario ids stored in datasets; append only.
SCENARIOS = (
    "open",
    "single_obstacle",
    "bottleneck",
    "random_obstacles",
    "dense_random",
    "narrow_bottleneck",
    "large_obstacles",
)
SCENARIO_ID = {name: i for i, name in enumerate(SCENARIOS)}


# --------------------------------------------------------------------------- #
# Map container
# --------------------------------------------------------------------------- #
def _frozen_array(x, shape: tuple[int | None, ...], name: str) -> np.ndarray:
    a = np.array(x, dtype=np.float64)
    if a.ndim != len(shape) or any(s is not None and s != d for s, d in zip(shape, a.shape)):
        raise ValueError(f"{name}: expected shape {shape}, got {a.shape}")
    a.setflags(write=False)
    return a


@dataclasses.dataclass(frozen=True, eq=False)
class MapSpec:
    """One map. Arrays are float64 and read-only; obstacles are unpadded."""

    scenario: str
    n_agents: int
    lower: np.ndarray          # (2,)
    upper: np.ndarray          # (2,)
    centers: np.ndarray        # (K, 2)
    radii: np.ndarray          # (K,)
    start_lo: np.ndarray       # (2,)
    start_hi: np.ndarray       # (2,)
    goal: np.ndarray           # (2,)
    goal_radius: float

    def __post_init__(self):
        if self.scenario not in SCENARIO_ID:
            raise ValueError(f"unknown scenario {self.scenario!r}")
        set_ = lambda k, v: object.__setattr__(self, k, v)
        set_("lower", _frozen_array(self.lower, (2,), "lower"))
        set_("upper", _frozen_array(self.upper, (2,), "upper"))
        set_("centers", _frozen_array(np.reshape(self.centers, (-1, 2)), (None, 2), "centers"))
        set_("radii", _frozen_array(np.reshape(self.radii, (-1,)), (None,), "radii"))
        set_("start_lo", _frozen_array(self.start_lo, (2,), "start_lo"))
        set_("start_hi", _frozen_array(self.start_hi, (2,), "start_hi"))
        set_("goal", _frozen_array(self.goal, (2,), "goal"))
        set_("goal_radius", float(self.goal_radius))
        set_("n_agents", int(self.n_agents))
        if len(self.centers) != len(self.radii):
            raise ValueError("centers and radii disagree on the obstacle count")
        if np.any(self.radii <= 0):
            raise ValueError("obstacle radii must be positive")
        if np.any(self.upper <= self.lower) or np.any(self.start_hi <= self.start_lo):
            raise ValueError("boxes must satisfy hi > lo")

    @property
    def n_obstacles(self) -> int:
        return len(self.radii)

    @property
    def size(self) -> np.ndarray:
        return self.upper - self.lower

    @property
    def start_center(self) -> np.ndarray:
        return 0.5 * (self.start_lo + self.start_hi)

    @property
    def start_area(self) -> float:
        w, h = self.start_hi - self.start_lo
        return float(w * h)

    def replace(self, **changes: Any) -> "MapSpec":
        return dataclasses.replace(self, **changes)


# --------------------------------------------------------------------------- #
# Generation parameters
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class LayoutParams:
    border_margin: float
    init_spacing: float
    start_area_fraction: tuple[float, float]
    start_aspect: tuple[float, float]
    max_start_width_fraction: float
    start_jitter: float
    goal_jitter: float
    min_start_goal_gap: float
    obstacle_clearance: float
    placement_tries: int
    max_map_tries: int

    @classmethod
    def from_config(cls, data: dict) -> "LayoutParams":
        return from_dict(cls, data, "layout")


@dataclasses.dataclass(frozen=True)
class PlacementContext:
    """Everything an obstacle generator may look at (start and goal are fixed)."""

    lower: np.ndarray
    upper: np.ndarray
    start_lo: np.ndarray
    start_hi: np.ndarray
    goal: np.ndarray
    goal_radius: float
    layout: LayoutParams
    count_scale: float = 1.0   # >1 for fixed-density maps (area ratio)

    def clear(self, center: np.ndarray, radius: float) -> bool:
        """Obstacle keeps `obstacle_clearance` from the start box and goal disk."""
        c = self.layout.obstacle_clearance
        if point_box_distance(center, self.start_lo, self.start_hi) < radius + c:
            return False
        return float(np.linalg.norm(center - self.goal)) >= radius + self.goal_radius + c


# --------------------------------------------------------------------------- #
# Small geometry helpers (NumPy)
# --------------------------------------------------------------------------- #
def point_box_distance(p, lo, hi) -> float:
    """Euclidean distance from point(s) p (..., 2) to an axis-aligned box (0 inside)."""
    p = np.asarray(p, dtype=np.float64)
    d = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    return np.sqrt(np.sum(d * d, axis=-1))


# --------------------------------------------------------------------------- #
# Start box and goal
# --------------------------------------------------------------------------- #
def sample_start_box(rng: np.random.Generator, n_agents: int, lower, upper, layout: LayoutParams):
    """Start box on the left side, sized so agents fill `start_area_fraction`."""
    phi = rng.uniform(*layout.start_area_fraction)
    area = n_agents * math.pi * layout.init_spacing**2 / 4.0 / phi
    aspect = rng.uniform(*layout.start_aspect)
    span_y = float(upper[1] - lower[1]) - 2 * layout.border_margin
    max_width = layout.max_start_width_fraction * float(upper[0] - lower[0])
    height = math.sqrt(area * aspect)
    if area / height > max_width:      # large swarms: grow the box vertically instead
        height = area / max_width
    height = min(height, span_y)
    width = area / height
    x0 = float(lower[0]) + layout.border_margin + rng.uniform(0.0, layout.start_jitter)
    y_lo = float(lower[1]) + layout.border_margin + height / 2
    y_hi = float(upper[1]) - layout.border_margin - height / 2
    yc = rng.uniform(y_lo, y_hi) if y_hi > y_lo else 0.5 * (y_lo + y_hi)
    lo = np.array([x0, yc - height / 2])
    hi = np.array([x0 + width, yc + height / 2])
    return lo, hi


def sample_goal(rng: np.random.Generator, goal_radius: float, lower, upper, layout: LayoutParams):
    """Goal disk on the right side, fully inside the domain margin."""
    gx = float(upper[0]) - layout.border_margin - goal_radius - rng.uniform(0.0, layout.goal_jitter)
    y_lo = float(lower[1]) + layout.border_margin + goal_radius
    y_hi = float(upper[1]) - layout.border_margin - goal_radius
    gy = rng.uniform(y_lo, y_hi) if y_hi > y_lo else 0.5 * (y_lo + y_hi)
    return np.array([gx, gy])


# --------------------------------------------------------------------------- #
# Obstacle generators: (rng, ctx, params) -> (centers, radii) or None
# --------------------------------------------------------------------------- #
Obstacles = tuple[np.ndarray, np.ndarray]


def _open(rng, ctx: PlacementContext, params: dict) -> Obstacles:
    return np.zeros((0, 2)), np.zeros((0,))


def _single(rng, ctx: PlacementContext, params: dict) -> Obstacles | None:
    s = 0.5 * (ctx.start_lo + ctx.start_hi)
    d = ctx.goal - s
    u = d / np.linalg.norm(d)
    normal = np.array([-u[1], u[0]])
    for _ in range(ctx.layout.placement_tries):
        radius = rng.uniform(*params["radius"])
        t = rng.uniform(*params["position"])
        off = rng.uniform(*params["lateral_offset"])
        c = s + t * d + off * normal
        if ctx.clear(c, radius):
            return c[None, :], np.array([radius])
    return None


def _scaled_count_range(count, scale: float) -> tuple[int, int]:
    lo, hi = count
    if scale != 1.0:
        lo, hi = max(1, round(lo * scale)), max(1, round(hi * scale))
    return int(lo), int(max(lo, hi))


def _random(rng, ctx: PlacementContext, params: dict) -> Obstacles | None:
    k_lo, k_hi = _scaled_count_range(params["count"], ctx.count_scale)
    n_obs = int(rng.integers(k_lo, k_hi + 1))
    x_lo, x_hi = float(ctx.start_hi[0]), float(ctx.goal[0] - ctx.goal_radius)
    if x_hi <= x_lo:
        return None
    centers, radii = [], []
    for _ in range(n_obs):
        for _ in range(ctx.layout.placement_tries):
            radius = rng.uniform(*params["radius"])
            c = np.array([rng.uniform(x_lo, x_hi), rng.uniform(ctx.lower[1], ctx.upper[1])])
            if ctx.clear(c, radius):
                centers.append(c)
                radii.append(radius)
                break
        else:
            return None
    return np.array(centers).reshape(-1, 2), np.array(radii)


def wall_circle_ys(gap_center: float, gap: float, radius: float, spacing: float, y_min: float, y_max: float):
    """y coordinates of wall circles leaving a surface-to-surface `gap`.

    The two circles bounding the gap sit exactly gap/2 + radius from its center;
    the chains then continue outward every `spacing` and end with a circle
    centered on the boundary, so no hidden gap is left at the walls.
    """
    below_top = gap_center - gap / 2 - radius
    above_bot = gap_center + gap / 2 + radius
    below = list(np.arange(below_top, y_min, -spacing))
    if not below or below[-1] > y_min + 1e-9:
        below.append(y_min)
    above = list(np.arange(above_bot, y_max, spacing))
    if not above or above[-1] < y_max - 1e-9:
        above.append(y_max)
    return np.array(sorted(below) + above)


def _wall(rng, ctx: PlacementContext, params: dict) -> Obstacles | None:
    radius, spacing = params["wall_radius"], params["wall_spacing"]
    if not spacing < 2 * radius:
        raise ValueError("wall_spacing must be < 2 * wall_radius for a solid wall")
    gap = rng.uniform(*params["gap"])
    c = ctx.layout.obstacle_clearance
    x_lo = float(ctx.start_hi[0]) + radius + c
    x_hi = float(ctx.goal[0] - ctx.goal_radius) - radius - c
    y_lo = float(ctx.lower[1]) + params["gap_edge_margin"] + gap / 2
    y_hi = float(ctx.upper[1]) - params["gap_edge_margin"] - gap / 2
    if x_hi < x_lo or y_hi < y_lo:
        return None
    xw = rng.uniform(x_lo, x_hi)
    # Center the gap near where the start-goal line crosses the wall. A gap far
    # off that line is a trap for agents that sense only ~r around them (like the
    # U-shaped obstacles the doc excludes), not a narrow-passage test.
    s = 0.5 * (ctx.start_lo + ctx.start_hi)
    frac = (xw - s[0]) / (ctx.goal[0] - s[0])
    y_line = s[1] + frac * (ctx.goal[1] - s[1])
    yg = float(np.clip(y_line + rng.uniform(*params["gap_offset"]), y_lo, y_hi))
    ys = wall_circle_ys(yg, gap, radius, spacing, float(ctx.lower[1]), float(ctx.upper[1]))
    centers = np.stack([np.full_like(ys, xw), ys], axis=-1)
    radii = np.full(len(ys), float(radius))
    if not all(ctx.clear(ci, ri) for ci, ri in zip(centers, radii)):
        return None
    return centers, radii


GENERATORS: dict[str, Callable[..., Obstacles | None]] = {
    "open": _open,
    "single": _single,
    "random": _random,
    "wall": _wall,
}


# --------------------------------------------------------------------------- #
# One generation attempt
# --------------------------------------------------------------------------- #
def sample_map(
    scenario: str,
    scenario_cfg: dict,
    n_agents: int,
    world_size,
    sim: SimParams,
    layout: LayoutParams,
    rng_start: np.random.Generator,
    rng_goal: np.random.Generator,
    rng_obstacles: np.random.Generator,
    count_scale: float = 1.0,
) -> MapSpec | None:
    """One attempt at a map; None if placement fails (caller retries).

    Start box, goal and obstacles come from separate random streams. The goal
    radius follows the capacity rule R_g = c * r_sep * sqrt(N).
    """
    kind = scenario_cfg["kind"]
    if kind not in GENERATORS:
        raise ValueError(f"scenario {scenario!r}: unknown kind {kind!r}")
    lower = np.zeros(2)
    upper = np.asarray(world_size, dtype=np.float64)
    goal_radius = sim.goal_radius(n_agents)

    start_lo, start_hi = sample_start_box(rng_start, n_agents, lower, upper, layout)
    goal = sample_goal(rng_goal, goal_radius, lower, upper, layout)
    if point_box_distance(goal, start_lo, start_hi) < goal_radius + layout.min_start_goal_gap:
        return None

    ctx = PlacementContext(lower, upper, start_lo, start_hi, goal, goal_radius, layout, count_scale)
    obstacles = GENERATORS[kind](rng_obstacles, ctx, scenario_cfg)
    if obstacles is None:
        return None
    centers, radii = obstacles
    return MapSpec(
        scenario=scenario,
        n_agents=n_agents,
        lower=lower,
        upper=upper,
        centers=centers,
        radii=radii,
        start_lo=start_lo,
        start_hi=start_hi,
        goal=goal,
        goal_radius=goal_radius,
    )
