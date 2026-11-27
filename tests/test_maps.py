import math

import numpy as np
import pytest

from swarm.datasets import SplitSpec, generate_feasible_map
from swarm.feasibility import check_map
from swarm.maps import (
    SCENARIOS,
    MapSpec,
    _scaled_count_range,
    point_box_distance,
    wall_circle_ys,
)


def _spec(n_agents=30, setting="fixed_area", name="test_maps"):
    return SplitSpec(name=name, mixture="train_id", n_maps=1, episodes_per_map=1,
                     n_agents=n_agents, size_setting=setting)


def _maps(scenario, sim, cfg, n=6, **spec_kw):
    spec = _spec(**spec_kw)
    return [generate_feasible_map(spec, i, scenario, sim, cfg)[0] for i in range(n)]


def _wall_gaps(m: MapSpec) -> np.ndarray:
    """Surface-to-surface gaps between consecutive circles of a vertical wall."""
    ys = np.sort(m.centers[:, 1])
    return np.diff(ys) - 2 * m.radii[0]


# --------------------------------------------------------------------------- #
# Every scenario
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_generated_maps_respect_layout_rules(scenario, sim, data_cfg):
    layout = data_cfg.layout
    params = data_cfg.scenarios[scenario]
    for m in _maps(scenario, sim, data_cfg):
        assert m.scenario == scenario
        assert check_map(m, sim, data_cfg.feasibility_resolution).ok
        assert m.goal_radius == pytest.approx(sim.goal_radius(m.n_agents))
        # start box and goal stay inside the border margin
        assert np.all(m.start_lo >= m.lower + layout.border_margin - 1e-12)
        assert np.all(m.start_hi <= m.upper - layout.border_margin + 1e-12)
        assert np.all(m.goal - m.goal_radius >= m.lower + layout.border_margin - 1e-12)
        assert np.all(m.goal + m.goal_radius <= m.upper - layout.border_margin + 1e-12)
        # start box and goal are well separated
        assert point_box_distance(m.goal, m.start_lo, m.start_hi) >= m.goal_radius + layout.min_start_goal_gap
        # obstacles keep their clearance from start box and goal
        if m.n_obstacles:
            c = layout.obstacle_clearance
            assert np.all(point_box_distance(m.centers, m.start_lo, m.start_hi) - m.radii >= c - 1e-12)
            assert np.all(np.linalg.norm(m.centers - m.goal, axis=1) - m.radii >= m.goal_radius + c - 1e-12)
        # scenario-specific ranges
        kind = params["kind"]
        if kind == "open":
            assert m.n_obstacles == 0
        elif kind == "single":
            assert m.n_obstacles == 1
            assert params["radius"][0] <= m.radii[0] <= params["radius"][1]
        elif kind == "random":
            assert params["count"][0] <= m.n_obstacles <= params["count"][1]
            assert np.all((m.radii >= params["radius"][0]) & (m.radii <= params["radius"][1]))
        elif kind == "wall":
            gaps = _wall_gaps(m)
            assert params["gap"][0] - 1e-9 <= gaps.max() <= params["gap"][1] + 1e-9
            assert np.all(np.delete(gaps, gaps.argmax()) < 0)   # every other pair overlaps
            assert len(np.unique(m.centers[:, 0])) == 1           # a straight vertical wall
            # the gap sits near where the start-goal line crosses the wall (or at a clip limit)
            ys = np.sort(m.centers[:, 1])
            k = int(np.argmax(np.diff(ys)))
            gap_center = 0.5 * (ys[k] + ys[k + 1])
            s, xw = m.start_center, m.centers[0, 0]
            y_line = s[1] + (xw - s[0]) / (m.goal[0] - s[0]) * (m.goal[1] - s[1])
            half = 0.5 * (ys[k + 1] - ys[k]) - params["wall_radius"]
            lo = m.lower[1] + params["gap_edge_margin"] + half
            hi = m.upper[1] - params["gap_edge_margin"] - half
            off_lo, off_hi = params["gap_offset"]
            clipped = np.isclose(gap_center, lo) or np.isclose(gap_center, hi)
            assert clipped or off_lo - 1e-9 <= gap_center - y_line <= off_hi + 1e-9


def test_start_box_area_matches_target_fraction(sim, data_cfg):
    layout = data_cfg.layout
    for n in (20, 30, 100):
        for m in _maps("open", sim, data_cfg, n=5, n_agents=n):
            phi = n * math.pi * layout.init_spacing**2 / 4 / m.start_area
            assert layout.start_area_fraction[0] - 1e-9 <= phi <= layout.start_area_fraction[1] + 1e-9
            # width is capped unless the box already spans the full usable height
            width = m.start_hi[0] - m.start_lo[0]
            span_y = m.size[1] - 2 * layout.border_margin
            cap = max(layout.max_start_width_fraction * m.size[0], m.start_area / span_y)
            assert width <= cap + 1e-9


def test_generation_is_deterministic_and_index_dependent(sim, data_cfg):
    spec = _spec()
    a, _ = generate_feasible_map(spec, 3, "random_obstacles", sim, data_cfg)
    b, _ = generate_feasible_map(spec, 3, "random_obstacles", sim, data_cfg)
    c, _ = generate_feasible_map(spec, 4, "random_obstacles", sim, data_cfg)
    for field in ("centers", "radii", "start_lo", "goal"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert not np.array_equal(a.goal, c.goal)


def test_fixed_density_maps_scale_domain(sim, data_cfg):
    (m,) = _maps("random_obstacles", sim, data_cfg, n=1, n_agents=120, setting="fixed_density")
    np.testing.assert_allclose(m.size, np.asarray(sim.world.size) * 2.0)


# --------------------------------------------------------------------------- #
# Wall construction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gap,center", [(0.35, 5.0), (1.5, 2.0), (0.8, 8.3)])
def test_wall_circle_ys_leaves_exact_gap_and_no_leaks(gap, center):
    radius, spacing = 0.3, 0.3
    ys = wall_circle_ys(center, gap, radius, spacing, 0.0, 10.0)
    d = np.diff(ys)
    assert ys[0] == 0.0 and ys[-1] == 10.0              # chains end on the boundaries
    k = int(np.argmax(d))
    assert d[k] - 2 * radius == pytest.approx(gap)      # exactly one gap, of the right width
    assert ys[k] < center < ys[k + 1]
    others = np.delete(d, k)
    assert np.all(others <= spacing + 1e-12) and np.all(others < 2 * radius)  # overlapping


# --------------------------------------------------------------------------- #
# OOD definitions really are out of distribution
# --------------------------------------------------------------------------- #
def test_ood_scenarios_lie_outside_training_ranges(data_cfg):
    sc = data_cfg.scenarios
    train_radius_max = max(sc[n]["radius"][1] for n in data_cfg.mixtures["train_id"] if "radius" in sc[n])
    assert sc["large_obstacles"]["radius"][0] > train_radius_max
    assert sc["dense_random"]["count"][0] > sc["random_obstacles"]["count"][1]
    assert sc["narrow_bottleneck"]["gap"][1] <= sc["bottleneck"]["gap"][0]
    assert set(data_cfg.mixtures["ood"]).isdisjoint(data_cfg.mixtures["train_id"])


def test_scaled_count_range():
    assert _scaled_count_range([3, 8], 1.0) == (3, 8)
    assert _scaled_count_range([3, 8], 4.0) == (12, 32)
    assert _scaled_count_range([3, 8], 0.1) == (1, 1)


# --------------------------------------------------------------------------- #
# MapSpec container
# --------------------------------------------------------------------------- #
def test_mapspec_arrays_are_read_only_float64(make_map):
    m = make_map(centers=[[5, 5]], radii=[1.0])
    assert m.centers.dtype == np.float64 and m.centers.shape == (1, 2)
    with pytest.raises(ValueError):
        m.centers[0, 0] = 1.0
    assert make_map().centers.shape == (0, 2)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"scenario": "maze"}, "unknown scenario"),
        ({"centers": [[1, 1], [2, 2]], "radii": [1.0]}, "obstacle count"),
        ({"centers": [[1, 1]], "radii": [-1.0]}, "positive"),
        ({"start_lo": (2.0, 4.0), "start_hi": (1.0, 6.0)}, "hi > lo"),
    ],
)
def test_mapspec_validation(make_map, kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_map(**kwargs)


def test_point_box_distance():
    lo, hi = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    assert point_box_distance([0.5, 0.5], lo, hi) == 0.0
    assert point_box_distance([2.0, 0.5], lo, hi) == pytest.approx(1.0)
    assert point_box_distance([4.0, 5.0], lo, hi) == pytest.approx(5.0)
    np.testing.assert_allclose(point_box_distance(np.array([[2.0, 0.5], [-1.0, -1.0]]), lo, hi),
                               [1.0, math.sqrt(2)])
