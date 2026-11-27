import numpy as np
import pytest

from swarm.feasibility import check_map, clearance_grid, make_grid, path_exists
from swarm.maps import wall_circle_ys

RES = 0.05


def _wall(gap, x=5.0, center=5.0, radius=0.3, spacing=0.3):
    ys = wall_circle_ys(center, gap, radius, spacing, 0.0, 10.0)
    return np.stack([np.full_like(ys, x), ys], -1), np.full(len(ys), radius)


def _solid_wall(x=5.0, radius=0.3, spacing=0.3):
    ys = np.arange(0.0, 10.0 + 1e-9, spacing)
    return np.stack([np.full_like(ys, x), ys], -1), np.full(len(ys), radius)


def test_open_map_is_feasible(make_map, sim):
    rep = check_map(make_map(), sim, RES)
    assert rep.ok and rep.reasons == ()


def test_solid_wall_is_infeasible(make_map, sim):
    c, r = _solid_wall()
    rep = check_map(make_map(centers=c, radii=r), sim, RES)
    assert not rep.ok
    assert "no collision-free path" in rep.reasons[0]


def test_gap_threshold_follows_inflated_agent_size(make_map, sim):
    inflated = 2 * (sim.agent.radius + sim.loss.safety_margin)   # 0.15
    c, r = _wall(gap=inflated - 3 * RES)
    assert not check_map(make_map(centers=c, radii=r), sim, RES).ok
    c, r = _wall(gap=0.35)                                       # narrowest OOD gap
    assert check_map(make_map(centers=c, radii=r), sim, RES).ok


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"goal_radius": 0.5}, "capacity rule"),
        ({"start_lo": (0.01, 3.0)}, "start box leaves"),
        ({"goal": (9.5, 5.0)}, "goal disk leaves"),
        ({"centers": [[3.5, 5.0]], "radii": [0.45]}, "start box"),   # 0.05 from the box
        ({"centers": [[8.5, 7.0]], "radii": [1.0]}, "goal disk"),
    ],
)
def test_specific_rejection_reasons(make_map, sim, kwargs, reason):
    rep = check_map(make_map(**kwargs), sim, RES)
    assert not rep.ok
    assert any(reason in r for r in rep.reasons), rep.reasons


def test_clearance_grid_is_exact_at_cell_centers(make_map):
    m = make_map(centers=[[5.0, 5.0]], radii=[1.0])
    clr, grid = clearance_grid(m, RES)
    assert clr.shape == grid.shape == (200, 200)
    X, Y = grid.mesh()
    expected = np.minimum(
        np.minimum.reduce([X, 10 - X, Y, 10 - Y]), np.hypot(X - 5, Y - 5) - 1.0
    )
    np.testing.assert_allclose(clr, expected)
    no_walls, _ = clearance_grid(m, RES, include_walls=False)
    np.testing.assert_allclose(no_walls, np.hypot(X - 5, Y - 5) - 1.0)


def test_grid_index_picks_nearest_cell(make_map):
    grid = make_grid(make_map(), RES)
    iy, ix = grid.index(np.array([0.0, 9.999]))
    assert (iy, ix) == (199, 0)
    iy, ix = grid.index(np.array([5.01, 2.49]))
    assert grid.xs[ix] == pytest.approx(5.025) and grid.ys[iy] == pytest.approx(2.475)


def test_path_exists_around_obstacle(make_map):
    small = dict(start_lo=(0.5, 4.5), start_hi=(1.0, 5.5), goal=(9.0, 5.0), goal_radius=0.5)
    m = make_map(centers=[[5.0, 5.0]], radii=[3.0], **small)   # big blocker, open above/below
    assert path_exists(m, 0.075, RES)
    # two overlapping disks spanning the full height block every route
    m = make_map(centers=[[5.0, 2.5], [5.0, 7.5]], radii=[2.6, 2.6], **small)
    assert not path_exists(m, 0.075, RES)
