import jax.numpy as jnp
import numpy as np
import pytest

from swarm.config import load_sim_params
from swarm.controllers import ControllerSpec, logits_from_weights
from swarm.metrics import evaluate, summarize, swept_contacts, swept_min_clearance, trajectory_metrics
from swarm.model import Episode, SimSpec
from swarm.rollout import Trajectory, terms

R_A, TOL = 0.05, 1e-9
B_SAFE = jnp.asarray(logits_from_weights([2.0, 1.0, 1.0, 1.5, 3.9], (4.0,) * 5))


def _episode(n, centers=((0.0, 0.0),), radii=(0.0,), valid=(0.0,), goal=(8.0, 5.0), rg=1.0, informed=None):
    return Episode(
        lower=jnp.zeros(2), upper=jnp.array([10.0, 10.0]),
        centers=jnp.array(centers, dtype=float), radii=jnp.array(radii, dtype=float),
        valid=jnp.array(valid, dtype=float), goal=jnp.array(goal), goal_radius=jnp.asarray(rg),
        x0=jnp.zeros((n, 2)), v0=jnp.zeros((n, 2)),
        informed=jnp.ones(n, bool) if informed is None else jnp.asarray(informed),
    )


def _traj(xs, vs=None):
    xs = jnp.asarray(xs, dtype=float)
    T1, n = xs.shape[:2]
    vs = jnp.zeros_like(xs) if vs is None else jnp.asarray(vs, dtype=float)
    return Trajectory(x=xs, v=vs, u=jnp.zeros((T1 - 1, n, 2)), weights=jnp.zeros((T1 - 1, n, 5)),
                      dominant=jnp.zeros((T1 - 1, n), int), features=jnp.zeros((T1 - 1, n, 4)))


def _spec(sim, T):
    return SimSpec(sim, ControllerSpec("fixed"), horizon=T)


# --------------------------------------------------------------------------- #
# Contact geometry (doc §3.1, §9.4)
# --------------------------------------------------------------------------- #
def test_constructed_tangency_counts_only_with_tolerance():
    ep = _episode(2)
    x = jnp.array([[2.0, 2.0], [2.1, 2.0]])           # (2.1 - 2.0)^2 = 0.010000000000000018 > 0.1^2
    assert bool(swept_contacts(x, x, ep, R_A, TOL)[0][0, 1])
    assert not bool(swept_contacts(x, x, ep, R_A, 0.0)[0][0, 1])


def test_separated_agents_and_padded_obstacles_are_not_contacts():
    ep = _episode(2, centers=[[2.0, 2.0]], radii=[1.0], valid=[0.0])
    x = jnp.array([[2.0, 2.0], [2.3, 2.0]])           # second agent sits inside a padded obstacle
    pair, obs, wall = swept_contacts(x, x, ep, R_A, TOL)
    assert not bool(pair.any()) and not bool(obs.any()) and not bool(wall.any())


def test_grazing_crossing_is_detected_and_clearance_is_negative():
    ep = _episode(2)
    x0 = jnp.array([[5.0, 5.0], [4.925, 5.09]])
    x1 = jnp.array([[5.0, 5.0], [5.075, 5.09]])
    assert bool(swept_contacts(x0, x1, ep, R_A, TOL)[0][0, 1])
    assert float(swept_min_clearance(x0, x1, ep, R_A)) == pytest.approx(0.09 - 0.1)


def test_obstacle_and_wall_contacts():
    ep = _episode(2, centers=[[5.0, 5.0]], radii=[1.0], valid=[1.0])
    x = jnp.array([[6.04, 5.0], [0.03, 2.0]])
    _, obs, wall = swept_contacts(x, x, ep, R_A, TOL)
    assert bool(obs[0, 0]) and bool(wall[1]) and not bool(wall[0])


# --------------------------------------------------------------------------- #
# Event counting and episode-level metrics
# --------------------------------------------------------------------------- #
def test_persistent_contact_is_one_event(sim):
    gaps = [0.5, 0.09, 0.09, 0.09, 0.5, 0.5, 0.08, 0.5]   # touch (steps 0-3), part, touch again (5-6)
    xs = [[[5.0, 5.0], [5.0 + g, 5.0]] for g in gaps]
    m = trajectory_metrics(_traj(xs), _episode(2), _spec(sim, len(gaps) - 1))
    assert int(m.pair_events) == 2
    assert int(m.contact_steps) == 6
    assert not bool(m.collision_free)
    assert float(m.min_clearance) == pytest.approx(0.08 - 0.1)


def test_arrival_completion_and_deadlock(sim):
    T = 60
    goal = np.array([8.0, 5.0])
    xs = np.zeros((T + 1, 3, 2))
    xs[:, 0] = [8.0, 5.0]                                  # always inside
    xs[:10, 1] = [3.0, 5.0]
    xs[10:, 1] = [8.5, 5.0]                                # enters at step 10 and stays
    xs[:, 2] = [3.0, 8.0]                                  # never arrives, never moves
    ep = _episode(3, goal=goal, informed=[True, True, False])
    m = trajectory_metrics(_traj(xs), ep, _spec(sim, T))
    assert float(m.arrival) == pytest.approx(2 / 3)
    assert float(m.arrival_informed) == 1.0 and float(m.arrival_uninformed) == 0.0
    assert not bool(m.success)
    assert float(m.deadlock) == pytest.approx(1 / 3)       # the agent at rest outside
    np.testing.assert_allclose(np.asarray(m.arrival_curve)[[0, 9, 10, T]], [1 / 3, 1 / 3, 2 / 3, 2 / 3])
    assert not bool(m.completed) and int(m.completion_step) == T

    sim_low = load_sim_params(overrides={"evaluation": {"success_threshold": 0.6}})
    m = trajectory_metrics(_traj(xs), ep, _spec(sim_low, T))
    assert bool(m.success) and bool(m.completed) and int(m.completion_step) == 10


def test_completion_requires_the_dwell(sim):
    T = 40
    xs = np.zeros((T + 1, 1, 2))
    xs[:, 0] = [3.0, 5.0]
    xs[5:5 + sim.evaluation.completion_dwell - 1, 0] = [8.0, 5.0]   # in the goal one step too short
    xs[30:, 0] = [8.0, 5.0]                                          # re-enters but too late to dwell
    m = trajectory_metrics(_traj(xs), _episode(1), _spec(sim, T))
    assert not bool(m.completed)


def test_uninformed_fields_are_nan_when_everyone_is_informed(sim):
    xs = np.zeros((11, 2, 2))
    m = trajectory_metrics(_traj(xs), _episode(2), _spec(sim, 10))
    assert np.isnan(float(m.arrival_uninformed))
    assert np.all(np.isnan(np.asarray(m.arrival_curve_uninformed)))


# --------------------------------------------------------------------------- #
# Full evaluation
# --------------------------------------------------------------------------- #
def test_evaluate_matches_training_loss_and_chunks(make_episode, sim):
    eps = Episode(*(jnp.stack(f) for f in zip(
        make_episode(centers=[[5.0, 5.0]], radii=[0.8]), make_episode(seed=1), make_episode(seed=2))))
    spec = SimSpec(sim, ControllerSpec("fixed"), horizon=200)
    m = evaluate(B_SAFE, eps, spec)
    np.testing.assert_allclose(np.mean(m.loss.total), terms(B_SAFE, eps, spec).total, rtol=1e-12)
    m2 = evaluate(B_SAFE, eps, spec, chunk=2)
    for a, b in zip(m.arrival, m2.arrival):
        assert float(a) == float(b)
    s = summarize(m, spec)
    assert s["episodes"] == 3 and 0.0 <= s["success_rate"] <= 1.0
    assert s["arrival"] > 0.9                              # an easy open map: the swarm gets there


def test_halving_dt_preserves_outcomes(make_episode, sim):
    """Doc §9.4: dt/2 over the same physical time should not change the conclusions."""
    eps = Episode(*(jnp.stack(f) for f in zip(make_episode(centers=[[5.0, 5.0]], radii=[0.8]),
                                              make_episode(seed=4, centers=[[5.5, 4.0]], radii=[0.6]))))
    T = 300
    half = load_sim_params(overrides={"simulation": {"dt": sim.simulation.dt / 2}})
    m1 = evaluate(B_SAFE, eps, SimSpec(sim, ControllerSpec("fixed"), horizon=T))
    m2 = evaluate(B_SAFE, eps, SimSpec(half, ControllerSpec("fixed"), horizon=2 * T))
    np.testing.assert_allclose(m1.arrival, m2.arrival, atol=0.05)
    np.testing.assert_array_equal(m1.collision_free, m2.collision_free)
    t1 = np.asarray(m1.completion_step) * sim.simulation.dt
    t2 = np.asarray(m2.completion_step) * half.simulation.dt
    np.testing.assert_allclose(t1, t2, rtol=0.1)
    np.testing.assert_allclose(m1.min_clearance, m2.min_clearance, atol=0.03)
