import jax
import jax.numpy as jnp
import numpy as np
import pytest

from swarm.features import FEATURE_NAMES, local_features
from swarm.forces import goal_force
from swarm.geometry import unit_direction
from swarm.observation import local_goal_observation
from swarm.perception import neighborhood

GOAL, RG = jnp.array([8.0, 5.0]), 1.3


def _obs(sim, x, v, informed, goal=GOAL, rg=RG, gamma_missing=0.0, hard=True):
    a = neighborhood(x, 1.0, 0.3, 0.4, hard).a
    return local_goal_observation(x, v, goal, rg, informed, a, sim, gamma_missing)


def _informed(n, k):
    return jnp.arange(n) < k


# --------------------------------------------------------------------------- #
# Partially informed observation (doc §9.4 P1-P3b)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hard", [True, False])
def test_p1_uninformed_observation_ignores_the_goal(sim, random_state, hard):
    x, v = random_state(n=12)
    h = _informed(12, 3)
    a = _obs(sim, x, v, h, hard=hard)
    b = _obs(sim, x, v, h, goal=jnp.array([1.0, 9.0]), rg=2.0, hard=hard)
    for fa, fb in zip(a, b):
        np.testing.assert_array_equal(fa[3:], fb[3:])               # uninformed: identical
    assert float(jnp.max(jnp.abs(a.goal_force[:3] - b.goal_force[:3]))) > 0.1   # informed: changed


def test_p2_all_informed_is_the_fully_informed_model(sim, random_state):
    x, v = random_state(n=10)
    o = _obs(sim, x, v, jnp.ones(10, dtype=bool))
    ell = sim.goal.length_scale_factor * RG
    np.testing.assert_allclose(o.goal_force, goal_force(x, v, GOAL, sim.agent.v_max, ell))
    np.testing.assert_allclose(o.heading, unit_direction(GOAL - x))
    dist = jnp.linalg.norm(GOAL - x, axis=-1)
    np.testing.assert_allclose(o.gamma, jnp.tanh(jnp.maximum(dist - RG, 0) / 2.0), atol=1e-6)


def test_p3_uninformed_gamma_equals_arrived_gamma(sim):
    x = jnp.array([[8.2, 5.1], [3.0, 3.0]])                     # informed in goal, uninformed far
    o = _obs(sim, x, jnp.zeros((2, 2)), jnp.array([True, False]))
    assert float(o.gamma[0]) == 0.0 and float(o.gamma[1]) == 0.0
    np.testing.assert_array_equal(o.goal_force[1], 0.0)


def test_p3b_missing_value_changes_uninformed_gamma_only(sim, random_state):
    x, v = random_state(n=8)
    h = _informed(8, 2)
    g0, g1 = _obs(sim, x, v, h, gamma_missing=0.0), _obs(sim, x, v, h, gamma_missing=1.0)
    np.testing.assert_array_equal(g0.gamma[:2], g1.gamma[:2])
    np.testing.assert_array_equal(g1.gamma[2:], 1.0)
    np.testing.assert_array_equal(g0.gamma[2:], 0.0)


def test_uninformed_heading_is_local_mean_velocity(sim):
    x = jnp.array([[3.0, 3.0], [3.4, 3.0], [9.0, 9.0]])
    v = jnp.array([[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    o = _obs(sim, x, v, jnp.array([False, False, False]))
    np.testing.assert_allclose(o.heading[0], [0.0, 1.0], atol=1e-6)
    np.testing.assert_array_equal(o.heading[2], 0.0)               # at rest, alone: zero heading
    g = jax.grad(lambda x: jnp.sum(_obs(sim, x, v, jnp.zeros(3, bool), hard=False).heading))(x)
    assert bool(jnp.all(jnp.isfinite(g)))


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def test_features_shape_and_formulas():
    count = jnp.array([0.0, 5.0, 50.0])
    d_near = jnp.array([jnp.inf, 0.3, -0.2])
    d_wall = jnp.array([4.0, 2.0, 1.0])
    gamma = jnp.array([0.0, 0.5, 0.9])
    coh = jnp.array([[0.0, 0.0], [0.3, 0.4], [0.0, 0.0]])
    z = local_features(count, d_near, d_wall, gamma, coh, 1.0, 5.0)
    assert z.shape == (3, len(FEATURE_NAMES))
    np.testing.assert_allclose(z[:, 0], jnp.tanh(count / 5.0))
    np.testing.assert_allclose(z[:, 1], [1.0, 0.3, 0.0])        # none in range / near / inside
    np.testing.assert_allclose(z[:, 2], gamma)
    np.testing.assert_allclose(z[:, 3], [0.0, 0.5, 0.0], atol=1e-6)


def test_features_stay_in_unit_box_in_degenerate_states():
    d_near = jnp.array([-5.0, jnp.inf, 0.1])                    # inside obstacle, none, near
    d_wall = jnp.array([-1.0, -0.3, 30.0])                      # outside domain twice
    z = local_features(jnp.array([0.0, 3.0, 100.0]), d_near, d_wall, jnp.zeros(3),
                       jnp.array([[0.9, 0.0], [0.0, 0.0], [0.1, 0.1]]), 1.0, 5.0)
    assert bool(jnp.all((z >= 0) & (z <= 1)))
    g = jax.grad(lambda c: jnp.sum(local_features(jnp.zeros(3), d_near, d_wall, jnp.zeros(3), c, 1.0, 5.0)))(
        jnp.zeros((3, 2)))
    assert bool(jnp.all(jnp.isfinite(g)))
